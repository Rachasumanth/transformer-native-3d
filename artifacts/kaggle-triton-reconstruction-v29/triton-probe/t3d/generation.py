"""Three-stage latent flow orchestration; trained encoders/models remain external."""
from dataclasses import dataclass
import math
import operator
import torch
from torch import nn
from .flow import SparseFlowDiT,flow_loss,sample_flow
from .coupled_occupancy import CoupledOccupancyFlow,containment_mask
from .serialization import serialization_orders


@dataclass
class GeneratedLatents:
    coords: torch.Tensor
    shape: torch.Tensor
    material: torch.Tensor
    rig: torch.Tensor | None
    skeleton_coords: torch.Tensor | None = None


class ThreeStageFlow(nn.Module):
    """Occupancy -> geometry/optional co-located rig -> shape-conditioned material.

    One asset per call. Occupancy is a continuous flow over a bounded complete
    coarse grid, thresholded at inference. Geometry/material use sparse support.
    Optional coupled skeleton occupancy is supported; prompt encoders remain external.
    """
    def __init__(self,shape_channels,material_channels,condition_channels,*,grid_size=(64,64,64),
                 rig_channels=0,width=96,heads=4,depth=4,patch_size=256,max_grid_tokens=262144,
                 separate_skeleton=False,skeleton_containment='aabb',absolute_occupancy_coords=False):
        super().__init__()
        grid_size=tuple(operator.index(v) for v in grid_size)
        if len(grid_size)!=3 or min(grid_size)<=0 or math.prod(grid_size)>max_grid_tokens:
            raise ValueError('Coarse grid must be positive and within the token budget')
        if type(separate_skeleton) is not bool or type(absolute_occupancy_coords) is not bool:raise ValueError('Occupancy flags must be boolean')
        if absolute_occupancy_coords and not separate_skeleton:raise ValueError('Absolute occupancy coordinates require separate skeleton')
        if skeleton_containment not in ('aabb','enclosed'):raise ValueError('Unknown skeleton containment')
        if not separate_skeleton and skeleton_containment!='aabb':raise ValueError('Containment requires separate skeleton')
        if shape_channels<=0 or material_channels<=0 or rig_channels<0:
            raise ValueError('Invalid latent channel dimensions')
        self.grid_size=grid_size
        self.shape_channels,self.material_channels,self.rig_channels=shape_channels,material_channels,rig_channels
        options=dict(width=width,heads=heads,depth=depth,patch_size=patch_size)
        self.separate_skeleton=separate_skeleton;self.skeleton_containment=skeleton_containment
        self.occupancy=(CoupledOccupancyFlow(condition_channels,absolute_coords=absolute_occupancy_coords,**options) if separate_skeleton
                        else SparseFlowDiT(1,condition_channels,**options))
        self.geometry=SparseFlowDiT(shape_channels+rig_channels,condition_channels,**options)
        self.material=SparseFlowDiT(material_channels,condition_channels,shape_channels=shape_channels,**options)
        self.register_buffer('grid_coords',torch.cartesian_prod(*(torch.arange(n) for n in grid_size)))

    def _orders(self,coords):
        depth=max(1,(max(self.grid_size)-1).bit_length())
        # Build integer permutations once per stage; reuse for all flow steps.
        mappings=serialization_orders(coords.detach().cpu().numpy(),depth=depth)
        return [torch.as_tensor(order,device=coords.device,dtype=torch.int64) for order,_ in mappings.values()]

    def _validate_support(self,coords):
        if (coords.ndim!=2 or coords.shape[1]!=3 or coords.dtype!=torch.int64
                or coords.device!=self.grid_coords.device or not len(coords)
                or (coords<0).any() or (coords>=coords.new_tensor(self.grid_size)).any()
                or len(torch.unique(coords,dim=0))!=len(coords)):
            raise ValueError('Expected unique nonempty in-grid int64 coordinates on model device')

    def losses(self,coords,shape,material,condition,*,rig=None,skeleton_coords=None,drop_probability=.1):
        self._validate_support(coords)
        expected=[(shape,self.shape_channels),(material,self.material_channels)]
        if self.rig_channels:
            if rig is None:raise ValueError('Rig channels require rig training targets')
            expected.append((rig,self.rig_channels))
        elif rig is not None:raise ValueError('Model has no rig branch')
        for values,channels in expected:
            if values.shape!=(len(coords),channels) or not values.is_floating_point() or not torch.isfinite(values).all():
                raise ValueError('Latents must be finite aligned (N,D) tensors')
        occupancy=shape.new_zeros((len(self.grid_coords),1))
        indices=(coords[:,0]*self.grid_size[1]+coords[:,1])*self.grid_size[2]+coords[:,2]
        occupancy[indices]=1
        orders=self._orders(coords)
        if self.separate_skeleton:
            if skeleton_coords is None:raise ValueError('Separate skeleton branch requires skeleton coordinates')
            if skeleton_coords.ndim!=2 or skeleton_coords.shape[1]!=3 or skeleton_coords.dtype!=torch.int64 or skeleton_coords.device!=coords.device:
                raise ValueError('Skeleton coordinates must be same-device int64 (N,3)')
            if len(skeleton_coords):
                self._validate_support(skeleton_coords)
                skeleton_indices=(skeleton_coords[:,0]*self.grid_size[1]+skeleton_coords[:,1])*self.grid_size[2]+skeleton_coords[:,2]
                if not torch.all(containment_mask(self.grid_coords,occupancy[:,0]>0,self.skeleton_containment)[skeleton_indices]):raise ValueError('Skeleton occupancy violates chosen containment mask')
            else:skeleton_indices=indices[:0]
            skeleton=occupancy.new_zeros(occupancy.shape);skeleton[skeleton_indices]=1
            paired=self.occupancy.loss(occupancy,skeleton,self.grid_coords,condition,
                orders=self._orders(self.grid_coords),drop_probability=drop_probability,containment=self.skeleton_containment)
            occupancy_losses=dict(occupancy=paired['shape'],skeleton_occupancy=paired['skeleton'])
        else:
            if skeleton_coords is not None:raise ValueError('Model has no separate skeleton branch')
            occupancy_losses=dict(occupancy=flow_loss(self.occupancy,occupancy,self.grid_coords,condition,
                     orders=self._orders(self.grid_coords),drop_probability=drop_probability))
        return dict(**occupancy_losses,
                    geometry=flow_loss(self.geometry,torch.cat((shape,rig),dim=1) if rig is not None else shape,
                     coords,condition,orders=orders,drop_probability=drop_probability),
                    material=flow_loss(self.material,material,coords,condition,orders=orders,
                     shape=shape,drop_probability=drop_probability))

    @torch.no_grad()
    def generate(self,condition,*,steps=50,guidance=3.,occupancy_threshold=.5,generator=None):
        if not math.isfinite(occupancy_threshold) or not 0<=occupancy_threshold<=1:
            raise ValueError('Occupancy threshold must lie in [0,1]')
        parameter=next(self.parameters(),None)
        dtype=torch.float32 if parameter is None else parameter.dtype
        def noise(n,c):return torch.randn(n,c,device=self.grid_coords.device,dtype=dtype,generator=generator)
        if self.separate_skeleton:
            shape_mask,skeleton_mask=self.occupancy.sample(noise(len(self.grid_coords),1),noise(len(self.grid_coords),1),
                self.grid_coords,condition,steps=steps,guidance=guidance,orders=self._orders(self.grid_coords),threshold=occupancy_threshold,containment=self.skeleton_containment)
            coords=self.grid_coords[shape_mask];skeleton_coords=self.grid_coords[skeleton_mask]
        else:
            occupancy=sample_flow(self.occupancy,noise(len(self.grid_coords),1),self.grid_coords,condition,
                                  steps=steps,guidance=guidance,orders=self._orders(self.grid_coords))
            coords=self.grid_coords[occupancy[:,0]>occupancy_threshold];skeleton_coords=None
        if not len(coords):
            return GeneratedLatents(coords,noise(0,self.shape_channels),noise(0,self.material_channels),
                                     noise(0,self.rig_channels) if self.rig_channels else None,skeleton_coords)
        orders=self._orders(coords)
        geometry=sample_flow(self.geometry,noise(len(coords),self.shape_channels+self.rig_channels),
                             coords,condition,steps=steps,guidance=guidance,orders=orders)
        shape=geometry[:,:self.shape_channels]
        material=sample_flow(self.material,noise(len(coords),self.material_channels),coords,condition,
                              steps=steps,guidance=guidance,orders=orders,shape=shape)
        return GeneratedLatents(coords,shape,material,geometry[:,self.shape_channels:] if self.rig_channels else None,skeleton_coords)
