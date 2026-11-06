"""Three-stage latent flow orchestration; trained encoders/models remain external."""
from dataclasses import dataclass
import math
import operator
import torch
from torch import nn
from .flow import SparseFlowDiT,flow_loss,sample_flow
from .serialization import serialization_orders


@dataclass
class GeneratedLatents:
    coords: torch.Tensor
    shape: torch.Tensor
    material: torch.Tensor
    rig: torch.Tensor | None


class ThreeStageFlow(nn.Module):
    """Occupancy -> geometry/optional co-located rig -> shape-conditioned material.

    One asset per call. Occupancy is a continuous flow over a bounded complete
    coarse grid, thresholded at inference. Geometry/material use sparse support.
    Separate skeleton occupancy/adapters and prompt encoders are not implemented.
    """
    def __init__(self,shape_channels,material_channels,condition_channels,*,grid_size=(64,64,64),
                 rig_channels=0,width=96,heads=4,depth=4,patch_size=256,max_grid_tokens=262144):
        super().__init__()
        grid_size=tuple(operator.index(v) for v in grid_size)
        if len(grid_size)!=3 or min(grid_size)<=0 or math.prod(grid_size)>max_grid_tokens:
            raise ValueError('Coarse grid must be positive and within the token budget')
        if shape_channels<=0 or material_channels<=0 or rig_channels<0:
            raise ValueError('Invalid latent channel dimensions')
        self.grid_size=grid_size
        self.shape_channels,self.material_channels,self.rig_channels=shape_channels,material_channels,rig_channels
        options=dict(width=width,heads=heads,depth=depth,patch_size=patch_size)
        self.occupancy=SparseFlowDiT(1,condition_channels,**options)
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

    def losses(self,coords,shape,material,condition,*,rig=None,drop_probability=.1):
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
        return dict(occupancy=flow_loss(self.occupancy,occupancy,self.grid_coords,condition,
                     orders=self._orders(self.grid_coords),drop_probability=drop_probability),
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
        occupancy=sample_flow(self.occupancy,noise(len(self.grid_coords),1),self.grid_coords,condition,
                              steps=steps,guidance=guidance,orders=self._orders(self.grid_coords))
        coords=self.grid_coords[occupancy[:,0]>occupancy_threshold]
        if not len(coords):
            return GeneratedLatents(coords,noise(0,self.shape_channels),noise(0,self.material_channels),
                                     noise(0,self.rig_channels) if self.rig_channels else None)
        orders=self._orders(coords)
        geometry=sample_flow(self.geometry,noise(len(coords),self.shape_channels+self.rig_channels),
                             coords,condition,steps=steps,guidance=guidance,orders=orders)
        shape=geometry[:,:self.shape_channels]
        material=sample_flow(self.material,noise(len(coords),self.material_channels),coords,condition,
                              steps=steps,guidance=guidance,orders=orders,shape=shape)
        return GeneratedLatents(coords,shape,material,geometry[:,self.shape_channels:] if self.rig_channels else None)
