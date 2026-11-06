"""Coupled shape/skeleton occupancy flow with block-level cross-branch adapters.

One asset per call. This is a small trainable reference; pretrained image/text
encoders, production-scale training and geometric bone containment are separate.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .flow import SparseFlowDiT, flow_pair


def shape_bounding_box_mask(coords, shape_mask):
    """Mask grid coordinates in the occupied shape AABB; empty shape gives empty support."""
    if coords.ndim!=2 or coords.shape[1]!=3 or shape_mask.shape!=(len(coords),) or shape_mask.dtype!=torch.bool:
        raise ValueError('Expected coordinates (N,3) and boolean shape mask')
    if not shape_mask.any():return torch.zeros_like(shape_mask)
    occupied=coords[shape_mask];lo=occupied.min(0).values;hi=occupied.max(0).values
    return ((coords>=lo)&(coords<=hi)).all(1)


class CoupledOccupancyFlow(nn.Module):
    """Two occupancy DiTs sharing spatial support and conditioning tokens."""
    def __init__(self, condition_channels, *, width=96, heads=4, depth=4, patch_size=256):
        super().__init__()
        options=dict(channels=1,condition_channels=condition_channels,width=width,
                     heads=heads,depth=depth,patch_size=patch_size)
        self.shape=SparseFlowDiT(**options);self.skeleton=SparseFlowDiT(**options)
        self.shape_to_skeleton=nn.ModuleList(nn.Linear(width,width) for _ in range(depth))
        self.skeleton_to_shape=nn.ModuleList(nn.Linear(width,width) for _ in range(depth))
        for layer in (*self.shape_to_skeleton,*self.skeleton_to_shape):
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    @staticmethod
    def _contexts(branch,time,condition):
        width=branch.width;half=width//2
        freq=torch.exp(-math.log(10000)*torch.arange(half,device=time.device,dtype=time.dtype)/max(1,half-1))
        phase=1000*time*freq;context=branch.time(torch.cat((phase.cos(),phase.sin())))
        prompt=branch.null_prompt if condition is None else branch.prompt(condition)
        if prompt.ndim!=2 or not len(prompt):raise ValueError('Condition must contain prompt tokens')
        return context,prompt

    def forward(self,shape,skeleton,coords,time,condition=None,*,orders=None):
        if (shape.ndim!=2 or shape.shape[1:]!=(1,) or shape.shape!=skeleton.shape or not len(shape)
                or coords.shape!=(len(shape),3) or shape.device!=skeleton.device or shape.device!=coords.device
                or not torch.isfinite(shape).all() or not torch.isfinite(skeleton).all()):
            raise ValueError('Matching finite (N,1) streams and same-device (N,3) coordinates required')
        time=torch.as_tensor(time,device=shape.device,dtype=shape.dtype)
        if time.numel()!=1 or not torch.isfinite(time).all() or not 0<=time.item()<=1:raise ValueError('Scalar time in [0,1] required')
        if orders is None:orders=[torch.arange(len(shape),device=shape.device)]
        if not orders:raise ValueError('At least one serialization order required')
        expected=torch.arange(len(shape),device=shape.device)
        for order in orders:
            if order.device!=shape.device or order.dtype!=torch.int64 or not torch.equal(order.sort().values,expected):
                raise ValueError('Orders must be same-device permutations')
        sc,sp=self._contexts(self.shape,time,condition);kc,kp=self._contexts(self.skeleton,time,condition)
        s=self.shape.input(shape);k=self.skeleton.input(skeleton)
        for i,(sb,kb) in enumerate(zip(self.shape.blocks,self.skeleton.blocks)):
            order=orders[i%len(orders)]
            next_s=sb(s,coords,sc,sp,order);next_k=kb(k,coords,kc,kp,order)
            s=next_s+self.skeleton_to_shape[i](next_k)
            k=next_k+self.shape_to_skeleton[i](next_s)
        return self.shape.output(self.shape.norm(s)),self.skeleton.output(self.skeleton.norm(k))

    def loss(self,shape,skeleton,coords,condition=None,*,orders=None,drop_probability=.1):
        if not 0<=drop_probability<=1:raise ValueError('Invalid conditioning dropout')
        if (shape.shape!=skeleton.shape or shape.ndim!=2 or shape.shape[1:]!=(1,)
                or not torch.isfinite(shape).all() or not torch.isfinite(skeleton).all()
                or not torch.all((shape==0)|(shape==1)) or not torch.all((skeleton==0)|(skeleton==1))
                ):
            raise ValueError('Binary shape and skeleton occupancy required')
        if torch.any((skeleton[:,0]>0)&~shape_bounding_box_mask(coords,shape[:,0]>0)):
            raise ValueError('Skeleton occupancy must lie within shape bounding box')
        time=torch.rand((),device=shape.device,dtype=shape.dtype)
        noisy_s,target_s=flow_pair(shape,torch.randn_like(shape),time)
        noisy_k,target_k=flow_pair(skeleton,torch.randn_like(skeleton),time)
        prompt=None if torch.rand((),device=shape.device)<drop_probability else condition
        pred_s,pred_k=self(noisy_s,noisy_k,coords,time,prompt,orders=orders)
        return {'shape':F.mse_loss(pred_s,target_s),'skeleton':F.mse_loss(pred_k,target_k)}

    @torch.no_grad()
    def sample(self,shape_noise,skeleton_noise,coords,condition=None,*,orders=None,steps=50,guidance=3.,threshold=.5):
        if type(steps) is not int or steps<=0 or not math.isfinite(guidance) or not 0<=threshold<=1:
            raise ValueError('Valid steps, guidance and threshold required')
        s=shape_noise.clone();k=skeleton_noise.clone()
        for i in range(steps):
            time=s.new_tensor(i/steps);cs,ck=self(s,k,coords,time,condition,orders=orders)
            us,uk=self(s,k,coords,time,None,orders=orders)
            s=s+(us+guidance*(cs-us))/steps;k=k+(uk+guidance*(ck-uk))/steps
        shape_mask=s[:,0]>threshold
        skeleton_mask=(k[:,0]>threshold)&shape_bounding_box_mask(coords,shape_mask)
        return shape_mask,skeleton_mask
