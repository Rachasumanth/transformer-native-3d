"""Rectified-flow sparse DiT foundations; conditioning encoders/training pending.

One asset per forward pass. Caller-supplied integer serialization permutations
can cycle Z/TZ/H/TH across layers; attention gathers only each bounded patch.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def flow_pair(clean,noise,time):
    if clean.shape!=noise.shape or time.numel()!=1 or not 0<=time.item()<=1:
        raise ValueError('Matching tensors and scalar time in [0,1] required')
    return (1-time)*noise+time*clean,clean-noise


def _unit(vector,fallback):
    norm=vector.norm(dim=-1,keepdim=True)
    return torch.where(norm>1e-6,vector/norm.clamp_min(1e-6),fallback)


class GramSchmidtRoPE(nn.Module):
    def __init__(self,heads,head_dim):
        super().__init__()
        if head_dim%6:raise ValueError('3D axial RoPE head dimension must be divisible by six')
        initial=torch.tensor([[1.,0,0],[0,1.,0]]).repeat(heads,1,1)
        self.axes=nn.Parameter(initial)
        self.register_buffer('frequencies',10000.**(-torch.arange(head_dim//6,dtype=torch.float32)/max(1,head_dim//6)))

    def rotation(self):
        u=_unit(self.axes[:,0],self.axes.new_tensor([1.,0,0]).expand(len(self.axes),3))
        raw=self.axes[:,1]-(self.axes[:,1]*u).sum(-1,keepdim=True)*u
        basis=F.one_hot(u.abs().argmin(-1),3).to(u.dtype)
        fallback=_unit(basis-(basis*u).sum(-1,keepdim=True)*u,basis)
        v=_unit(raw,fallback)
        return torch.stack((u,v,torch.cross(u,v,dim=-1)),dim=1)

    def forward(self,value,coords):
        angles=torch.einsum('nj,haj->hna',coords.to(value.dtype),self.rotation().to(value.dtype))
        angles=angles[:,:,:,None]*self.frequencies.to(value.dtype)
        paired=value.reshape(value.shape[0],value.shape[1],3,-1,2)
        a,b=paired[...,0],paired[...,1]
        return torch.stack((a*angles.cos()-b*angles.sin(),a*angles.sin()+b*angles.cos()),dim=-1).flatten(2)


class SerializedAttention(nn.Module):
    def __init__(self,width,heads,patch_size):
        super().__init__()
        if width%heads or patch_size<=0:raise ValueError('Invalid attention dimensions')
        self.heads,self.patch_size=heads,patch_size
        self.qkv=nn.Linear(width,3*width)
        self.rope=GramSchmidtRoPE(heads,width//heads)
        self.output=nn.Linear(width,width)

    def forward(self,x,coords,order):
        n,width=x.shape
        q,k,v=self.qkv(x).reshape(n,3,self.heads,width//self.heads).permute(1,2,0,3).unbind(0)
        q,k=self.rope(q,coords),self.rope(k,coords)
        chunks=[]
        for start in range(0,n,self.patch_size):
            index=order[start:start+self.patch_size]
            attention=F.scaled_dot_product_attention(q[:,index],k[:,index],v[:,index])
            chunks.append(attention.transpose(0,1).reshape(len(index),width))
        return self.output(torch.cat(chunks)[torch.argsort(order)])


class FlowBlock(nn.Module):
    def __init__(self,width,heads,patch_size):
        super().__init__()
        self.norm1=nn.LayerNorm(width,elementwise_affine=False)
        self.norm2=nn.LayerNorm(width,elementwise_affine=False)
        self.attention=SerializedAttention(width,heads,patch_size)
        self.cross=nn.MultiheadAttention(width,heads,batch_first=True)
        self.ffn=nn.Sequential(nn.Linear(width,4*width),nn.GELU(),nn.Linear(4*width,width))
        self.modulation=nn.Linear(width,6*width)
        nn.init.zeros_(self.modulation.weight);nn.init.zeros_(self.modulation.bias)

    def forward(self,x,coords,time_context,prompt,order):
        shift,scale,gate,shift2,scale2,gate2=self.modulation(F.silu(time_context)).chunk(6)
        query=self.norm1(x)*(1+scale)+shift
        local=self.attention(query,coords,order)
        cross,_=self.cross(query[None],prompt[None],prompt[None],need_weights=False)
        x=x+gate*(local+cross[0])
        return x+gate2*self.ffn(self.norm2(x)*(1+scale2)+shift2)


class SparseFlowDiT(nn.Module):
    def __init__(self,channels,condition_channels,width=96,heads=4,depth=4,patch_size=256,shape_channels=0):
        super().__init__()
        if depth<=0 or width%2:raise ValueError('Positive depth and even width required')
        self.width,self.channels,self.shape_channels=width,channels,shape_channels
        self.input=nn.Linear(channels,width)
        self.prompt=nn.Linear(condition_channels,width)
        self.null_prompt=nn.Parameter(torch.zeros(1,width))
        self.time=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width))
        self.shape=nn.Linear(shape_channels,width) if shape_channels else None
        self.blocks=nn.ModuleList([FlowBlock(width,heads,patch_size) for _ in range(depth)])
        self.norm=nn.LayerNorm(width)
        self.output=nn.Linear(width,channels)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,x,coords,time,condition=None,*,orders=None,shape=None):
        if x.ndim!=2 or not len(x) or x.shape[1]!=self.channels or coords.shape!=(len(x),3):
            raise ValueError('Expected nonempty (N,C) values and (N,3) coordinates')
        time=torch.as_tensor(time,device=x.device,dtype=x.dtype)
        if time.numel()!=1 or not 0<=time.item()<=1:raise ValueError('Scalar time in [0,1] required')
        if orders is None:orders=[torch.arange(len(x),device=x.device)]
        for order in orders:
            if order.device!=x.device or order.dtype!=torch.int64 or not torch.equal(order.sort().values,torch.arange(len(x),device=x.device)):
                raise ValueError('Orders must be same-device permutations of token rows')
        if not orders:raise ValueError('At least one serialization order required')
        half=self.width//2
        frequencies=torch.exp(-math.log(10000)*torch.arange(half,device=x.device,dtype=x.dtype)/max(1,half-1))
        phase=1000*time*frequencies
        context=self.time(torch.cat((phase.cos(),phase.sin())))
        prompt=self.null_prompt if condition is None else self.prompt(condition)
        if prompt.ndim!=2 or not len(prompt):raise ValueError('Condition must contain prompt tokens')
        features=self.input(x)
        if self.shape is not None:
            if shape is None or shape.shape!=(len(x),self.shape_channels):raise ValueError('Aligned clean shape latents required')
            features=features+self.shape(shape)
        elif shape is not None:raise ValueError('Model has no shape conditioning branch')
        for i,block in enumerate(self.blocks):features=block(features,coords,context,prompt,orders[i%len(orders)])
        return self.output(self.norm(features))


def flow_loss(model,clean,coords,condition,*,orders=None,shape=None,drop_probability=.1):
    if not 0<=drop_probability<=1:raise ValueError('Drop probability must lie in [0,1]')
    time=torch.rand((),device=clean.device,dtype=clean.dtype)
    noisy,target=flow_pair(clean,torch.randn_like(clean),time)
    drop=torch.rand((),device=clean.device)<drop_probability
    prediction=model(noisy,coords,time,None if drop else condition,orders=orders,shape=shape)
    return F.mse_loss(prediction,target)


@torch.no_grad()
def sample_flow(model,noise,coords,condition,*,steps=50,guidance=3.,orders=None,shape=None):
    if steps<=0 or not math.isfinite(guidance):raise ValueError('Positive steps and finite guidance required')
    x=noise.clone()
    for i in range(steps):
        time=x.new_tensor(i/steps)
        conditional=model(x,coords,time,condition,orders=orders,shape=shape)
        unconditional=model(x,coords,time,None,orders=orders,shape=shape)
        x=x+(unconditional+guidance*(conditional-unconditional))/steps
    return x
