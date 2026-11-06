"""Experimental T3DV v2 profile 2: shared-support latents and optional skeleton."""
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
from .container import _HEADER, _align, _bpt, _CURVES
from .schema import VoxelAsset
from .articulation import Skeleton
from .serialization import serialization_order
from .container_io import atomic_write, seal, verify_and_strip


@dataclass(frozen=True)
class LatentAsset:
    coords: np.ndarray
    shape: np.ndarray
    grid_size: tuple
    aabb: np.ndarray
    shape_model_sha256: str
    material: np.ndarray | None = None
    material_model_sha256: str | None = None
    rig: np.ndarray | None = None
    rig_model_sha256: str | None = None
    skeleton: Skeleton | None = None

    def __post_init__(self):
        # Validate source geometry metadata with the raw schema, without fake voxels.
        geometry=VoxelAsset(np.empty((0,3),int),np.empty((0,3)),np.empty((0,3),bool),self.grid_size,self.aabb)
        object.__setattr__(self,'grid_size',geometry.grid_size)
        object.__setattr__(self,'aabb',geometry.aabb)
        c=np.array(self.coords,copy=True)
        latent_grid=(np.array(geometry.grid_size)+15)//16
        if (c.ndim!=2 or c.shape[1]!=3 or not np.issubdtype(c.dtype,np.integer)
                or np.any(c<0) or np.any(c>=latent_grid)):
            raise ValueError('Latent coordinates must be integer (N,3) within ceil(grid/16)')
        serialization_order(c,depth=max(1,(int(latent_grid.max())-1).bit_length()))
        c=c.astype(np.uint16);c.setflags(write=False);object.__setattr__(self,'coords',c)
        for name in ('shape','material','rig'):
            value=getattr(self,name);identity=getattr(self,name+'_model_sha256')
            if value is None:
                if name=='shape' or identity is not None:
                    raise ValueError('Shape latents required; absent streams must have absent model identity')
                continue
            a=np.array(value,dtype=float,copy=True)
            if a.ndim!=2 or a.shape[0]!=len(c) or not 1<=a.shape[1]<=65535 or not np.isfinite(a).all():
                raise ValueError('Latent streams must be finite (N,D), 1<=D<=65535')
            if not isinstance(identity,str) or not re.fullmatch('[0-9a-f]{64}',identity) or identity=='0'*64:
                raise ValueError('Each stream requires a nonzero checkpoint SHA256')
            a.setflags(write=False);object.__setattr__(self,name,a)
        if self.skeleton is not None and not isinstance(self.skeleton,Skeleton):
            raise ValueError('Skeleton must be a validated Skeleton')


@dataclass(frozen=True)
class LatentContainer:
    latents: LatentAsset
    bpt_tokens: np.ndarray
    curve: str


def dumps_latents(asset, *, curve='z', bpt_tokens=None):
    if curve not in _CURVES:
        raise ValueError('Unknown curve')
    grid=(np.array(asset.grid_size)+15)//16
    order,_=serialization_order(asset.coords,depth=max(1,(int(grid.max())-1).bit_length()),
                curve='h' if curve in ('h','th') else 'z',transposed=curve in ('tz','th'))
    tokens=_bpt(bpt_tokens)
    n=len(asset.coords);j=0 if asset.skeleton is None else len(asset.skeleton.joints)
    if max(n,j,len(tokens))>=2**32:
        raise ValueError('Count exceeds uint32')
    dimensions=[0 if getattr(asset,k) is None else getattr(asset,k).shape[1] for k in ('shape','material','rig')]
    # Profile 4 is latent profile 2 with a SHA-256 integrity trailer.
    layout=[16,*dimensions]+[0]*15+[4,_CURVES.index(curve),64]+[0]*6
    with np.errstate(over='ignore',invalid='ignore'):
        bounds=asset.aabb.astype('<f4')
        sizes=((bounds[1].astype(float)-bounds[0])/asset.grid_size).astype('<f4')
    if not np.isfinite(bounds).all() or np.any(bounds[1]<=bounds[0]) or np.any(sizes<=0) or not np.isfinite(sizes).all():
        raise ValueError('Geometry cannot be represented as float32')
    header=_HEADER.pack(b'T3DV',2,*bounds.ravel(),*asset.grid_size,*sizes,0,n,len(tokens),j,*layout)
    blocks=[asset.coords[order].astype('<u2').tobytes()]
    for name in ('shape','material','rig'):
        value=getattr(asset,name)
        if value is None:
            blocks.append(b'');continue
        with np.errstate(over='ignore'):quantized=value[order].astype('<f2')
        if not np.isfinite(quantized).all():raise ValueError('Latent float16 overflow')
        blocks.append(quantized.tobytes())
    blocks.append(b''.join(bytes(32) if getattr(asset,name) is None else bytes.fromhex(getattr(asset,name+'_model_sha256'))
                            for name in ('shape','material','rig')))
    joints=np.empty((0,3)) if asset.skeleton is None else asset.skeleton.joints
    with np.errstate(over='ignore'):joints=joints.astype('<f4')
    if not np.isfinite(joints).all():raise ValueError('Joint float32 overflow')
    blocks.extend([joints.tobytes(),b'' if asset.skeleton is None else asset.skeleton.parents.astype('<i4').tobytes(),tokens.tobytes()])
    result=bytearray(header)
    for block in blocks:
        result.extend(bytes(_align(len(result))-len(result)));result.extend(block)
    return seal(bytes(result))


def loads_latents(data, *, expected_models=None):
    data=memoryview(data)
    if len(data)<128:raise ValueError('Truncated header')
    fields=_HEADER.unpack(data[:128])
    if fields[:2]!=(b'T3DV',2):raise ValueError('Unsupported magic/version')
    bounds=np.array(fields[2:8]).reshape(2,3);grid=tuple(fields[8:11]);size=np.array(fields[11:14])
    raw,n,bpt,j=fields[14:18];layout=fields[18:]
    profile=layout[19]
    if (raw or layout[0]!=16 or profile not in (2,4) or layout[20] not in range(4) or layout[21]!=64
            or any(layout[4:19]) or any(layout[22:]) or not layout[1]):
        raise ValueError('Unsupported latent profile/layout')
    if profile==4:data=verify_and_strip(data)
    if min(grid)<=0 or not np.isfinite(size).all() or not np.allclose(size,(bounds[1]-bounds[0])/grid,rtol=2e-7,atol=0):
        raise ValueError('Invalid voxel dimensions')
    cursor=128
    def block(length):
        nonlocal cursor
        start=_align(cursor);end=start+length
        if end>len(data) or any(data[cursor:start]):raise ValueError('Truncated block or invalid padding')
        cursor=end;return data[start:end]
    coords=np.frombuffer(block(n*6),dtype='<u2').reshape(n,3)
    streams=[]
    for d in layout[1:4]:
        payload=block(n*d*2)
        streams.append(None if d==0 else np.frombuffer(payload,dtype='<f2').reshape(n,d))
    identities=bytes(block(96));kwargs={}
    for i,name in enumerate(('shape','material','rig')):
        identity=identities[32*i:32*i+32].hex()
        if streams[i] is None:
            if identity!='0'*64:raise ValueError('Absent stream has model identity')
            identity=None
        kwargs[name]=streams[i];kwargs[name+'_model_sha256']=identity
        if expected_models is not None and name in expected_models and expected_models[name]!=identity:
            raise ValueError(f'{name} decoder checkpoint identity mismatch')
    joints=np.frombuffer(block(j*12),dtype='<f4').reshape(j,3)
    parents=np.frombuffer(block(j*4),dtype='<i4')
    tokens=np.frombuffer(block(bpt*2),dtype='<u2').copy()
    if cursor!=len(data):raise ValueError('Unexpected trailing payload')
    asset=LatentAsset(coords,grid_size=grid,aabb=bounds,skeleton=Skeleton(joints,parents) if j else None,**kwargs)
    curve=_CURVES[layout[20]];latent_grid=(np.array(grid)+15)//16
    order,_=serialization_order(asset.coords,depth=max(1,(int(latent_grid.max())-1).bit_length()),
                    curve='h' if curve in ('h','th') else 'z',transposed=curve in ('tz','th'))
    if not np.array_equal(order,np.arange(n)):raise ValueError('Latent coordinate order mismatch')
    _bpt(tokens);tokens.setflags(write=False)
    return LatentContainer(asset,tokens,curve)


def save_latents(path,asset,**kwargs):
    data=dumps_latents(asset,**kwargs);loads_latents(data);atomic_write(path,data)


def load_latents(path,**kwargs):return loads_latents(Path(path).read_bytes(),**kwargs)
