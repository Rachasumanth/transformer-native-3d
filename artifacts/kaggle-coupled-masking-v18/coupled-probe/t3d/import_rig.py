"""Core single-skin GLB geometry/weights and sampled animation poses.

This data path is for SkinAE preprocessing, separate from the static PBR voxel
importer. Animation sampling supports core translation, rotation and scale.
"""
from dataclasses import dataclass
import copy
import hashlib
import math
from pathlib import Path
import numpy as np
from .articulation import Skeleton
from .gltf_accessors import GLBAccessors
from .materials import PBRMaterial
from .normalization import normalize_asset


def _slerp(a, b, fraction):
    """Shortest-arc quaternion interpolation for glTF LINEAR rotation tracks."""
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    dot=float(np.dot(a,b))
    if dot<0:
        b=-b;dot=-dot
    dot=float(np.clip(dot,-1,1))
    if dot>.9995:
        value=a+fraction*(b-a)
    else:
        angle=math.acos(dot);scale=math.sin(angle)
        value=(math.sin((1-fraction)*angle)/scale)*a+(math.sin(fraction*angle)/scale)*b
    norm=float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm<1e-10:
        raise ValueError('Animation produced an invalid quaternion')
    return value/norm


def _sample_track(times, values, interpolation, time, *, rotation=False):
    times=np.asarray(times,dtype=float).reshape(-1)
    values=np.asarray(values,dtype=float)
    if (not len(times) or times[0]<0 or not np.isfinite(times).all() or np.any(np.diff(times)<=0)
            or not np.isfinite(values).all()):
        raise ValueError('Animation track has invalid nonnegative times or values')
    if interpolation=='CUBICSPLINE':
        if len(times)<2:raise ValueError('CUBICSPLINE requires at least two keyframes')
        if len(values)!=3*len(times):raise ValueError('CUBICSPLINE output must have three values per key')
        values=values.reshape(len(times),3,-1)
        incoming,keys,outgoing=values[:,0],values[:,1],values[:,2]
    elif interpolation in ('LINEAR','STEP'):
        if len(values)!=len(times):raise ValueError('Animation output count differs from input times')
        keys=values
    else:
        raise ValueError('Unsupported glTF animation interpolation')
    if np.ndim(time)!=0 or not math.isfinite(float(time)):
        raise ValueError('Animation time must be a finite scalar')
    time=float(time)
    if time<=times[0]:result=keys[0]
    elif time>=times[-1]:result=keys[-1]
    else:
        right=int(np.searchsorted(times,time,side='right'));left=right-1
        if interpolation=='STEP':
            result=keys[left]
        else:
            duration=times[right]-times[left];u=(time-times[left])/duration
            if interpolation=='LINEAR':
                result=_slerp(keys[left],keys[right],u) if rotation else (1-u)*keys[left]+u*keys[right]
            else:
                u2=u*u;u3=u2*u
                result=((2*u3-3*u2+1)*keys[left]
                        +(u3-2*u2+u)*duration*outgoing[left]
                        +(-2*u3+3*u2)*keys[right]
                        +(u3-u2)*duration*incoming[right])
    result=np.asarray(result,dtype=float)
    if rotation:
        norm=float(np.linalg.norm(result))
        if not math.isfinite(norm) or norm<1e-10:
            raise ValueError('Animation produced an invalid quaternion')
        result=result/norm
    return result


def sample_animation_nodes(nodes, animation, read_accessor, time):
    """Apply one glTF animation at ``time`` and return copied node TRS data.

    Supports core translation, rotation and scale channels using STEP, LINEAR,
    or CUBICSPLINE interpolation. Morph weights and animated matrix nodes are
    rejected explicitly.
    """
    result=copy.deepcopy(nodes);samplers=animation.get('samplers',[])
    channels=animation.get('channels',[])
    if not channels:raise ValueError('Animation must contain at least one supported channel')
    seen=set()
    for channel in channels:
        if not isinstance(channel,dict):raise ValueError('Invalid animation channel')
        sampler_id=channel.get('sampler');target=channel.get('target',{})
        if not isinstance(target,dict):raise ValueError('Invalid animation channel target')
        node_id=target.get('node');path=target.get('path')
        # The glTF specification says channels without a target node should be ignored.
        if node_id is None:continue
        if (not isinstance(sampler_id,int) or not 0<=sampler_id<len(samplers)
                or not isinstance(node_id,int) or not 0<=node_id<len(result)
                or path not in ('translation','rotation','scale')):
            raise ValueError('Invalid or unsupported animation channel target')
        key=(node_id,path)
        if key in seen:raise ValueError('Animation targets a node path more than once')
        seen.add(key)
        node=result[node_id]
        if 'matrix' in node:raise ValueError('Animated matrix nodes are unsupported')
        sampler=samplers[sampler_id];interpolation=sampler.get('interpolation','LINEAR')
        input_id=sampler.get('input');output_id=sampler.get('output')
        if not isinstance(input_id,int) or not isinstance(output_id,int):
            raise ValueError('Animation sampler accessors are required')
        times=np.asarray(read_accessor(input_id),dtype=float)
        output=np.asarray(read_accessor(output_id),dtype=float)
        expected_width=4 if path=='rotation' else 3
        if times.ndim!=2 or times.shape[1]!=1 or output.ndim!=2 or output.shape[1]!=expected_width:
            raise ValueError('Animation accessor dimensions do not match target path')
        value=_sample_track(times[:,0],output,interpolation,time,rotation=path=='rotation')
        node[path]=value.tolist()
    return result


def node_world_matrices(nodes):
    parents = np.full(len(nodes), -1, dtype=np.int64)
    local = []
    for i, node in enumerate(nodes):
        for child in node.get('children', []):
            if not isinstance(child, int) or not 0 <= child < len(nodes) or parents[child] != -1:
                raise ValueError('Invalid or multiply parented node')
            parents[child] = i
        if 'matrix' in node:
            if any(k in node for k in ('translation','rotation','scale')):
                raise ValueError('Node mixes matrix and TRS')
            matrix = np.asarray(node['matrix'], dtype=float).reshape(4,4,order='F')
        else:
            t = np.asarray(node.get('translation',[0,0,0]),dtype=float)
            q = np.asarray(node.get('rotation',[0,0,0,1]),dtype=float)
            s = np.asarray(node.get('scale',[1,1,1]),dtype=float)
            if t.shape != (3,) or q.shape != (4,) or s.shape != (3,) or not np.isclose(np.linalg.norm(q),1,atol=1e-5):
                raise ValueError('Invalid node TRS')
            x,y,z,w = q/np.linalg.norm(q)
            rotation = np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                                 [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                                 [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
            matrix = np.eye(4);matrix[:3,:3] = rotation@np.diag(s);matrix[:3,3] = t
        if not np.isfinite(matrix).all() or not np.allclose(matrix[3],[0,0,0,1],atol=1e-7,rtol=0):
            raise ValueError('Invalid affine node matrix')
        local.append(matrix)
    world = np.empty((len(nodes),4,4));done = set()
    for start in range(len(nodes)):
        path=[];seen=set();current=start
        while current != -1 and current not in done:
            if current in seen:raise ValueError('Node hierarchy contains a cycle')
            seen.add(current);path.append(current);current=int(parents[current])
        for i in reversed(path):
            world[i] = local[i] if parents[i] == -1 else world[parents[i]]@local[i]
            done.add(i)
    return world, parents


@dataclass
class ImportedRig:
    vertices: np.ndarray
    faces: np.ndarray
    weights: np.ndarray
    skeleton: Skeleton
    source_sha256: str
    joint_nodes: np.ndarray
    maximum_weight_correction: float
    maximum_bind_palette_identity_delta: float
    pbr_materials: list | None = None
    face_materials: np.ndarray | None = None
    corner_uv: np.ndarray | None = None

    def normalized(self):
        vertices,joints,transform = normalize_asset(self.vertices,self.skeleton.joints)
        return ImportedRig(np.asarray(vertices),self.faces.copy(),self.weights.copy(),
                           Skeleton(joints,self.skeleton.parents),self.source_sha256,
                           self.joint_nodes.copy(),self.maximum_weight_correction,
                           self.maximum_bind_palette_identity_delta,
                           None if self.pbr_materials is None else list(self.pbr_materials),
                           None if self.face_materials is None else self.face_materials.copy(),
                           None if self.corner_uv is None else self.corner_uv.copy()),transform


def _pbr_texture(doc, glb, texture_info, image_cache):
    if texture_info is None:return None,None
    if texture_info.get('extensions') or texture_info.get('texCoord',0)!=0:
        raise ValueError('Only untransformed TEXCOORD_0 PBR textures are supported')
    texture_id=texture_info.get('index');textures=doc.get('textures',[])
    if type(texture_id) is not int or not 0<=texture_id<len(textures):
        raise ValueError('Invalid PBR texture index')
    texture=textures[texture_id]
    if texture.get('extensions') or 'source' not in texture:
        raise ValueError('Extended or source-less PBR textures are unsupported')
    image_id=texture['source']
    if image_id not in image_cache:image_cache[image_id]=glb.image(image_id)
    samplers=doc.get('samplers',[]);sampler_id=texture.get('sampler')
    if sampler_id is None:sampler={}
    elif type(sampler_id) is int and 0<=sampler_id<len(samplers):sampler=samplers[sampler_id]
    else:raise ValueError('Invalid PBR sampler index')
    wrap_s=sampler.get('wrapS',10497);wrap_t=sampler.get('wrapT',10497)
    if wrap_s!=wrap_t or wrap_s not in (10497,33071):
        raise ValueError('PBR texture must use matching repeat or clamp wrapping on both axes')
    return image_cache[image_id],('repeat' if wrap_s==10497 else 'clamp')


def _factor_pbr_material(doc, material_index, glb, image_cache):
    materials=doc.get('materials',[])
    if material_index is None:
        material={}
    else:
        if type(material_index) is not int or not 0<=material_index<len(materials):
            raise ValueError('Primitive references an invalid material')
        material=materials[material_index]
    if material.get('extensions'):
        raise ValueError('Extended rigged PBR materials are unsupported')
    pbr=material.get('pbrMetallicRoughness',{})
    emissive=np.asarray(material.get('emissiveFactor',[0,0,0]),dtype=float)
    unsupported=(pbr.get('extensions') is not None
                 or material.get('normalTexture') is not None
                 or material.get('occlusionTexture') is not None
                 or material.get('emissiveTexture') is not None
                 or np.any(emissive!=0))
    if unsupported:
        raise ValueError('Textured, normal, occlusion, and emissive rigged materials are unsupported')
    base_texture,base_wrap=_pbr_texture(doc,glb,pbr.get('baseColorTexture'),image_cache)
    mr_texture,mr_wrap=_pbr_texture(doc,glb,pbr.get('metallicRoughnessTexture'),image_cache)
    if base_wrap is not None and mr_wrap is not None and base_wrap!=mr_wrap:
        raise ValueError('Base-color and metallic-roughness textures need matching wrap modes')
    return PBRMaterial(
        base_color=pbr.get('baseColorFactor',[1,1,1,1]),
        metallic=pbr.get('metallicFactor',1),
        roughness=pbr.get('roughnessFactor',1),
        base_texture=base_texture,
        metallic_roughness_texture=mr_texture,
        wrap=base_wrap or mr_wrap or 'repeat',
        alpha_mode=material.get('alphaMode','OPAQUE'),
        alpha_cutoff=material.get('alphaCutoff',.5),
    )


def _animation_nodes(doc, glb, animation_index, time):
    """Return node TRS dictionaries at bind or one sampled animation time."""
    nodes=doc.get('nodes',[])
    if (animation_index is None)!=(time is None):
        raise ValueError('Animation index and sample time must be supplied together')
    if animation_index is None:
        return nodes
    animations=doc.get('animations',[])
    if not isinstance(animation_index,int) or not 0<=animation_index<len(animations):
        raise ValueError('Invalid animation index')
    animation=animations[animation_index]
    # Core animation samplers use float scalar times and float vector outputs.
    for sampler in animation.get('samplers',[]):
        for key in ('input','output'):
            accessor_id=sampler.get(key)
            if not isinstance(accessor_id,int) or not 0<=accessor_id<len(doc.get('accessors',[])):
                raise ValueError('Invalid animation accessor index')
            accessor=doc['accessors'][accessor_id]
            if accessor.get('componentType')!=5126:
                raise ValueError('Float32 animation accessors required')
        if doc['accessors'][sampler['input']].get('type')!='SCALAR':
            raise ValueError('Animation input accessor must be scalar')
    return sample_animation_nodes(nodes,animation,glb.read,time)


def _skin_scene(nodes, world, parents, doc):
    """Resolve the single skinned mesh instance and its joint hierarchy."""
    scenes=doc.get('scenes',[]);scene=doc.get('scene',0)
    if not 0 <= scene < len(scenes):raise ValueError('A default scene is required')
    active=set();stack=list(scenes[scene].get('nodes',[]))
    for root in stack:
        if not 0 <= root < len(nodes) or parents[root] != -1:raise ValueError('Invalid scene root')
    while stack:
        i=stack.pop()
        if i in active:raise ValueError('Repeated scene node')
        active.add(i);stack.extend(nodes[i].get('children',[]))
    mesh_nodes=[i for i in sorted(active) if 'mesh' in nodes[i]]
    if not mesh_nodes or any('skin' not in nodes[i] for i in mesh_nodes):
        raise ValueError('Expected only skinned mesh instances in selected scene')
    skin_ids={nodes[i]['skin'] for i in mesh_nodes}
    if len(skin_ids)!=1:raise ValueError('Multiple skins require separate training samples')
    skin_id=next(iter(skin_ids));skins=doc.get('skins',[])
    if not 0 <= skin_id < len(skins):raise ValueError('Invalid skin index')
    skin=skins[skin_id];joint_nodes=np.asarray(skin['joints'],dtype=np.int64)
    if joint_nodes.ndim!=1 or not len(joint_nodes) or len(set(joint_nodes))!=len(joint_nodes) or not set(joint_nodes)<=active:
        raise ValueError('Invalid or inactive skin joints')
    lookup={int(node):j for j,node in enumerate(joint_nodes)};joint_parents=[]
    for node in joint_nodes:
        p=int(parents[node])
        while p!=-1 and p not in lookup:p=int(parents[p])
        joint_parents.append(-1 if p==-1 else lookup[p])
    return joint_nodes,joint_parents,skin,mesh_nodes


def rig_joint_world(path, *, animation_index=None, time=None):
    """Return the skin joint node ids and their world matrices at bind or a pose time.

    Matrices use the glTF column-vector convention. Bind and sampled calls share
    the same node hierarchy, so ``pose @ inverse(bind)`` is the joint's world
    deformation. This helper does not infer joints or retarget them.
    """
    glb=GLBAccessors(path);doc=glb.document
    if doc.get('extensionsRequired'):raise ValueError('Required extensions unsupported')
    nodes=_animation_nodes(doc,glb,animation_index,time)
    world,parents=node_world_matrices(nodes)
    joint_nodes,_,_,_=_skin_scene(nodes,world,parents,doc)
    return joint_nodes,world[joint_nodes]


def load_rig_glb(path, *, animation_index=None, time=None, include_pbr=False):
    """Load one core-glTF skinned mesh, optionally retaining core PBR.

    ``include_pbr=True`` captures per-primitive base-color, metallic, roughness
    and alpha factors and base-color/metallic-roughness textures for Track A
    voxel projection. Normal, occlusion, emissive and extended PBR channels are
    rejected explicitly in this mode because the six-channel schema omits them.
    """
    if type(include_pbr) is not bool:raise ValueError('include_pbr must be a boolean')
    glb=GLBAccessors(path);doc=glb.document
    if doc.get('extensionsRequired'):raise ValueError('Required extensions unsupported')
    nodes=_animation_nodes(doc,glb,animation_index,time)
    world,parents=node_world_matrices(nodes)
    joint_nodes,joint_parents,skin,mesh_nodes=_skin_scene(nodes,world,parents,doc)
    skeleton=Skeleton(world[joint_nodes,:3,3],joint_parents)
    inverse=np.repeat(np.eye(4)[None],len(joint_nodes),axis=0)
    if 'inverseBindMatrices' in skin:
        a=doc['accessors'][skin['inverseBindMatrices']]
        if a['type']!='MAT4' or a['componentType']!=5126:raise ValueError('Float32 inverse binds required')
        raw=glb.read(skin['inverseBindMatrices'])
        if raw.shape!=(len(joint_nodes),16):raise ValueError('Inverse-bind count mismatch')
        inverse=raw.reshape(-1,4,4).transpose(0,2,1)
        if not np.allclose(inverse[:,3,:],[0,0,0,1],atol=1e-6,rtol=0):raise ValueError('Nonaffine inverse binds')
    matrices=world[joint_nodes]@inverse
    bind_delta=max(float(np.max(np.abs(np.linalg.inv(world[node])[None]@matrices-np.eye(4))))
                   for node in mesh_nodes)
    vertices=[];faces=[];weights=[];offset=0;correction=0.
    pbr_materials=[];material_ids={};face_materials=[];corner_uv=[];image_cache={}
    for node in mesh_nodes:
        mesh_id=nodes[node]['mesh']
        if not 0 <= mesh_id < len(doc.get('meshes',[])):raise ValueError('Invalid mesh index')
        for primitive in doc['meshes'][mesh_id]['primitives']:
            if primitive.get('mode',4)!=4 or primitive.get('targets') or primitive.get('extensions'):
                raise ValueError('Only core triangles without morph targets supported')
            position_id=primitive['attributes']['POSITION'];a=doc['accessors'][position_id]
            if a['type']!='VEC3' or a['componentType']!=5126:raise ValueError('Float32 positions required')
            v=glb.read(position_id);w,delta=glb.skin_weights(primitive,len(joint_nodes));correction=max(correction,delta)
            if 'indices' in primitive:
                index=primitive['indices'];a=doc['accessors'][index]
                if a['type']!='SCALAR' or a['componentType'] not in (5121,5123,5125) or a.get('normalized',False):
                    raise ValueError('Unsigned triangle indices required')
                indices=glb.read(index)[:,0].astype(np.int64)
            else:indices=np.arange(len(v))
            if len(indices)%3 or (indices>=len(v)).any():raise ValueError('Invalid triangle indices')
            if include_pbr:
                material_id=primitive.get('material')
                if material_id is not None and type(material_id) is not int:
                    raise ValueError('Primitive references an invalid material')
                if material_id not in material_ids:
                    material_ids[material_id]=len(pbr_materials)
                    pbr_materials.append(_factor_pbr_material(doc,material_id,glb,image_cache))
                uv_id=primitive['attributes'].get('TEXCOORD_0')
                if uv_id is None:
                    if (pbr_materials[material_ids[material_id]].base_texture is not None
                            or pbr_materials[material_ids[material_id]].metallic_roughness_texture is not None):
                        raise ValueError('Textured rigged primitive has no TEXCOORD_0')
                    uv=np.zeros((len(v),2),dtype=float)
                else:
                    uv=glb.read(uv_id)
                    if uv.shape!=(len(v),2):raise ValueError('TEXCOORD_0 must have shape (V,2)')
                triangles=indices.reshape(-1,3)
                corner_uv.append(uv[triangles])
                face_materials.append(np.full(len(triangles),material_ids[material_id],dtype=np.int64))
            # glTF skin evaluation in world coordinates: sum(w * joint_world * inverse_bind * position).
            blended=np.einsum('nj,jab->nab',w,matrices)
            evaluated=np.einsum('nab,nb->na',blended,np.column_stack((v,np.ones(len(v)))))
            vertices.append(evaluated[:,:3]);weights.append(w);faces.append(indices.reshape(-1,3)+offset);offset+=len(v)
    if not vertices:raise ValueError('No skinned triangle primitives')
    return ImportedRig(np.concatenate(vertices),np.concatenate(faces),np.concatenate(weights),skeleton,
                       hashlib.sha256(Path(path).read_bytes()).hexdigest(),joint_nodes,correction,bind_delta,
                       pbr_materials if include_pbr else None,
                       np.concatenate(face_materials) if include_pbr else None,
                       np.concatenate(corner_uv) if include_pbr else None)
