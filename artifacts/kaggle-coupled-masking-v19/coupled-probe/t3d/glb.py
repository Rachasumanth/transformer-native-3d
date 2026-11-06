"""Package baked PBR maps and prepared geometry into a self-contained GLB."""
import json
import struct
import numpy as np
from PIL import Image
import trimesh
from .articulation import Skeleton


def _glb_chunks(data):
    if not isinstance(data,(bytes,bytearray)) or len(data)<20:
        raise ValueError('Expected a complete binary glTF')
    magic,version,total=struct.unpack_from('<4sII',data)
    if magic!=b'glTF' or version!=2 or total!=len(data):
        raise ValueError('Invalid GLB header')
    chunks=[];cursor=12
    while cursor<len(data):
        if cursor+8>len(data):raise ValueError('Truncated GLB chunk header')
        length,kind=struct.unpack_from('<I4s',data,cursor);cursor+=8
        end=cursor+length
        if end>len(data):raise ValueError('Truncated GLB chunk')
        chunks.append((kind,bytes(data[cursor:end])));cursor=end
    if not chunks or chunks[0][0]!=b'JSON' or sum(k==b'JSON' for k,_ in chunks)!=1:
        raise ValueError('GLB must contain exactly one leading JSON chunk')
    return chunks


def limit_skin_influences(weights, *, max_influences=16):
    """Keep the strongest influences representable by core glTF JOINTS_n sets.

    Returns normalized top-k weights and a summary of discarded probability
    mass. Ties preserve ascending joint order. The caller decides when this
    lossy projection is appropriate; ``attach_skin`` itself remains lossless.
    """
    values=np.asarray(weights,dtype=float)
    if (type(max_influences) is not int or not 1<=max_influences<=16
            or values.ndim!=2 or not values.shape[0] or not values.shape[1]
            or not np.isfinite(values).all() or np.any(values<0)):
        raise ValueError('Weights must be finite nonnegative (V,J) values and max_influences in [1,16]')
    row_sums=values.sum(axis=1)
    if not np.isfinite(row_sums).all() or np.any(row_sums<=0):
        raise ValueError('Every skin-weight row must have positive finite mass')
    values=values/row_sums[:,None]
    order=np.argsort(-values,axis=1,kind='stable')
    selected=order[:,:min(max_influences,values.shape[1])]
    retained=np.take_along_axis(values,selected,axis=1)
    retained_mass=retained.sum(axis=1)
    result=np.zeros_like(values)
    np.put_along_axis(result,selected,retained/retained_mass[:,None],axis=1)
    discarded=np.maximum(0,1-retained_mass)
    report={'max_influences':max_influences,
            'max_influences_before':int(np.count_nonzero(values>0,axis=1).max()),
            'max_influences_after':int(np.count_nonzero(result>0,axis=1).max()),
            'truncated_rows':int(np.count_nonzero(discarded>0)),
            'mean_discarded_mass':float(discarded.mean()),
            'maximum_discarded_mass':float(discarded.max()),
            'maximum_row_sum_error':float(np.max(np.abs(result.sum(axis=1)-1)))}
    return result,report



def rewrite_skin_weights(data, weights, *, max_influences=16):
    """Replace normalized skin weights in one existing self-contained GLB.

    Existing JOINTS_n/WEIGHTS_n accessors are rewritten in place, so geometry,
    materials, animation and the joint hierarchy are preserved byte-for-byte
    outside the skin-weight accessors. Reducing influences requires enough
    existing sets; this function never grows the attribute set.
    """
    values=np.asarray(weights,dtype=float)
    limited,audit=limit_skin_influences(values,max_influences=max_influences)
    chunks=_glb_chunks(data)
    if len(chunks)!=2 or chunks[1][0]!=b'BIN\x00':
        raise ValueError('Weight rewriting requires one embedded JSON and BIN chunk')
    document=json.loads(chunks[0][1].decode('utf-8'))
    meshes=document.get('meshes',[])
    skins=document.get('skins',[])
    if len(meshes)!=1 or len(meshes[0].get('primitives',[]))!=1:
        raise ValueError('Weight rewriting supports one mesh primitive')
    if len(skins)!=1:
        raise ValueError('Weight rewriting supports exactly one skin')
    primitive=meshes[0]['primitives'][0];attributes=primitive.get('attributes',{})
    accessors=document.get('accessors',[])
    if 'POSITION' not in attributes or 'JOINTS_0' not in attributes or 'WEIGHTS_0' not in attributes:
        raise ValueError('Mesh primitive must have POSITION and skin-weight attributes')
    position_id=attributes['POSITION']
    if not isinstance(position_id,int) or not 0<=position_id<len(accessors):
        raise ValueError('Invalid POSITION accessor')
    vertex_count=accessors[position_id].get('count')
    joint_count=len(skins[0].get('joints',[]))
    if limited.shape!=(vertex_count,joint_count):
        raise ValueError('Skin weights must match POSITION count and skin joint count')
    groups=[]
    for name in attributes:
        if name.startswith(('JOINTS_','WEIGHTS_')):
            suffix=name.split('_',1)[1]
            if not suffix.isdigit():
                raise ValueError('Invalid skin attribute suffix')
            groups.append(int(suffix))
    groups=sorted(set(groups))
    if not groups or groups!=list(range(len(groups))):
        raise ValueError('Contiguous skin attribute sets required')
    sets=max(1,(int(np.count_nonzero(limited>0,axis=1).max())+3)//4)
    if sets>len(groups):
        raise ValueError('max_influences requires more skin attribute sets than the GLB has')
    joint_accessor=accessors[attributes['JOINTS_0']]
    joint_component=joint_accessor.get('componentType')
    if joint_component==5121:joint_dtype=np.dtype('<u1')
    elif joint_component==5123:joint_dtype=np.dtype('<u2')
    else:raise ValueError('JOINTS_0 must use unsigned byte or short components')
    order=np.argsort(-limited,axis=1,kind='stable')
    sorted_weights=np.take_along_axis(limited,order,axis=1)
    totals=np.zeros(vertex_count,dtype=np.float32)
    for group in range(sets):
        start=4*group;end=min(start+4,joint_count)
        if end>start:totals+=sorted_weights[:,start:end].astype('<f4').sum(axis=1)
    if np.any(totals<=0):
        raise ValueError('Float32 skin weights underflowed to zero')
    binary=bytearray(chunks[1][1]);views=document.get('bufferViews',[])
    def write_accessor(accessor_id,array):
        if not isinstance(accessor_id,int) or not 0<=accessor_id<len(accessors):
            raise ValueError('Invalid skin accessor')
        accessor=accessors[accessor_id]
        if accessor.get('sparse') or 'bufferView' not in accessor:
            raise ValueError('Sparse or buffer-less skin accessors are unsupported')
        view_id=accessor['bufferView']
        if not isinstance(view_id,int) or not 0<=view_id<len(views):
            raise ValueError('Invalid skin bufferView')
        view=views[view_id]
        if view.get('buffer',0)!=0 or view.get('byteStride') not in (None,16):
            raise ValueError('Skin accessor must be tightly packed in the embedded buffer')
        if accessor.get('type')!='VEC4' or accessor.get('count')!=vertex_count:
            raise ValueError('Skin accessor must be VEC4 with POSITION count')
        offset=view.get('byteOffset',0)+accessor.get('byteOffset',0)
        payload=np.ascontiguousarray(array).tobytes()
        if offset<0 or offset+len(payload)>len(binary):
            raise ValueError('Skin accessor lies outside the embedded buffer')
        binary[offset:offset+len(payload)]=payload
    for index,group in enumerate(groups):
        joint_id=attributes[f'JOINTS_{group}'];weight_id=attributes[f'WEIGHTS_{group}']
        joint_accessor=accessors[joint_id];weight_accessor=accessors[weight_id]
        if (joint_accessor.get('componentType')!=joint_component
                or joint_accessor.get('type')!='VEC4' or weight_accessor.get('type')!='VEC4'
                or weight_accessor.get('componentType')!=5126):
            raise ValueError('Invalid skin attribute dimensions or component types')
        if joint_dtype==np.dtype('<u1') and np.any(order>=256):
            raise ValueError('Joint index exceeds unsigned-byte capacity')
        jset=np.zeros((vertex_count,4),dtype=joint_dtype)
        wset=np.zeros((vertex_count,4),dtype='<f4')
        if index<sets:
            start=4*index;end=min(start+4,joint_count);width=max(0,end-start)
            if width:
                jset[:,:width]=order[:,start:end].astype(joint_dtype)
                wset[:,:width]=sorted_weights[:,start:end].astype('<f4')/totals[:,None]
            jset[wset==0]=0
        write_accessor(joint_id,jset);write_accessor(weight_id,wset)
    bin_chunk=bytes(binary)
    if len(bin_chunk)!=len(chunks[1][1]):
        raise ValueError('Skin rewriting changed the embedded buffer length')
    json_chunk=chunks[0][1]
    total=12+8+len(json_chunk)+8+len(bin_chunk)
    result=(struct.pack('<4sII',b'glTF',2,total)
            +struct.pack('<I4s',len(json_chunk),b'JSON')+json_chunk
            +struct.pack('<I4s',len(bin_chunk),b'BIN\x00')+bin_chunk)
    audit=dict(audit,vertex_count=int(vertex_count),joint_count=int(joint_count),
               skin_attribute_sets_before=len(groups),skin_attribute_sets_after=sets)
    return result,audit


def attach_skin(data, weights, skeleton):
    """Attach supplied dense skin weights and a validated skeleton to one-mesh GLB.

    Weights are encoded losslessly into up to four JOINTS_n/WEIGHTS_n sets.
    This does not infer weights, produce animation clips, or remesh geometry.
    """
    if not isinstance(skeleton,Skeleton):raise ValueError('A validated Skeleton is required')
    chunks=_glb_chunks(data);document=json.loads(chunks[0][1].decode('utf-8'))
    if len(chunks)!=2 or chunks[1][0]!=b'BIN\x00':
        raise ValueError('Skin attachment requires a self-contained GLB with one BIN chunk')
    if (len(document.get('buffers',[]))!=1 or 'uri' in document['buffers'][0]
            or len(document.get('meshes',[]))!=1 or len(document.get('scenes',[]))!=1):
        raise ValueError('Skin attachment supports one embedded buffer, mesh, and scene')
    mesh_nodes=[i for i,node in enumerate(document.get('nodes',[])) if 'mesh' in node]
    if len(mesh_nodes)!=1 or document['nodes'][mesh_nodes[0]].get('mesh')!=0:
        raise ValueError('Exactly one mesh node is required')
    mesh_node=mesh_nodes[0]
    if 'skin' in document['nodes'][mesh_node]:raise ValueError('Mesh already has a skin')
    primitives=document['meshes'][0].get('primitives',[])
    if len(primitives)!=1:raise ValueError('Skin attachment requires one mesh primitive')
    primitive=primitives[0]
    # Trimesh omits the optional GL_ARRAY_BUFFER / GL_ELEMENT_ARRAY_BUFFER hints.
    # Adding them makes the emitted GLB clean under the Khronos validator and
    # helps clients that distinguish vertex and index buffer views.
    buffer_views=document.get('bufferViews',[])
    def set_view_target(accessor_id,target):
        if not isinstance(accessor_id,int) or not 0<=accessor_id<len(document.get('accessors',[])):
            raise ValueError('Primitive references an invalid accessor')
        view_id=document['accessors'][accessor_id].get('bufferView')
        if not isinstance(view_id,int) or not 0<=view_id<len(buffer_views):
            raise ValueError('Primitive accessor must reference a bufferView')
        existing=buffer_views[view_id].get('target')
        if existing is not None and existing!=target:
            raise ValueError('Primitive bufferView has a conflicting target')
        buffer_views[view_id]['target']=target
    for accessor_id in primitive.get('attributes',{}).values():
        set_view_target(accessor_id,34962)
    if 'indices' in primitive:set_view_target(primitive['indices'],34963)
    if any(key.startswith(('JOINTS_','WEIGHTS_')) for key in primitive.get('attributes',{})):
        raise ValueError('Mesh already contains skin attributes')
    position_id=primitive.get('attributes',{}).get('POSITION')
    accessors=document.get('accessors',[])
    if not isinstance(position_id,int) or not 0<=position_id<len(accessors):
        raise ValueError('Mesh POSITION accessor is missing')
    vertex_count=accessors[position_id].get('count')
    values=np.asarray(weights,dtype=float);joints=skeleton.joints;parents=skeleton.parents
    if (not isinstance(vertex_count,int) or values.shape!=(vertex_count,len(joints))
            or not np.isfinite(values).all() or np.any(values<0)
            or not np.allclose(values.sum(1),1,atol=1e-6,rtol=0)):
        raise ValueError('Skin weights must be normalized finite (V,J) values matching POSITION')
    if len(joints)>65536:raise ValueError('glTF joint indices exceed uint16 capacity')
    influence_count=np.count_nonzero(values>0,axis=1)
    sets=max(1,(int(influence_count.max())+3)//4)
    if sets>4:raise ValueError('At most 16 nonzero influences per vertex are supported')
    # Stable descending sort preserves joint-index order for tied weights.
    order=np.argsort(-values,axis=1,kind='stable')
    sorted_weights=np.take_along_axis(values,order,axis=1)
    joint_dtype=np.dtype('<u1') if len(joints)<=256 else np.dtype('<u2')
    joint_component=5121 if len(joints)<=256 else 5123
    binary=bytearray(chunks[1][1]);buffer_views=document.setdefault('bufferViews',[])
    accessors=document.setdefault('accessors',[])
    def add_accessor(array,component_type,kind,target):
        while len(binary)%4:binary.append(0)
        offset=len(binary);payload=np.ascontiguousarray(array).tobytes();binary.extend(payload)
        view_id=len(buffer_views)
        view=dict(buffer=0,byteOffset=offset,byteLength=len(payload))
        if target is not None:view['target']=target
        buffer_views.append(view)
        accessor_id=len(accessors)
        accessors.append(dict(bufferView=view_id,byteOffset=0,componentType=component_type,
                              count=len(array),type=kind))
        return accessor_id
    for group in range(sets):
        start=4*group;end=min(start+4,len(joints))
        jset=np.zeros((vertex_count,4),dtype=joint_dtype)
        wset=np.zeros((vertex_count,4),dtype='<f4')
        width=max(0,end-start)
        if width:
            jset[:,:width]=order[:,start:end]
            wset[:,:width]=sorted_weights[:,start:end]
        jset[wset==0]=0
        # Float32 conversion is followed by row renormalization to preserve glTF's sum-to-one rule.
        if group==0:
            totals=np.zeros(vertex_count,dtype=np.float32)
            for component in range(sets):
                lo=component*4;hi=min(lo+4,len(joints))
                if hi>lo:totals+=sorted_weights[:,lo:hi].astype(np.float32).sum(axis=1)
            if np.any(totals<=0):raise ValueError('Float32 skin weights underflowed to zero')
        wset/=totals[:,None]
        attributes=primitive['attributes']
        attributes[f'JOINTS_{group}']=add_accessor(jset,joint_component,'VEC4',34962)
        attributes[f'WEIGHTS_{group}']=add_accessor(wset,5126,'VEC4',34962)

    # The exported mesh is already in its bind pose. Joint world transforms are
    # translations at the supplied positions, so inverse binds cancel them.
    matrices=np.repeat(np.eye(4,dtype=np.float64)[None],len(joints),axis=0)
    matrices[:,:3,3]=-joints
    inverse_accessor=add_accessor(matrices.transpose(0,2,1).astype('<f4').reshape(len(joints),16),
                                  5126,'MAT4',None)
    nodes=document.setdefault('nodes',[])
    if not nodes or any(key in nodes[0] for key in ('matrix','translation','rotation','scale')):
        raise ValueError('Expected an identity root node for attaching joint hierarchy')
    root_id=0;node_ids=list(range(len(nodes),len(nodes)+len(joints)))
    children=[[] for _ in joints];roots=[]
    for index,parent in enumerate(parents):
        if parent<0:roots.append(index)
        else:children[int(parent)].append(index)
    for index,node_id in enumerate(node_ids):
        parent=int(parents[index])
        translation=joints[index] if parent<0 else joints[index]-joints[parent]
        node=dict(name=f'joint_{index}',translation=np.asarray(translation,dtype=float).tolist())
        if children[index]:node['children']=[node_ids[child] for child in children[index]]
        nodes.append(node)
    nodes[root_id].setdefault('children',[]).extend(node_ids[index] for index in roots)
    if any(key in nodes[mesh_node] for key in ('matrix','translation','rotation','scale')):
        raise ValueError('Skinned mesh node must have an identity transform')
    # A skin is evaluated in mesh space; keeping its node as a non-root child
    # invites unsupported parent transforms to affect the result.
    for node in nodes:
        if mesh_node in node.get('children',[]):
            node['children'].remove(mesh_node)
    scene_roots=document['scenes'][document.get('scene',0)].setdefault('nodes',[])
    if mesh_node not in scene_roots:scene_roots.append(mesh_node)
    skins=document.setdefault('skins',[])
    skin_id=len(skins)
    skin=dict(joints=node_ids,inverseBindMatrices=inverse_accessor)
    if len(roots)==1:skin['skeleton']=node_ids[roots[0]]
    skins.append(skin);nodes[mesh_node]['skin']=skin_id
    document['buffers'][0]['byteLength']=len(binary)
    json_chunk=json.dumps(document,separators=(',',':'),ensure_ascii=False).encode('utf-8')
    json_chunk+=b' '*((-len(json_chunk))%4)
    binary=bytes(binary)+bytes((-len(binary))%4)
    total=12+8+len(json_chunk)+8+len(binary)
    return (struct.pack('<4sII',b'glTF',2,total)
            +struct.pack('<I4s',len(json_chunk),b'JSON')+json_chunk
            +struct.pack('<I4s',len(binary),b'BIN\x00')+binary)


def pack_glb(vertices, faces, uv, normals, pbr, *, source_up='z', alpha_mode='OPAQUE', alpha_cutoff=.5,
             skin_weights=None, skeleton=None):
    """Return binary glTF bytes. pbr is H×W×6 linear RGB/metal/rough/alpha.

    UV input uses the rasterizer convention and is vertically flipped for glTF.
    z-up positions/normals map (x,y,z) to (x,z,-y); y-up is already aligned.
    Alpha mode is explicit: use BLEND for translucent material, MASK for cutouts.
    This packages supplied baked textures; it does not bake or repair geometry.
    """
    v=np.array(vertices,dtype=np.float32,copy=True)
    f=np.asarray(faces)
    t=np.array(uv,dtype=np.float32,copy=True)
    n=np.array(normals,dtype=np.float32,copy=True)
    p=np.asarray(pbr)
    if v.ndim!=2 or v.shape[1]!=3 or not len(v) or not np.isfinite(v).all():
        raise ValueError('vertices must be finite nonempty (V,3)')
    if f.ndim!=2 or f.shape[1]!=3 or not len(f) or not np.issubdtype(f.dtype,np.integer):
        raise ValueError('faces must be nonempty integer (F,3)')
    if (f<0).any() or (f>=len(v)).any():raise ValueError('face index out of range')
    if t.shape!=(len(v),2) or not np.isfinite(t).all() or (t<0).any() or (t>1).any():
        raise ValueError('UVs must be finite (V,2) in [0,1]')
    if n.shape!=v.shape or not np.isfinite(n).all() or (np.linalg.norm(n,axis=1)==0).any():
        raise ValueError('normals must be finite nonzero (V,3)')
    if p.ndim!=3 or p.shape[-1]!=6 or min(p.shape[:2])<1 or not np.isfinite(p).all() or (p<0).any() or (p>1).any():
        raise ValueError('PBR must be finite (H,W,6) in [0,1]')
    if source_up not in ('y','z'):raise ValueError('source_up must be y or z')
    if alpha_mode not in ('OPAQUE','MASK','BLEND'):raise ValueError('invalid alpha mode')
    if not np.isfinite(alpha_cutoff) or not 0<=alpha_cutoff<=1:raise ValueError('invalid alpha cutoff')
    if source_up=='z':
        v=v[:,[0,2,1]]*np.array([1,1,-1],dtype=np.float32)
        n=n[:,[0,2,1]]*np.array([1,1,-1],dtype=np.float32)
    n=n/np.linalg.norm(n,axis=1,keepdims=True)
    t[:,1]=1-t[:,1]
    rgb=p[...,:3]
    srgb=np.where(rgb<=.0031308,12.92*rgb,1.055*np.power(rgb,1/2.4)-.055)
    byte=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
    rgba=np.concatenate([byte(srgb),byte(p[...,5:6])],axis=-1)
    mr=np.concatenate([np.zeros(p.shape[:2]+(1,),dtype=np.uint8),byte(p[...,4:5]),byte(p[...,3:4])],axis=-1)
    material_args=dict(baseColorTexture=Image.fromarray(rgba),
        baseColorFactor=np.full(4,255,dtype=np.uint8),metallicRoughnessTexture=Image.fromarray(mr),
        metallicFactor=1.,roughnessFactor=1.,alphaMode=alpha_mode,doubleSided=True)
    if alpha_mode=='MASK':material_args['alphaCutoff']=alpha_cutoff
    material=trimesh.visual.material.PBRMaterial(**material_args)
    mesh=trimesh.Trimesh(vertices=v,faces=f,vertex_normals=n,process=False,
                        visual=trimesh.visual.TextureVisuals(uv=t,material=material))
    if (skin_weights is None)!=(skeleton is None):
        raise ValueError('skin_weights and skeleton must be supplied together')
    data=mesh.export(file_type='glb',include_normals=True)
    if skin_weights is None:return data
    if source_up=='z':
        rotated=skeleton.joints[:,[0,2,1]]*np.array([1.,1.,-1.])
        skeleton=Skeleton(rotated,skeleton.parents)
    return attach_skin(data,skin_weights,skeleton)
