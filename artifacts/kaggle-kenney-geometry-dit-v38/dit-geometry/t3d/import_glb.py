"""Static core-glTF ingestion into normalized raw voxel assets (CPU reference)."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
import trimesh
from .materials import PBRMaterial, sample_pbr
from .normalization import normalize_asset
from .voxelize import mesh_to_voxels
from .container import save


def _document(data, *, geometry_only=False):
    if len(data) < 20 or struct.unpack_from('<4sII', data) != (b'glTF', 2, len(data)):
        raise ValueError('Expected a complete glTF 2.0 binary file')
    length, kind = struct.unpack_from('<II', data, 12)
    if kind != 0x4e4f534a or 20+length > len(data):
        raise ValueError('Invalid GLB JSON chunk')
    doc = json.loads(data[20:20+length])
    if doc.get('skins') or doc.get('animations'):
        raise ValueError('Rigged/animated GLB import is pending; refusing to drop articulation')
    allowed = {'KHR_texture_transform', 'KHR_materials_unlit'} if geometry_only else set()
    if set(doc.get('extensionsRequired', [])) - allowed:
        raise ValueError('Required glTF extensions are not supported')
    for item in doc.get('buffers', []) + doc.get('images', []):
        if item.get('uri') and not item['uri'].startswith('data:'):
            raise ValueError('Only self-contained GLB resources are supported')
    if not geometry_only:
        for sampler in doc.get('samplers', []):
            if sampler.get('wrapS',10497) != 10497 or sampler.get('wrapT',10497) != 10497:
                raise ValueError('Importer currently supports repeat samplers only')
        for material in doc.get('materials', []):
            if material.get('extensions'):
                raise ValueError('Extended materials are unsupported')
            for value in list(material.values()) + list(material.get('pbrMetallicRoughness',{}).values()):
                if isinstance(value, dict) and ('index' in value):
                    if value.get('texCoord',0) != 0 or value.get('extensions'):
                        raise ValueError('Only untransformed TEXCOORD_0 textures are supported')
    for mesh in doc.get('meshes', []):
        for primitive in mesh.get('primitives', []):
            if primitive.get('mode',4) != 4 or primitive.get('targets') or primitive.get('extensions'):
                raise ValueError('Only static triangle primitives are supported')
            if not geometry_only and 'COLOR_0' in primitive.get('attributes',{}):
                raise ValueError('Vertex-color material multiplication is not yet supported')
    return doc


@dataclass
class ImportedMesh:
    vertices: np.ndarray
    faces: np.ndarray
    corner_uv: np.ndarray
    materials: list
    face_materials: np.ndarray
    source_sha256: str


def load_glb(path):
    path = Path(path)
    data = path.read_bytes()
    _document(data)
    scene = trimesh.load_scene(path, process=False)
    vertices, faces, uvs, materials, mappings = [], [], [], [], []
    offset = 0
    def texture(image):
        return None if image is None else np.asarray(image.convert('RGBA'),dtype=float)/255
    for node in sorted(scene.graph.nodes_geometry):
        transform, name = scene.graph[node]
        mesh = scene.geometry[name]
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            raise ValueError('Scene includes non-triangle or empty geometry')
        v = trimesh.transform_points(mesh.vertices, transform)
        f = np.array(mesh.faces, copy=True)
        if np.linalg.det(transform[:3,:3]) < 0:
            f = f[:,[0,2,1]]
        visual = mesh.visual
        material = getattr(visual, 'material', None)
        if material is None:
            material = trimesh.visual.material.PBRMaterial()
        if not isinstance(material, trimesh.visual.material.PBRMaterial):
            raise ValueError('Expected one core PBR material per primitive')
        color = material.baseColorFactor
        color = np.ones(4) if color is None else np.asarray(color,dtype=float)/255
        materials.append(PBRMaterial(base_color=color,
                         metallic=1 if material.metallicFactor is None else material.metallicFactor,
                         roughness=1 if material.roughnessFactor is None else material.roughnessFactor,
                         base_texture=texture(material.baseColorTexture),
                         metallic_roughness_texture=texture(material.metallicRoughnessTexture),
                         alpha_mode=material.alphaMode or 'OPAQUE',
                         alpha_cutoff=.5 if material.alphaCutoff is None else material.alphaCutoff))
        uv = getattr(visual,'uv',None)
        if uv is None:
            if material.baseColorTexture is not None or material.metallicRoughnessTexture is not None:
                raise ValueError('Textured primitive has no UV coordinates')
            uv = np.zeros((len(v),2))
        else:
            uv = np.array(uv, copy=True)
            # Trimesh flips glTF V on import; restore the sampler's top-left convention.
            uv[:,1] = 1-uv[:,1]
        vertices.append(v); faces.append(f+offset); uvs.append(uv[f])
        mappings.append(np.full(len(f),len(materials)-1,dtype=int))
        offset += len(v)
    if not vertices:
        raise ValueError('GLB contains no mesh instances')
    return ImportedMesh(np.concatenate(vertices),np.concatenate(faces),np.concatenate(uvs),
                        materials,np.concatenate(mappings),hashlib.sha256(data).hexdigest())



def load_glb_geometry(path):
    """Load static triangle geometry only; material appearance is deliberately omitted.

    Material/UV extensions cannot affect this geometry-only result. Skinning,
    animation, mesh extensions and unknown required extensions remain rejected.
    """
    path=Path(path);data=path.read_bytes();_document(data,geometry_only=True)
    scene=trimesh.load_scene(path,process=False);vertices=[];faces=[];offset=0
    for node in sorted(scene.graph.nodes_geometry):
        transform,name=scene.graph[node];mesh=scene.geometry[name]
        if not isinstance(mesh,trimesh.Trimesh) or not len(mesh.faces):
            raise ValueError('Scene includes non-triangle or empty geometry')
        v=trimesh.transform_points(mesh.vertices,transform);f=np.array(mesh.faces,copy=True)
        if np.linalg.det(transform[:3,:3])<0:f=f[:,[0,2,1]]
        vertices.append(v);faces.append(f+offset);offset+=len(v)
    if not vertices:raise ValueError('GLB contains no mesh instances')
    return np.concatenate(vertices),np.concatenate(faces),hashlib.sha256(data).hexdigest()

def prepare_glb(source, destination, *, grid_size=32, allow_multiple_crossings=False):
    imported = load_glb(source)
    normalized, _, transform = normalize_asset(imported.vertices)
    vertices = np.asarray(normalized)
    # Avoid insignificant normalization roundoff outside the canonical cube.
    vertices = np.clip(vertices,-.5,.5)
    diagnostics={}
    asset = mesh_to_voxels(vertices,imported.faces,grid_size=(grid_size,)*3,aabb=[[-.5]*3,[.5]*3],
                           diagnostics=diagnostics,reject_multiple_crossings=not allow_multiple_crossings)
    asset = sample_pbr(asset,vertices,imported.faces,imported.materials,imported.face_materials,
                       corner_uv=imported.corner_uv)
    destination = Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    save(destination,asset)
    return dict(source_sha256=imported.source_sha256,prepared_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                faces=len(imported.faces),instances_materials=len(imported.materials),active_voxels=len(asset.coords),
                normalization=dict(center=transform.center,scale=transform.scale),grid_size=grid_size,
                pbr_min=asset.pbr.min(axis=0).tolist(),pbr_max=asset.pbr.max(axis=0).tolist(),
                crossing_diagnostics=diagnostics,multiple_crossings_allowed=allow_multiple_crossings)


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('source');parser.add_argument('destination');parser.add_argument('--grid-size',type=int,default=32)
    parser.add_argument('--allow-multiple-crossings',action='store_true',help='Diagnostic import that may lose thin/overlapping surfaces')
    args=parser.parse_args()
    report=prepare_glb(args.source,args.destination,grid_size=args.grid_size,allow_multiple_crossings=args.allow_multiple_crossings)
    Path(args.destination).with_suffix('.import.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
