"""Audit static and rigged pilot meshes for lossless 128-bin BPT eligibility."""
from collections import Counter
import hashlib
from itertools import permutations, product
import json
from pathlib import Path
import time
from urllib.parse import unquote
import numpy as np
import trimesh

from t3d.bpt import encode, decode, quantize, quantization_diagnostics
from t3d.import_glb import load_glb
from t3d.import_rig import load_rig_glb
from t3d.gltf_accessors import GLBAccessors
from t3d.metrics import surface_comparison
from t3d.normalization import normalize_asset


def cube_rotation_quantization(vertices,faces):
    """Check the 24 proper axis-aligned cube rotations allowed for augmentation."""
    outcomes=[]
    for permutation in permutations(range(3)):
        basis=np.eye(3,dtype=int)[list(permutation)]
        for signs in product((-1,1),repeat=3):
            rotation=np.diag(signs)@basis
            if round(np.linalg.det(rotation))!=1:
                continue
            diag=quantization_diagnostics(vertices@rotation.T,faces,bins=128)
            outcomes.append((diag['collapsed_faces'],diag['collinear_faces']))
    return dict(rotations_tested=len(outcomes),safe_rotations=sum(c==0 and l==0 for c,l in outcomes),
        best_collapsed_faces=min(c for c,l in outcomes),
        best_collinear_faces=min(l for c,l in outcomes),
        best_total_degenerate_faces=min(c+l for c,l in outcomes))


def oriented_face_multiset(q, faces):
    result=[]
    for face in np.asarray(q)[np.asarray(faces)]:
        coords=[tuple(row) for row in face]
        result.append(min(tuple(coords[i:]+coords[:i]) for i in range(3)))
    return Counter(result)


def load_geometry_only_glb(path):
    """Load static triangles while ignoring material-only glTF extensions."""
    path=Path(path)
    if path.suffix.lower()=='.gltf':
        doc=json.loads(path.read_text())
        digest=hashlib.sha256()
        digest.update(path.name.encode()+b'\0'+path.read_bytes())
        for uri in sorted({item['uri'] for key in ('buffers','images')
                           for item in doc.get(key,[]) if 'uri' in item
                           and not item['uri'].startswith('data:')}):
            dependency=path.parent/unquote(uri)
            digest.update(uri.encode()+b'\0'+dependency.read_bytes())
        source_sha256=digest.hexdigest()
    else:
        glb=GLBAccessors(path)
        doc=glb.document
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    if doc.get('skins') or doc.get('animations'):
        raise ValueError('BPT static geometry preflight refuses rigged/animated GLBs')
    for extension in doc.get('extensionsRequired',[]):
        if not (extension.startswith('KHR_materials_') or extension=='KHR_texture_transform'):
            raise ValueError(f'Unsupported required non-material extension: {extension}')
    for mesh in doc.get('meshes',[]):
        for primitive in mesh.get('primitives',[]):
            if primitive.get('mode',4)!=4 or primitive.get('targets'):
                raise ValueError('BPT static geometry preflight requires static triangle primitives')
            for extension in primitive.get('extensions',{}):
                if extension!='KHR_materials_variants':
                    raise ValueError(f'Unsupported primitive extension: {extension}')
    scene=trimesh.load_scene(path,process=False)
    vertices=[];faces=[];offset=0
    for node in sorted(scene.graph.nodes_geometry):
        transform,name=scene.graph[node]
        mesh=scene.geometry[name]
        if not isinstance(mesh,trimesh.Trimesh) or not len(mesh.faces):
            raise ValueError('Scene includes non-triangle or empty geometry')
        v=trimesh.transform_points(mesh.vertices,transform)
        f=np.asarray(mesh.faces,dtype=np.int64).copy()
        if np.linalg.det(transform[:3,:3])<0:f=f[:,[0,2,1]]
        vertices.append(v);faces.append(f+offset);offset+=len(v)
    if not vertices:raise ValueError('GLB contains no mesh instances')
    return np.concatenate(vertices),np.concatenate(faces),source_sha256


def audit_asset(name,path,*,rigged=False):
    if rigged:
        source=load_rig_glb(path)
        source,_=source.normalized()
        vertices=np.clip(source.vertices,-.5,.5)
        faces=source.faces
        source_sha256=source.source_sha256
        kind='rigged-default-pose'
    else:
        try:
            source=load_glb(path)
            vertices,_,_=normalize_asset(source.vertices)
            vertices=np.clip(vertices,-.5,.5)
            faces=source.faces
            source_sha256=source.source_sha256
            kind='static-core-gltf'
        except ValueError:
            vertices,faces,source_sha256=load_geometry_only_glb(path)
            vertices,_,_=normalize_asset(vertices)
            vertices=np.clip(vertices,-.5,.5)
            kind='static-geometry-only-extended-materials'
    sweep=[quantization_diagnostics(vertices,faces,bins=bins)
           for bins in (128,256,512,1024,2048,4096,8192)]
    at_128=sweep[0]
    base=dict(asset=name,kind=kind,
        source_sha256=source_sha256,source_faces=len(faces),
        exact_unique_positions=int(len(np.unique(vertices,axis=0))),
        quantization_128_collapsed_faces=at_128['collapsed_faces'],
        quantization_128_collinear_faces=at_128['collinear_faces'],
        source_zero_area_faces=at_128['source_zero_area_faces'],
        source_tiny_faces_1e_8=at_128['source_tiny_faces_1e_8'],
        quantization_sweep=sweep,
        cube_rotation_preflight=cube_rotation_quantization(vertices,faces))
    started=time.perf_counter()
    try:
        tokens=encode(vertices,faces)
    except ValueError as error:
        return dict(**base,status='rejected',reason=str(error),quantization_bins=128)
    elapsed=time.perf_counter()-started
    centers=int(np.count_nonzero(tokens<512))
    decoded_vertices,decoded_faces=decode(tokens)
    qsrc=quantize(vertices)
    qdec=quantize(decoded_vertices)
    face_multiset_equal=(oriented_face_multiset(qsrc,faces)==
                         oriented_face_multiset(qdec,decoded_faces))
    metric=[surface_comparison(vertices,faces,decoded_vertices,decoded_faces,
                               samples=512,seed=seed) for seed in (0,1,2)]
    result=dict(**base,status='encoded',tokens=int(len(tokens)),
        raw_coordinate_tokens=int(9*len(faces)),
        token_fraction=float(len(tokens)/(9*len(faces))),
        sequence_reduction=float(1-len(tokens)/(9*len(faces))),
        patch_count=centers,patches_per_1000_faces=float(1000*centers/len(faces)),
        encode_seconds=elapsed,decoded_vertices=len(decoded_vertices),decoded_faces=len(decoded_faces),
        oriented_quantized_faces_preserved=face_multiset_equal,
        max_abs_vertex_quantization_error=float(np.max(np.abs(vertices-((qsrc+.5)/128-.5)))),
        rms_vertex_quantization_error=float(np.sqrt(np.mean((vertices-((qsrc+.5)/128-.5))**2))),
        surface_metrics=metric)
    if not np.array_equal(np.unique(qsrc,axis=0),np.unique(qdec,axis=0)):
        raise ValueError(f'{name}: token stream changed the quantized vertex set')
    if len(decoded_faces)!=len(faces) or not face_multiset_equal:
        raise ValueError(f'{name}: quantized oriented mesh topology did not round-trip')
    return result


def main():
    root=Path(__file__).resolve().parents[1]
    candidates=[(name,root/'artifacts/native-inputs'/f'{name}.glb',False)
                for name in ('Avocado','WaterBottle','Lantern')]
    candidates.extend((name,root/'artifacts/bpt-candidates'/name/f'{name}.glb',False)
                for name in ('BarramundiFish','BoomBox','SheenChair','Box','ScatteringSkull'))
    candidates.extend((f'PolyHaven_{name}',
        root/'artifacts/bpt-candidates'/f'PolyHaven_{name}'/f'{name}_1k.gltf',False)
        for name in ('Lantern_01','CoffeeTable_01','WoodenChair_01','food_lime_01','metal_toolbox'))
    candidates.extend((name,path,True) for name,path in (
        ('CesiumMan',root/'artifacts/rigged-inputs/CesiumMan/CesiumMan.glb'),
        ('Fox',root/'artifacts/rigged-inputs/Fox/Fox.glb'),
        ('RiggedSimple',root/'artifacts/rigged-inputs/RiggedSimple/RiggedSimple.glb')))
    results=[]
    source_hash=hashlib.sha256()
    for name,path,rigged in candidates:
        result=audit_asset(name,path,rigged=rigged)
        results.append(result)
        source_hash.update(name.encode()+b'\0')
        source_hash.update(bytes.fromhex(result['source_sha256']))
        print(name,result['status'],{k:result.get(k) for k in ('source_faces','reason',
            'quantization_128_collapsed_faces','quantization_128_collinear_faces',
            'source_zero_area_faces','cube_rotation_preflight','patch_count',
            'sequence_reduction')},flush=True)
    report=dict(scope='thirteen static Khronos/Poly Haven candidates and three rigged Khronos pilot GLBs; rigged geometry uses shared node-default normalization',
        denominator='9 coordinate tokens per source triangle before BPT',
        source_bundle_sha256=source_hash.hexdigest(),
        patch_count_note='BPT patches are not EdgeBreaker subsequence counts',cases=results)
    out=root/'artifacts/bpt/corpus-audit.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(f'Wrote {out}')


if __name__=='__main__':main()
