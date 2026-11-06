"""Round-trip retained rigged assets through the new GLB skin packer."""
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np

from t3d.glb import pack_glb
from t3d.import_rig import load_rig_glb


def vertex_normals(vertices,faces):
    triangles=vertices[faces]
    face=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    normals=np.zeros_like(vertices,dtype=float)
    for corner in range(3):np.add.at(normals,faces[:,corner],face)
    length=np.linalg.norm(normals,axis=1)
    if (length==0).any():raise ValueError('Source mesh contains isolated or degenerate vertices')
    return normals/length[:,None]


def export_roundtrip(source, output_path=None):
    source=Path(source);rig=load_rig_glb(source)
    pbr=np.tile(np.array([.5,.5,.5,0,.7,1.],dtype=np.float32),(2,2,1))
    data=pack_glb(rig.vertices,rig.faces,np.full((len(rig.vertices),2),.5),
        vertex_normals(rig.vertices,rig.faces),pbr,source_up='y',
        skin_weights=rig.weights,skeleton=rig.skeleton)
    if output_path is not None:
        output_path=Path(output_path);output_path.parent.mkdir(parents=True,exist_ok=True)
        output_path.write_bytes(data)
    with tempfile.TemporaryDirectory() as directory:
        output=Path(directory)/'roundtrip.glb';output.write_bytes(data)
        restored=load_rig_glb(output)
    if not np.array_equal(restored.faces,rig.faces):raise ValueError('GLB export changed face topology')
    return dict(source=str(source),source_sha256=rig.source_sha256,
        output_file=str(output_path) if output_path is not None else None,
        output_sha256=hashlib.sha256(data).hexdigest(),vertices=len(rig.vertices),
        faces=len(rig.faces),joints=len(rig.skeleton.joints),
        maximum_vertex_error=float(np.max(np.abs(restored.vertices-rig.vertices))),
        maximum_joint_error=float(np.max(np.abs(restored.skeleton.joints-rig.skeleton.joints))),
        maximum_weight_error=float(np.max(np.abs(restored.weights-rig.weights))),
        maximum_bind_palette_identity_delta=restored.maximum_bind_palette_identity_delta,
        animation_count=0)


def main():
    root=Path(__file__).resolve().parents[1]
    assets=('RiggedSimple','CesiumMan','Fox')
    output_dir=root/'artifacts/rigged-inputs/glb-export-roundtrip'
    results=[dict(asset=name,**export_roundtrip(
        root/'artifacts/rigged-inputs'/name/f'{name}.glb',output_dir/f'{name}.glb'))
             for name in assets]
    report=dict(scope='Core skinned GLB packer round-trip; flat placeholder PBR, no source material or animation transfer',
        method='Pack imported node-default vertices, dense source weights, and joint hierarchy; decode with independent project GLB importer',
        assets=results)
    output=root/'artifacts/rigged-inputs/glb-export-roundtrip-audit.json'
    output.write_text(json.dumps(report,indent=2)+'\n')
    for item in results:print(item['asset'],{k:item[k] for k in ('maximum_vertex_error','maximum_joint_error','maximum_weight_error','maximum_bind_palette_identity_delta')})
    print(f'Wrote {output}')


if __name__=='__main__':main()
