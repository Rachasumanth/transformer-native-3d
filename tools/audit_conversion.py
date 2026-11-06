"""Audit raw grid crossings and extracted incidence on the two retained GLBs."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from t3d.import_glb import load_glb
from t3d.normalization import normalize_asset
from t3d.voxelize import mesh_to_voxels
from t3d.extraction import voxels_to_mesh


def main():
    root=Path(__file__).resolve().parents[1]
    reports={}
    for name,filename in [('avocado','Avocado.glb'),('waterbottle','WaterBottle.glb')]:
        source=load_glb(root/'artifacts/import'/name/filename)
        normalized,_,_=normalize_asset(source.vertices)
        # Exact positional welding for source topology audit, independent of UV seams.
        _,inverse=np.unique(source.vertices,axis=0,return_inverse=True)
        face_indices=inverse[source.faces]
        source_edges=Counter(tuple(sorted((int(a),int(b)))) for tri in face_indices for a,b in zip(tri,np.roll(tri,-1)))
        diagnostic={}
        asset=mesh_to_voxels(np.clip(normalized,-.5,.5),source.faces,grid_size=(16,)*3,
                             aabb=[[-.5]*3,[.5]*3],diagnostics=diagnostic)
        _,faces=voxels_to_mesh(asset)
        edges=Counter(tuple(sorted((int(a),int(b)))) for tri in faces for a,b in zip(tri,np.roll(tri,-1)))
        reports[name]=dict(source_exact_weld_boundary_edges=sum(n==1 for n in source_edges.values()),
                           source_exact_weld_nonmanifold_edges=sum(n>2 for n in source_edges.values()),
                           extracted_boundary_edges=sum(n==1 for n in edges.values()),
                           extracted_nonmanifold_edges=sum(n>2 for n in edges.values()),crossings=diagnostic)
        print(name,{k:v for k,v in reports[name].items() if k!='crossings'},
              'multiple crossings:',diagnostic['multiple_crossing_edges'])
    output=root/'artifacts/import/crossing-audit.json'
    output.write_text(json.dumps(reports,indent=2)+'\n')

if __name__=='__main__':main()
