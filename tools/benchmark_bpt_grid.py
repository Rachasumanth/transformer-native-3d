"""Reproduce the 8192-face BPT compression and oriented round-trip fixture."""
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np

from t3d.bpt import encode, decode, quantize, quantization_diagnostics


def oriented_faces(q,faces):
    result=[]
    for face in faces:
        coords=[tuple(row) for row in q[face]]
        result.append(min(tuple(coords[i:]+coords[:i]) for i in range(3)))
    return Counter(result)


def main():
    side=65
    axis=np.linspace(-.5,.5,side)
    vertices=np.asarray([(x,y,0.) for y in axis for x in axis])
    faces=[]
    for y in range(side-1):
        for x in range(side-1):
            a=y*side+x;b=a+1;c=a+side;d=c+1
            faces.extend(((a,b,d),(a,d,c)))
    faces=np.asarray(faces,dtype=np.int64)
    before=quantization_diagnostics(vertices,faces,bins=128)
    if not before['eligible']:
        raise ValueError(f'Grid fixture is not eligible: {before}')
    started=time.perf_counter()
    tokens=encode(vertices,faces)
    elapsed=time.perf_counter()-started
    decoded_vertices,decoded_faces=decode(tokens)
    preserved=(oriented_faces(quantize(vertices),faces)==
               oriented_faces(quantize(decoded_vertices),decoded_faces))
    if not preserved or len(decoded_faces)!=len(faces):
        raise ValueError('BPT round-trip changed oriented quantized grid faces')
    report=dict(fixture='65x65 planar grid',vertices=len(vertices),faces=len(faces),
        tokens=len(tokens),tokens_per_face=len(tokens)/len(faces),
        fraction_of_raw_9_tokens=len(tokens)/(9*len(faces)),
        sequence_reduction=1-len(tokens)/(9*len(faces)),
        patches=int(np.count_nonzero(tokens<512)),encode_seconds=elapsed,
        quantization=before,oriented_quantized_faces_preserved=preserved)
    out=Path(__file__).resolve().parents[1]/'artifacts/bpt'
    out.mkdir(parents=True,exist_ok=True)
    np.save(out/'grid_tokens.npy',tokens)
    (out/'grid_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
