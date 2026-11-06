"""Build a CPU Kaggle notebook for one interleaved 64³ catalog conversion shard."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
from tools.notebook_bundle import source_bundle


def main():
    p = argparse.ArgumentParser(description='Build 64³ Kaggle conversion shard notebook')
    p.add_argument('--shard', type=int, required=True)
    p.add_argument('--shards', type=int, default=5)
    args = p.parse_args()

    if not 0 <= args.shard < args.shards:
        raise ValueError('Invalid shard')

    root = Path.cwd()
    files = [*sorted((root / 't3d').glob('*.py')), root / 'tools/prepare_kenney_catalog.py', root / 'artifacts/kenney-raw-pilot/candidate-index.json']
    payload = source_bundle(root, files)
    digest = hashlib.sha256(payload).hexdigest()
    label = f'kenney-64-shard-{args.shard}'

    source = (
        'import base64,hashlib,io,pathlib,subprocess,sys,zipfile,json,shutil\n'
        f'payload=base64.b64decode({base64.b64encode(payload).decode()!r})\n'
        f'assert hashlib.sha256(payload).hexdigest()=={digest!r}\n'
        'root=pathlib.Path("/kaggle/working/corpus-code");root.mkdir(exist_ok=True)\n'
        'with zipfile.ZipFile(io.BytesIO(payload)) as z:z.extractall(root)\n'
        'subprocess.run([sys.executable,"-m","pip","install","trimesh>=4,<5","hilbertcurve>=2,<3"],check=True)\n'
        'matches=list(pathlib.Path("/kaggle/input").rglob("acquisition-report.json"))\n'
        'print("raw acquisition reports:",[str(p) for p in matches],flush=True)\n'
        'assert len(matches)==1,"Expected exactly one mounted raw acquisition report"\n'
        'raw_root=matches[0].parent\n'
        f'out=pathlib.Path("/tmp/{label}")\n'
        f'subprocess.run([sys.executable,"-m","tools.prepare_kenney_catalog","--index",str(root/"artifacts/kenney-raw-pilot/candidate-index.json"),"--raw-root",str(raw_root),"--output",str(out),"--grid-size","64","--start","{args.shard}","--stop","944","--stride","{args.shards}","--workers","4"],cwd=root,check=True)\n'
        'report=json.loads((out/"conversion-report.json").read_text())\n'
        f'report["source_sha256"]={digest!r};(pathlib.Path("/kaggle/working")/"{label}-report.json").write_text(json.dumps(report,indent=2)+"\\n")\n'
        f'shutil.copyfile(out.parent/(out.name+".zip"),pathlib.Path("/kaggle/working")/"{label}.zip")\n'
    )

    # Active Kaggle username
    try:
        import kaggle
        owner = kaggle.api.get_config_value('username') or 'dileepkumarchakali'
    except Exception:
        owner = 'dileepkumarchakali'

    out = root / f'kaggle-kenney-corpus-64-{args.shard}'
    out.mkdir(exist_ok=True)
    (out / 'corpus.ipynb').write_text(
        json.dumps({
            'cells': [{'cell_type': 'code', 'id': 'corpus', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': source}],
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
            'nbformat': 4,
            'nbformat_minor': 5
        }, indent=2) + '\n'
    )
    (out / 'kernel-metadata.json').write_text(
        json.dumps({
            'id': f'{owner}/t3d-kenney-corpus-64-conversion-shard-{args.shard}',
            'title': f'T3D Kenney Corpus 64 Conversion Shard {args.shard}',
            'code_file': 'corpus.ipynb',
            'language': 'python',
            'kernel_type': 'notebook',
            'is_private': True,
            'enable_gpu': False,
            'enable_internet': True,
            'dataset_sources': [f'{owner}/t3d-kenney-pinned-raw-catalog'],
            'competition_sources': [],
            'kernel_sources': [],
            'model_sources': []
        }, indent=2) + '\n'
    )
    print(f'Built shard {args.shard} for owner {owner} with code bundle digest: {digest}')


if __name__ == '__main__':
    main()
