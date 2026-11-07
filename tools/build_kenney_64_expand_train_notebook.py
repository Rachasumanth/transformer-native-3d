"""Bundle the expanded 19-kit in-distribution 64³ training notebook for Kaggle GPU."""
import argparse
import base64, hashlib, json
from pathlib import Path

from tools.notebook_bundle import source_bundle

root = Path.cwd()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--owner', default='sumu143')
    p.add_argument('--slug', default='transformer-native-3d-kenney-64-expand-train')
    p.add_argument('--steps', default='20000')
    p.add_argument('--batch', default='8')
    p.add_argument('--limit', default='0')
    p.add_argument('--expanded-dataset', default='rahul09865/t3d-kenney-64-id-expanded')
    return p.parse_args()


def source_code(a):
    files = [
        *sorted((root / 't3d').glob('*.py')),
        root / 'tools/train_kenney_64_occupancy_dit.py',
        root / 'tools/train_kenney_64_geometry_dit.py',
        root / 'tools/make_id_split.py',
        root / 'tools/generate_corpus_prompts.py',
    ]
    payload = source_bundle(root, files)
    digest = hashlib.sha256(payload).hexdigest()
    source = (
        'import base64,hashlib,io,pathlib,subprocess,sys,zipfile,json,os,re,shutil,importlib.util\n'
        f'payload=base64.b64decode({base64.b64encode(payload).decode()!r})\n'
        f'assert hashlib.sha256(payload).hexdigest()=={digest!r}\n'
        'root=pathlib.Path("/kaggle/working/t3d-expand-train");root.mkdir(exist_ok=True)\n'
        'with zipfile.ZipFile(io.BytesIO(payload)) as z:z.extractall(root)\n'
        'assert importlib.util.find_spec("transformers") is not None, "transformers not preinstalled"\n'
        'assert __import__("torch").cuda.is_available(), "CUDA not available in kernel"\n'
        'input_root=pathlib.Path("/kaggle/input")\n'
        'clip_configs=[p for p in input_root.rglob("config.json") if "t3d-clip-text-base-patch32" in str(p)]\n'
        'assert clip_configs,"CLIP snapshot not found in dataset mounts"\n'
        'clip_dir=str(clip_configs[0].parent)\n'
        'manifests=[p for p in input_root.rglob("manifest.json") if "kenney-corpus-64-id-expanded" in str(p)]\n'
        'if not manifests:\n'
        '    zips=list(input_root.rglob("kenney-corpus-64-id-expanded.zip"))\n'
        '    assert zips,"expanded corpus manifest/zip not found"\n'
        '    shutil.unpack_archive(str(zips[0]),"/kaggle/working/expanded-unpacked")\n'
        '    manifests=[p for p in pathlib.Path("/kaggle/working/expanded-unpacked").rglob("manifest.json") if "kenney-corpus-64-id-expanded" in str(p)]\n'
        'assert manifests,"expanded corpus manifest not found"\n'
        'manifest=manifests[0]\n'
        'vaes=[p for p in input_root.rglob("triton-kenney-corpus-64-*.pt")]\n'
        'if not vaes:\n'
        '    zips=list(input_root.rglob("checkpoints.zip"))\n'
        '    assert zips,"VAE checkpoint/zip not found"\n'
        '    shutil.unpack_archive(str(zips[0]),"/kaggle/working/vae-unpacked")\n'
        '    vaes=[p for p in pathlib.Path("/kaggle/working/vae-unpacked").rglob("triton-kenney-corpus-64-*.pt")]\n'
        'assert vaes,"VAE checkpoint not found"\n'
        'def step(p):\n'
        '    m=re.search(r"(\\d+)\\.pt$",p.name);return int(m.group(1)) if m else 0\n'
        'vae=max(vaes,key=step)\n'
        'env={**os.environ,"PYTHONPATH":str(root),"T3D_MANIFEST":str(manifest),"T3D_CLIP_DIR":clip_dir}\n'
        f'subprocess.run([sys.executable,"-u","-m","tools.train_kenney_64_occupancy_dit"],cwd=root,env={{**env,"T3D_STEPS":"{a.steps}","T3D_BATCH":"{a.batch}","T3D_OUT":"/kaggle/working"}},check=True)\n'
        f'subprocess.run([sys.executable,"-u","-m","tools.train_kenney_64_geometry_dit"],cwd=root,env={{**env,"T3D_VAE":str(vae),"T3D_STEPS":"{a.steps}","T3D_OUT":"/kaggle/working"}},check=True)\n'
        f'for name in ("occupancy-dit-64-report.json","geometry-dit-64-report.json","occupancy-dit-64-{a.steps}.pt","geometry-dit-64-{a.steps}.pt"):\n'
        '    p=pathlib.Path("/kaggle/working")/name;assert p.exists(),f"missing {name}"\n'
        'summary={"source_sha256":'+repr(digest)+',"manifest":str(manifest),"vae":str(vae),"occupancy":json.loads((pathlib.Path("/kaggle/working")/"occupancy-dit-64-report.json").read_text()),"geometry":json.loads((pathlib.Path("/kaggle/working")/"geometry-dit-64-report.json").read_text())}\n'
        'pathlib.Path("/kaggle/working/expand-training-summary.json").write_text(json.dumps(summary,indent=2)+"\\n")\n'
        'print(json.dumps({"occupancy_iou":summary["occupancy"].get("mean_val_occupancy_iou"),"geometry_dist":summary["geometry"].get("mean_surface_distance"),"geometry_f":summary["geometry"].get("mean_fscore_001")}))\n'
    )
    return out_nb(source), kernel_meta(source, a, digest)


def out_nb(source):
    return json.dumps({
        'cells': [{
            'cell_type': 'code',
            'id': 'train',
            'metadata': {},
            'execution_count': None,
            'outputs': [],
            'source': source
        }],
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
        'nbformat': 4,
        'nbformat_minor': 5
    }, indent=2) + '\n'


def kernel_meta(source, a, digest):
    return json.dumps({
        'id': f'{a.owner}/{a.slug}',
        'title': ' '.join(w.capitalize() for w in a.slug.replace('-', ' ').split()),
        'code_file': 'kenney_64_expand_train.ipynb',
        'language': 'python',
        'kernel_type': 'notebook',
        'is_private': True,
        'enable_gpu': True,
        'enable_internet': False,
        'machine_shape': 'NvidiaTeslaT4',
        'dataset_sources': [a.expanded_dataset, 'rahul09865/t3d-kenney-64-pipeline', 'rahul09865/t3d-clip-text-base-patch32'],
        'competition_sources': [],
        'kernel_sources': [],
        'model_sources': []
    }, indent=2) + '\n'


def main():
    args = parse_args()
    nb, meta = source_code(args)
    out = root / 'kaggle-kenney-64-expand-train'
    out.mkdir(exist_ok=True)
    (out / 'kenney_64_expand_train.ipynb').write_text(nb)
    (out / 'kernel-metadata.json').write_text(meta)
    print(f"Generated expanded-training 64³ notebook in {out} (owner={args.owner}, steps={args.steps}, limit={args.limit})")


if __name__ == '__main__':
    main()
