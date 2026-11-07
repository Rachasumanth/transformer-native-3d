"""Build (and optionally push) Kaggle training kernels for FORGE-3D.

The notebook embeds the repository source as a base64 zip, mounts the corpus
dataset read-only, extracts shards to /kaggle/working, optionally resumes from
a checkpoint dataset, and runs a FORGE training module.
"""
import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path

from tools.notebook_bundle import source_bundle

ROOT = Path(__file__).resolve().parents[2]
TASKS = {
    'gct': {
        'module': 'forge.train_gct',
        'slugs': {
            'shivakrishna846': 'forge3d-gct-dual-t4',
            'sumu143': 'forge3d-gct-dual-t4',
        },
    },
    'mat': {
        'module': 'forge.train_mat',
        'slugs': {
            'shivakrishna846': 'forge3d-mat-dual-t4',
            'sumu143': 'forge3d-mat-dual-t4',
        },
    },
}


def notebook_title(slug):
    return ' '.join(word.capitalize() for word in slug.replace('-', ' ').split())


def bootstrap_code(task, corpus_dataset, resume_dataset, steps, batch, extra_env):
    module = TASKS[task]['module']
    payload = source_bundle(ROOT, sorted((ROOT / 'forge').glob('*.py')) +
                            sorted((ROOT / 't3d').glob('*.py')))
    digest = hashlib.sha256(payload).hexdigest()
    env = {
        'FORGE_MANIFEST': '/kaggle/working/corpus/manifest.json',
        'FORGE_OUT': '/kaggle/working/forge-out',
        'FORGE_STEPS': str(steps),
        'FORGE_BATCH': str(batch),
        'FORGE_WORKERS': '4',
        'FORGE_VAL_EVERY': '500',
        'FORGE_CKPT_EVERY': '1000',
        'FORGE_VAL_ASSETS': '48',
        'PYTHONUNBUFFERED': '1',
    }
    env.update(extra_env)
    lines = [
        'import base64,hashlib,io,json,os,pathlib,shutil,subprocess,sys,tarfile,zipfile',
        'import torch',
        f'payload=base64.b64decode({base64.b64encode(payload).decode()!r})',
        f'assert hashlib.sha256(payload).hexdigest()=={digest!r}',
        'root=pathlib.Path("/kaggle/working/forge3d");root.mkdir(parents=True,exist_ok=True)',
        'with zipfile.ZipFile(io.BytesIO(payload)) as z:z.extractall(root)',
        'sys.path.insert(0,str(root))',
        'assert torch.cuda.is_available(),"CUDA not available"',
        'print("gpus",torch.cuda.device_count(),[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])',
        'inputs=pathlib.Path("/kaggle/input")',
        'manifests=[p for p in inputs.rglob("manifest.json") if "corpus" in str(p).lower()]',
        'assert manifests,"corpus manifest not found in dataset mounts"',
        'manifest=manifests[0]',
        'document=json.loads(manifest.read_text())',
        'work=pathlib.Path("/kaggle/working/corpus");work.mkdir(parents=True,exist_ok=True)',
        '(work/"assets").mkdir(parents=True,exist_ok=True)',
        'shards=sorted(p for p in manifest.parent.rglob("*.tar") if "shard" in p.name)',
        'print("shards",len(shards),"assets",len(document["assets"]))',
        'if shards:',
        '    for tar in shards:',
        '        with tarfile.open(tar) as archive:archive.extractall(work)',
        'else:',
        '    for row in document["assets"]:',
        '        name=pathlib.Path(row["path"]).name',
        '        matches=list(manifest.parent.rglob(name))',
        '        assert matches,"missing npz "+name',
        '        shutil.copy2(matches[0],work/"assets"/name)',
        '(work/"manifest.json").write_text(manifest.read_text())',
        'print("corpus assets",len(document["assets"]))',
        'resume=None',
    ]
    if resume_dataset:
        lines += [
            f'resume_candidates=[p for p in inputs.rglob("forge-*.pt") if "{resume_dataset.split("/")[-1]}" in str(p)]',
            'resume_candidates+= [p for p in inputs.rglob("forge-*.pt")]',
            'if resume_candidates:',
            '    resume=sorted(resume_candidates,key=lambda p:p.name)[-1]',
            '    print("resume",resume)',
        ]
    lines += [
        'out=pathlib.Path("/kaggle/working/forge-out");out.mkdir(parents=True,exist_ok=True)',
        'env=os.environ.copy()',
        f'env.update({env!r})',
        'if resume:env["FORGE_RESUME"]=str(resume)',
        'print("env",json.dumps(env,indent=2))',
        f'subprocess.run([sys.executable,"-m","{module}"],check=True,env=env,cwd=str(root))',
        'export=pathlib.Path("/kaggle/working/export");export.mkdir(exist_ok=True)',
        'for path in out.glob("*"):',
        '    shutil.copy2(path,export/path.name)',
        'print(sorted(p.name for p in export.iterdir()))',
    ]
    return '\n'.join(lines)


def build(task, owner, corpus_dataset, resume_dataset, steps, batch, extra_env, slug=''):
    slug = slug or TASKS[task]['slugs'].get(owner, f'forge3d-{task}-dual-t4')
    source = bootstrap_code(task, corpus_dataset, resume_dataset, steps, batch, extra_env)
    cell = {'cell_type': 'code', 'execution_count': None, 'metadata': {},
            'outputs': [], 'source': [line + '\n' for line in source.splitlines()]}
    notebook = {
        'cells': [cell],
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.11'},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }
    metadata = {
        'id': f'{owner}/{slug}',
        'title': notebook_title(slug),
        'code_file': f'{slug}.ipynb',
        'language': 'python',
        'kernel_type': 'notebook',
        'is_private': True,
        'enable_gpu': True,
        'enable_internet': False,
        'machine_shape': 'NvidiaTeslaT4',
        'dataset_sources': [source for source in (corpus_dataset, resume_dataset) if source],
    }
    directory = ROOT / 'artifacts/forge3d/kernels' / slug
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f'{slug}.ipynb').write_text(json.dumps(notebook, indent=1) + '\n')
    (directory / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return directory, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=sorted(TASKS), required=True)
    parser.add_argument('--owner', default='shivakrishna846')
    parser.add_argument('--corpus', default='shivakrishna846/forge3d-corpus-v1')
    parser.add_argument('--resume', default='')
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--env', nargs='*', default=[])
    parser.add_argument('--slug', default='')
    parser.add_argument('--push', action='store_true')
    parser.add_argument('--profile', default='7')
    args = parser.parse_args()
    extra_env = {}
    for item in args.env:
        key, _, value = item.partition('=')
        extra_env[key] = value
    directory, metadata = build(args.task, args.owner, args.corpus, args.resume,
                                args.steps, args.batch, extra_env, args.slug)
    print(directory)
    print(json.dumps(metadata, indent=2))
    if args.push:
        command = ['kaggle', 'kernels', 'push', '-p', str(directory)]
        if args.profile:
            command = ['kaggle-profile', 'run', args.profile, '--', *command]
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
