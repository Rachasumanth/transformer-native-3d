"""Verify completed Kaggle SkinAE cross-validation reports and checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def audit(output, *, expected_source_sha256=None,
          manifest_path='artifacts/skin-pose-pilot/manifest.json'):
    output=Path(output);manifest_raw=Path(manifest_path).read_bytes()
    manifest=json.loads(manifest_raw);manifest_sha=hashlib.sha256(manifest_raw).hexdigest()
    expected_assets={row['name'] for row in manifest['assets']}
    if len(expected_assets)!=3:raise ValueError('This audit expects the three-source pilot')
    rows=[];source_hashes=set()
    for holdout in sorted(expected_assets):
        suffix=f'-holdout-{holdout}'
        report_path=output/f'skin-pose-training-report{suffix}.json'
        checkpoint_path=output/f'skin-pose-pilot{suffix}.pt'
        report=json.loads(report_path.read_text())
        if report['manifest_sha256']!=manifest_sha:raise ValueError(f'{holdout}: manifest identity mismatch')
        if expected_source_sha256 and report['source_sha256']!=expected_source_sha256:
            raise ValueError(f'{holdout}: source bundle identity mismatch')
        source_hashes.add(report['source_sha256'])
        if set(report['validation_assets'])!={holdout} or set(report['train_assets'])!=expected_assets-{holdout}:
            raise ValueError(f'{holdout}: source split leakage or incomplete fold')
        expected_train_poses=sum(row['poses'] for row in manifest['assets'] if row['name']!=holdout)
        expected_val_poses=next(row['poses'] for row in manifest['assets'] if row['name']==holdout)
        if report['train_pose_count']!=expected_train_poses or report['validation_pose_count']!=expected_val_poses:
            raise ValueError(f'{holdout}: pose count mismatch')
        digest=sha256(checkpoint_path)
        if report['checkpoint_sha256']!=digest:raise ValueError(f'{holdout}: checkpoint hash mismatch')
        metric=report['best_validation'][holdout]
        if not metric['kl']<metric['uniform_kl']:
            raise ValueError(f'{holdout}: held-out KL did not beat uniform prediction')
        if report['best_step']<0 or report['best_step']>report['steps']:
            raise ValueError(f'{holdout}: invalid selected checkpoint step')
        rows.append(dict(heldout_asset=holdout,train_assets=report['train_assets'],
            train_pose_count=report['train_pose_count'],validation_pose_count=report['validation_pose_count'],
            best_step=report['best_step'],steps=report['steps'],
            kl=metric['kl'],uniform_kl=metric['uniform_kl'],
            mean_absolute_weight_error=metric['mean_absolute_weight_error'],
            dominant_joint_accuracy=metric['dominant_joint_accuracy'],checkpoint_sha256=digest))
    if len(source_hashes)!=1:raise ValueError('Fold reports do not share the same code/data bundle')
    mean_kl=float(np.mean([row['kl'] for row in rows]))
    mean_uniform=float(np.mean([row['uniform_kl'] for row in rows]))
    return dict(scope='Three-fold source-asset-held-out pilot verification; not research-scale validation',
        source_sha256=next(iter(source_hashes)),
        manifest_sha256=manifest_sha,folds=rows,mean_heldout_kl=mean_kl,
        mean_uniform_kl=mean_uniform,mean_kl_beats_uniform=mean_kl<mean_uniform)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('output')
    parser.add_argument('--source-sha256')
    parser.add_argument('--manifest',default='artifacts/skin-pose-pilot/manifest.json')
    parser.add_argument('--report',default='')
    args=parser.parse_args()
    result=audit(args.output,expected_source_sha256=args.source_sha256,manifest_path=args.manifest)
    encoded=json.dumps(result,indent=2)+'\n'
    if args.report:Path(args.report).write_text(encoded)
    print(encoded,end='')


if __name__=='__main__':main()
