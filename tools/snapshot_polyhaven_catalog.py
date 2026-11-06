"""Snapshot Poly Haven's CC0 model metadata for Track B candidate selection."""
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.request import Request, urlopen


def get_json(url):
    request = Request(url, headers={'User-Agent':
        'Transformer-Native-3D research corpus screen/1.0'})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def main():
    models = get_json('https://api.polyhaven.com/assets?t=models')
    candidates = []
    for asset_id, info in models.items():
        polycount = info.get('polycount')
        if info.get('type') != 2 or not isinstance(polycount, int) or polycount < 8000:
            continue
        candidates.append(dict(asset_id=asset_id, name=info.get('name'),
            reported_polycount=polycount, categories=info.get('categories', []),
            authors=info.get('authors', {}), description=info.get('description'),
            source_model_hash=info.get('files_hash'),
            asset_page=f'https://polyhaven.com/a/{asset_id}'))
    candidates.sort(key=lambda row: (row['reported_polycount'], row['asset_id']))
    report = dict(source='https://api.polyhaven.com/assets?t=models',
        snapshot_utc=datetime.now(timezone.utc).isoformat(),
        license='CC0-1.0 per https://polyhaven.com/license',
        total_model_assets=len(models), minimum_reported_polycount=8000,
        candidate_count=len(candidates), candidates=candidates)
    root = Path(__file__).resolve().parents[1]
    output = root/'artifacts/bpt-candidates/polyhaven-catalog-screen.json'
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(f'Models: {len(models)}; candidates: {len(candidates)}; wrote {output}')


if __name__ == '__main__':
    main()
