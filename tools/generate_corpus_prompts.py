"""Generate deterministic, canonical text prompts for the Kenney 3D corpus."""
import json
from pathlib import Path


def id_to_prompt(asset_id: str) -> str:
    """Map a Kenney asset ID into an aligned natural English text prompt."""
    name = asset_id[len('kenney-'):] if asset_id.startswith('kenney-') else asset_id
    parts = name.split('-kit-')
    pack = parts[0] if len(parts) > 1 else 'asset'
    item = parts[1] if len(parts) > 1 else name

    # Clean hyphenated words
    item_clean = item.replace('-', ' ')
    article = 'an' if item_clean[0].lower() in 'aeiou' else 'a'

    pack_descriptors = {
        'car': 'vehicle asset',
        'food': 'food item',
        'furniture': 'furniture piece',
        'nature': 'nature environment asset',
        'pirate': 'pirate theme asset',
        'space': 'sci-fi space asset'
    }
    descriptor = pack_descriptors.get(pack, 'asset')
    return f"{article} {item_clean} {descriptor}"


def main():
    root = Path('artifacts/kenney-corpus-32')
    manifest_path = root / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f'{manifest_path} not found')

    manifest = json.loads(manifest_path.read_text())
    prompts = {}
    for item in manifest['assets']:
        aid = item['asset_id']
        prompts[aid] = {
            'prompt': id_to_prompt(aid),
            'split': item['split'],
            'group': item.get('source_group', '')
        }

    out_file = root / 'prompts.json'
    out_file.write_text(json.dumps(dict(version=1, total=len(prompts), prompts=prompts), indent=2) + '\n')
    print(f"Generated {len(prompts)} prompts in {out_file}")


if __name__ == '__main__':
    main()
