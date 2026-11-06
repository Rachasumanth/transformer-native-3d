"""Package two upstream CC0 integration assets; not a curated training corpus."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]

def main():
    output=ROOT/'artifacts/pilot-dataset'
    output.mkdir(parents=True,exist_ok=True)
    records=[]
    for name, model, split in [('avocado','Avocado','train'),('waterbottle','WaterBottle','validation')]:
        source=ROOT/'artifacts/import'/name
        prepared=source/f'{name}.t3d'
        shutil.copyfile(prepared,output/prepared.name)
        shutil.copyfile(source/'SOURCE.md',output/f'{name}-SOURCE.md')
        shutil.copyfile(source/f'{name}.import.json',output/f'{name}.import.json')
        records.append(dict(asset_id=f'khronos-{name}',path=prepared.name,
                            sha256=hashlib.sha256(prepared.read_bytes()).hexdigest(),split=split,
                            source_sha256=json.loads((source/f'{name}.import.json').read_text())['source_sha256'],
                            source=f'https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/{model}',
                            license='CC0-1.0; Microsoft, 2017; retained upstream SOURCE.md'))
    (output/'manifest.json').write_text(json.dumps(dict(version=1,assets=records),indent=2)+'\n')
    metadata=dict(title='Transformer Native 3D Pilot',id='sumu143/transformer-native-3d-pilot',
                  licenses=[dict(name='CC0-1.0')],
                  description='Two low-resolution prepared Khronos CC0 sample assets for pipeline integration. Avocado train; WaterBottle validation. Not a curated corpus or quality benchmark.')
    (output/'dataset-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(output)

if __name__=='__main__':main()
