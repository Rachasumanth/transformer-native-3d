"""Extract the source specification without dropping image-based equations."""
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'Transformer-Native 3D Format Blueprint.docx'
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
A = '{http://schemas.openxmlformats.org/drawingml/2006/main}'
R = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'


def extract():
    with zipfile.ZipFile(SOURCE) as archive:
        relationships = ET.fromstring(archive.read('word/_rels/document.xml.rels'))
        targets = {r.attrib['Id']: r.attrib['Target'] for r in relationships}
        asset_dir = ROOT / 'docs' / 'equations'
        asset_dir.mkdir(parents=True, exist_ok=True)

        def inline(element):
            parts = []
            for node in element.iter():
                if node.tag == W + 't':
                    parts.append(node.text or '')
                elif node.tag == A + 'blip':
                    target = targets[node.attrib[R + 'embed']]
                    name = Path(target).name
                    (asset_dir / name).write_bytes(archive.read('word/' + target))
                    parts.append(f' ![Source equation {name}](docs/equations/{name}) ')
            return ''.join(parts)

        body = ET.fromstring(archive.read('word/document.xml')).find(W + 'body')
        blocks = []
        start = None
        end = None
        for element in body:
            if element.tag == W + 'tbl':
                rows = [[inline(cell).replace('|', '\\|') for cell in row.findall(W + 'tc')]
                        for row in element.findall(W + 'tr')]
                text = '\n'.join('| ' + ' | '.join(row) + ' |' for row in
                                 [rows[0], ['---'] * len(rows[0]), *rows[1:]])
            else:
                text = inline(element)
            if text.startswith('Part 2:'):
                start = len(blocks)
            if text == 'Works cited':
                end = len(blocks)
            blocks.append(text)
        assert start is not None and end is not None
        provenance = ('<!-- Extracted from the supplied DOCX; equations remain original images.\n'
                      f'SHA256: {hashlib.sha256(SOURCE.read_bytes()).hexdigest()} -->\n\n')
        (ROOT / 'research.md').write_text(provenance + '\n\n'.join(blocks) + '\n')
        (ROOT / 'plan.md').write_text(provenance + '\n\n'.join(blocks[start:end]) + '\n')


if __name__ == '__main__':
    extract()
