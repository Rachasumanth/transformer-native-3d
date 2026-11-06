"""Create reproducible notebook source archives without filesystem metadata."""
import io
from pathlib import Path
import zipfile


def source_bundle(root, files):
    root = Path(root)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(map(Path, files)):
            entry = zipfile.ZipInfo(path.relative_to(root).as_posix(), (1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, path.read_bytes())
    return buffer.getvalue()
