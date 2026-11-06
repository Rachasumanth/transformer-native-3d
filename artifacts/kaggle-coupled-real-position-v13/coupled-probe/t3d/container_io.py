"""Shared integrity trailer and atomic file writing for T3DV containers."""
import hashlib
import hmac
import os
from pathlib import Path
import tempfile


TRAILER_SIZE = hashlib.sha256().digest_size


def seal(data: bytes) -> bytes:
    """Append a SHA-256 digest over the complete header and payload."""
    return data + hashlib.sha256(data).digest()


def verify_and_strip(data):
    """Verify a trailing SHA-256 digest and return a view without it."""
    view = memoryview(data)
    if len(view) < 128 + TRAILER_SIZE:
        raise ValueError('Truncated checksummed T3DV container')
    content, supplied = view[:-TRAILER_SIZE], bytes(view[-TRAILER_SIZE:])
    expected = hashlib.sha256(content).digest()
    if not hmac.compare_digest(supplied, expected):
        raise ValueError('T3DV SHA-256 integrity check failed')
    return content


def atomic_write(path, data: bytes):
    """Durably replace a file after its complete encoded bytes are available."""
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=destination.parent,
                                         prefix=f'.{destination.name}.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
        # Persist the directory entry where the platform supports directory fsync.
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
