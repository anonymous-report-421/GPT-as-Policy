"""Small file helpers shared by the three independent components."""
import hashlib
import json
import os
from pathlib import Path
import uuid


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    # Different containers can publish the same derived ledger. A fixed .tmp
    # name is unsafe even when callers use advisory locks on shared storage.
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    stream = temporary.open('x')  # Preserve the normal open/umask permissions.
    try:
        with stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        # Only our uniquely created staging file; never another writer's file.
        temporary.unlink(missing_ok=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


class InputError(ValueError):
    """Rejected before inference/physics: safe for the caller to correct arguments."""
