import hashlib
import os
from pathlib import Path

from .domain import Problem


class Storage:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def path(self, key):
        p = (self.root / key).resolve()
        if not p.is_relative_to(self.root):
            raise Problem("INVALID_PATH", "Invalid asset path")
        return p

    def write(self, key, data):
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        from uuid import uuid4

        tmp = p.with_name(p.name + f".{uuid4().hex}.tmp")
        try:
            with tmp.open("xb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, p)
            fd = os.open(p.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            tmp.unlink(missing_ok=True)
        return hashlib.sha256(data).hexdigest()
