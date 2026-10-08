"""Isolated browser-test backend. Never used by the production launcher."""

import tempfile
import threading
from pathlib import Path

import numpy as np
import uvicorn
from PIL import Image

from cleaner.api import create_app
from cleaner.config import Settings
from cleaner.images import decode, png
from cleaner.worker import Worker

root = Path(tempfile.mkdtemp(prefix="stillroom-e2e-"))
settings = Settings(data=root, origin="http://127.0.0.1:8765")
app = create_app(settings)
(root / "bootstrap.secret").write_text("browser-test-only")
app = create_app(settings)


def runner(operation, **args):
    if operation == "decode":
        im, transform = decode(Path(args["path"]).read_bytes())
        return {"png": png(im), "transform": transform, "size": im.size}
    if operation == "embed":
        return {"vector": [1.0] + [0.0] * 767, "manifest": "browser-fixture", "diagnostics": {"device": "test"}}
    im = Image.open(args["path"])
    mask = np.zeros((im.height, im.width), np.float32)
    mask[4:-4, 4:-4] = 1
    return {"mask": mask, "diagnostics": {"device": "test", "manifest": "browser-fixture"}}


worker = Worker(settings, runner)


def loop():
    import time

    while True:
        worker.tick()
        time.sleep(0.1)


threading.Thread(target=loop, daemon=True).start()
uvicorn.run(app, host="127.0.0.1", port=8765, access_log=False)
