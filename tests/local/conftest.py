import hashlib
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from cleaner.api import create_app
from cleaner.config import Settings
from cleaner.worker import Worker


@pytest.fixture
def setup(tmp_path):
    settings = Settings(data=tmp_path, origin="http://testserver")
    app = create_app(settings)
    with TestClient(app) as client:
        secret = (tmp_path / "bootstrap.secret").read_text()
        r = client.post("/v1/session/bootstrap", json={"secret": secret}, headers={"Origin": "http://testserver"})
        assert r.status_code == 200
        client.headers.update({"Origin": "http://testserver", "X-CSRF-Token": r.json()["csrf"]})

        def runner(operation, **args):
            if operation == "decode":
                from cleaner.images import decode, png

                image, transform = decode(open(args["path"], "rb").read())
                return {"png": png(image), "transform": transform, "size": image.size}
            if operation == "embed":
                vector = [0.0] * 768
                vector[0] = 1
                return {"vector": vector, "manifest": "test_siglip", "diagnostics": {"device": "cpu"}}
            im = Image.open(args["path"])
            mask = np.zeros((im.height, im.width), np.float32)
            mask[4:-4, 4:-4] = 1
            return {"mask": mask, "diagnostics": {"manifest": "test_birefnet", "device": "cpu"}}

        worker = Worker(settings, runner)
        yield client, worker, app


def raw_image(size=(32, 24), alpha=255):
    im = Image.new("RGBA", size, (230, 30, 40, alpha))
    out = io.BytesIO()
    im.save(out, format="PNG")
    return out.getvalue()


def upload(client, worker, raw=None, **extra):
    raw = raw or raw_image()
    r = client.post(
        "/v1/uploads",
        json={
            "filename": "photo.png",
            "mime": "image/png",
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            **extra,
        },
    )
    assert r.status_code == 201, r.text
    data = r.json()
    r = client.put(data["upload_url"], content=raw)
    assert r.status_code == 200, r.text
    r = client.post(f"/v1/uploads/{data['asset_id']}/complete")
    assert r.status_code == 202, r.text
    assert worker.tick()
    return data["asset_id"]
