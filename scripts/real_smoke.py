"""Run actual adapters through API/worker in an isolated smoke library."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from cleaner.api import create_app
from cleaner.config import Settings
from cleaner.worker import Worker


def main():
    source = Path(os.environ.get("MODEL_CHECK_ROOT", "/tmp/stillroom-model-check"))
    root = Path(tempfile.mkdtemp(prefix="stillroom-real-e2e-"))
    (root / "models").symlink_to(source / "models", target_is_directory=True)
    settings = Settings(data=root, origin="http://testserver")
    app = create_app(settings)
    worker = Worker(settings)
    raw = (source / "fixtures/red-bag.png").read_bytes()
    with TestClient(app) as client:
        r = client.post(
            "/v1/session/bootstrap",
            json={"secret": (root / "bootstrap.secret").read_text()},
            headers={"Origin": "http://testserver"},
        )
        client.headers.update({"Origin": "http://testserver", "X-CSRF-Token": r.json()["csrf"]})
        reservation = client.post(
            "/v1/uploads",
            json={
                "filename": "synthetic-red-bag.png",
                "mime": "image/png",
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            },
        ).json()
        assert client.put(reservation["upload_url"], content=raw).status_code == 200
        assert client.post("/v1/uploads/" + reservation["asset_id"] + "/complete").status_code == 202
        worker.tick()
        asset = client.get("/v1/assets/" + reservation["asset_id"]).json()
        assert asset["state"] == "ready", asset
        results = []
        for selection in [
            {"mode": "automatic", "allow_lightweight_fallback": True},
            {"mode": "interactive", "points": [{"x": 0.5, "y": 0.5, "label": 1}, {"x": 0.05, "y": 0.05, "label": 0}]},
        ]:
            job = client.post(
                "/v1/jobs",
                json={
                    "input_asset_id": asset["id"],
                    "segmentation": selection,
                    "restoration": {"denoise": "classical", "tone": "classical"},
                    "upscale": {"scale": 2},
                },
                headers={"Idempotency-Key": selection["mode"]},
            ).json()
            worker.tick()
            result = client.get("/v1/jobs/" + job["job_id"]).json()
            assert result["status"] == "succeeded", result
            results.append({"mode": selection["mode"], "status": result["status"], "warnings": result["warnings"]})
        search = client.post("/v1/searches", json={"query": "a red handbag"}).json()
        worker.tick()
        result = client.get("/v1/searches/" + search["job_id"]).json()
        assert result["status"] == "succeeded", result
        assert result["result"]["results"][0]["asset_id"] == asset["id"]
        report = {
            "root": str(root),
            "real_models": True,
            "synthetic_fixture_only": True,
            "jobs": results,
            "search": "passed",
        }
        (source / "benchmarks/real-e2e.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
