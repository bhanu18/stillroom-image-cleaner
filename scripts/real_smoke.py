"""Run actual adapters through API/worker in an isolated smoke library."""

import hashlib
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from cleaner.api import create_app
from cleaner.benchmark import measurement
from cleaner.config import Settings
from cleaner.worker import Worker


def main():
    source = Path(os.environ.get("MODEL_CHECK_ROOT", "/tmp/stillroom-model-check"))
    root = Path(tempfile.mkdtemp(prefix="stillroom-real-e2e-"))
    (root / "models").symlink_to(source / "models", target_is_directory=True)
    settings = replace(Settings.env(), data=root, origin="http://testserver")
    app = create_app(settings)
    worker = Worker(settings)
    expected = os.environ.get("EXPECT_INFERENCE_DEVICE")
    measurements = []
    original_run = worker.run_model

    def measured(job, operation, **args):
        value = original_run(job, operation, **args)
        if operation != "decode":
            diagnostics = value["diagnostics"]
            if expected:
                assert diagnostics["device"] == expected, diagnostics
                assert not diagnostics.get("fallback_reason"), diagnostics
            measurements.append({"operation": operation, "query": bool(args.get("query")), **diagnostics})
        return value

    worker.run_model = measured
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
            {"mode": "automatic", "allow_lightweight_fallback": False},
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
        # The automatic path uses BiRefNet; exercise U2-Net explicitly as well.
        _, u2net = measurement(source, "u2net", source / "fixtures/red-bag.png", None,
                               expected or (settings.device if settings.device != "auto" else "cpu"))
        if expected:
            assert u2net["device"] == expected, u2net
            assert any(row["query"] for row in measurements)
            assert any(row["operation"] == "embed" and not row["query"] for row in measurements)
            assert {"siglip", "sam2", "birefnet"} <= {
                row["manifest"].split("_")[0] for row in measurements
            }
        report = {
            "root": str(root),
            "real_models": True,
            "synthetic_fixture_only": True,
            "jobs": results,
            "inference": measurements,
            "u2net": u2net,
            "search": "passed",
        }
        (source / "benchmarks").mkdir(exist_ok=True)
        (source / "benchmarks/real-e2e.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
