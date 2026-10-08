import io
import time

import numpy as np
from conftest import raw_image, upload
from PIL import Image
from sqlalchemy import text

from cleaner.db import Database, now
from cleaner.domain import JobRequest, Problem
from cleaner.images import compose, restore
from cleaner.lifecycle import backup, replay
from cleaner.storage import Storage


def create(client, id, **extra):
    return client.post(
        "/v1/jobs", json={"input_asset_id": id, **extra}, headers={"Idempotency-Key": str(time.time_ns())}
    )


def test_extraction_and_original_preservation(setup):
    c, w, a = setup
    raw = raw_image()
    id = upload(c, w, raw)
    assert c.get(f"/v1/assets/{id}").json()["state"] == "ready"
    assert c.get(f"/v1/assets/{id}/content?representation=original").content == raw
    j = create(c, id)
    assert j.status_code == 202, j.text
    assert w.tick()
    result = c.get("/v1/jobs/" + j.json()["job_id"]).json()
    assert result["status"] == "succeeded", result
    foreground = result["result"]["artifacts"]["foreground"]
    token = c.post(f"/v1/assets/{foreground}/download").json()["url"]
    im = Image.open(io.BytesIO(c.get(token).content))
    assert im.size == (32, 24) and im.getchannel("A").getextrema() == (0, 255)
    assert c.get(f"/v1/jobs/{result['job_id']}/plan").json()["workflow"] == "baseline_v1"
    assert "retrieval_unavailable" in result["warnings"]


def test_auth_csrf_and_host(setup):
    c, w, a = setup
    assert c.get("/v1/assets", headers={"Host": "evil.example"}).status_code == 403
    assert c.post("/v1/uploads", json={}, headers={"X-CSRF-Token": "bad"}).status_code == 403
    assert c.get("/v1/assets", headers={"Origin": "https://evil.example"}).status_code == 403
    c.cookies.clear()
    assert c.get("/v1/assets").status_code == 401


def test_idempotency_and_cancellation(setup):
    c, w, a = setup
    id = upload(c, w)
    r = c.post("/v1/jobs", json={"input_asset_id": id}, headers={"Idempotency-Key": "same"})
    same = c.post("/v1/jobs", json={"input_asset_id": id}, headers={"Idempotency-Key": "same"})
    assert r.json()["job_id"] == same.json()["job_id"]
    changed = c.post(
        "/v1/jobs", json={"input_asset_id": id, "upscale": {"scale": 2}}, headers={"Idempotency-Key": "same"}
    )
    assert changed.status_code == 409
    job = r.json()["job_id"]
    assert c.post(f"/v1/jobs/{job}/cancel").status_code == 202
    assert not w.tick()
    assert c.get("/v1/jobs/" + job).json()["status"] == "canceled"


def test_worker_fencing_and_recovery(setup):
    c, w, a = setup
    id = upload(c, w)
    jid = create(c, id).json()["job_id"]
    old = w.db.claim("old")
    with w.db.tx() as con:
        con.execute(text("UPDATE jobs SET lease=:n WHERE id=:id"), {"n": now() - 1, "id": jid})
    new = w.db.claim("new")
    assert new["token"] > old["token"]
    try:
        w.finish(old, {})
    except Problem as e:
        assert e.code == "LEASE_LOST"
    else:
        assert False, "stale worker published"
    w.finish(new, {})


def test_upload_validation_and_mask_roles(setup):
    c, w, a = setup
    id = upload(c, w)
    bad = upload(c, w, b"bad image")
    assert c.get("/v1/assets/" + bad).json()["state"] == "failed"
    assert create(c, id, output={"format": "jpeg"}).status_code == 422
    assert create(c, id, segmentation={"mode": "interactive"}).status_code == 422
    assert create(c, id, inpaint={"erase_mask_asset_id": id}).status_code == 422
    other = upload(c, w, raw_image((16, 16)), role="erase_mask", parent_asset_id=id)
    assert c.get("/v1/assets/" + other).json()["state"] == "failed"


def test_search_and_delete(setup):
    c, w, a = setup
    id = upload(c, w)
    assert c.post(f"/v1/assets/{id}/index").status_code == 202
    w.tick()
    sj = c.post("/v1/searches", json={"query": "red object"}).json()["job_id"]
    w.tick()
    assert c.get("/v1/searches/" + sj).json()["result"]["results"][0]["asset_id"] == id
    token = c.post(f"/v1/assets/{id}/download").json()["url"]
    assert c.delete("/v1/assets/" + id).status_code == 202
    assert c.get(token).status_code == 404
    assert not w.db.all("SELECT * FROM embeddings")
    assert c.get("/v1/assets").json() == []


def test_feedback_consent_revocation(setup):
    c, w, a = setup
    id = upload(c, w)
    c.put("/v1/consent", json={"enabled": True})
    j = create(c, id, memory={"allow_indexing": True}).json()["job_id"]
    w.tick()
    assert c.post(f"/v1/jobs/{j}/feedback", json={"accepted": True, "revision": 1}).status_code == 200
    ex = w.db.one("SELECT * FROM examples")
    assert ex and w.memory.live_example(ex["id"])
    c.put("/v1/consent", json={"enabled": False})
    assert w.memory.live_example(ex["id"]) is None
    assert c.post(f"/v1/jobs/{j}/feedback", json={"accepted": False, "revision": 1}).status_code == 409


def test_backup_restore_deletion_replay(setup, tmp_path):
    c, w, a = setup
    id = upload(c, w)
    dest = tmp_path / "backup"
    backup(w.db, w.storage, dest)
    c.delete("/v1/assets/" + id)
    import shutil

    shutil.copytree(w.storage.path("tombstones"), dest / "tombstones", dirs_exist_ok=True)
    restored = Database(dest / "library.sqlite3")
    replay(restored, Storage(dest))
    assert restored.one("SELECT deleted FROM assets WHERE id=:id", id=id)["deleted"]
    assert not (dest / f"quarantine/{id}/original").exists()


def test_outside_inpaint_mask_unchanged():
    source = Image.fromarray(np.random.default_rng(42).integers(0, 256, (40, 40, 3), dtype=np.uint8)).convert("RGBA")
    mask = Image.new("L", (40, 40))
    mask.paste(255, (15, 15, 20, 20))
    req = JobRequest(
        input_asset_id="00000000-0000-4000-8000-000000000001",
        inpaint={"erase_mask_asset_id": "00000000-0000-4000-8000-000000000002", "feather_px": 0},
    ).model_dump(mode="json")
    result, _ = restore(source, req, mask)
    outside = np.asarray(mask) == 0
    assert np.array_equal(np.asarray(source)[outside], np.asarray(result)[outside])


def test_alpha_resize_avoids_hidden_color_bleed():
    data = np.zeros((4, 4, 4), np.uint8)
    data[:, :, :3] = [255, 0, 0]
    data[1:3, 1:3] = [0, 0, 255, 255]
    result = np.asarray(compose(Image.fromarray(data), np.ones((4, 4), np.float32), 2))
    assert result.shape == (8, 8, 4)
    assert (result[:, :, 0][result[:, :, 3] > 5] == 0).all()


def test_cross_tenant_denied(setup):
    c, w, a = setup
    id = upload(c, w)
    with w.db.tx() as con:
        con.execute(
            text(
                "INSERT INTO assets(id,tenant_id,role,state,filename,mime,bytes,sha256,created,expires) SELECT 'foreign','another',role,state,filename,mime,bytes,sha256,created,expires FROM assets WHERE id=:id"
            ),
            {"id": id},
        )
    id = "foreign"
    assert c.get("/v1/assets/" + id).status_code == 404


def test_queue_limit_and_deadline(setup):
    c, w, _ = setup
    id = upload(c, w)
    jobs = [create(c, id) for _ in range(4)]
    assert [r.status_code for r in jobs] == [202, 202, 202, 429]
    jid = jobs[0].json()["job_id"]
    with w.db.tx() as con:
        con.execute(text("UPDATE jobs SET deadline=:d WHERE id=:id"), {"d": now() - 1, "id": jid})
    w.tick()
    assert c.get("/v1/jobs/" + jid).json()["error"]["code"] == "DEADLINE_EXCEEDED"


def test_deletion_during_processing_blocks_publication(setup):
    c, w, _ = setup
    id = upload(c, w)
    jid = create(c, id).json()["job_id"]
    real = w.runner

    def deleting(operation, **args):
        result = real(operation, **args)
        if operation == "segment":
            c.delete("/v1/assets/" + id)
        return result

    w.runner = deleting
    w.tick()
    assert c.get("/v1/jobs/" + jid).status_code == 404
    assert w.db.one("SELECT COUNT(*) n FROM assets WHERE role='foreground'")["n"] == 0


def test_model_integrity_never_falls_back(setup):
    c, w, _ = setup
    id = upload(c, w)
    j = create(c, id, segmentation={"allow_lightweight_fallback": True}).json()["job_id"]
    real = w.runner
    calls = []

    def broken(operation, **args):
        if operation == "segment":
            calls.append(args)
            raise Problem("MODEL_INTEGRITY", "corrupted artifact", 503)
        return real(operation, **args)

    w.runner = broken
    w.tick()
    assert c.get("/v1/jobs/" + j).json()["error"]["code"] == "MODEL_INTEGRITY"
    assert len(calls) == 1


def test_oom_fallback_is_explicit_and_not_interactive(setup):
    c, w, _ = setup
    id = upload(c, w)
    real = w.runner

    def oom(operation, **args):
        if operation == "segment" and not args.get("lightweight"):
            raise Problem("INFERENCE_OOM", "Out of memory", 503)
        return real(operation, **args)

    w.runner = oom
    j = create(c, id, segmentation={"allow_lightweight_fallback": True}).json()["job_id"]
    w.tick()
    assert c.get("/v1/jobs/" + j).json()["status"] == "succeeded"
    j = create(
        c,
        id,
        segmentation={
            "mode": "interactive",
            "allow_lightweight_fallback": True,
            "points": [{"x": 0.5, "y": 0.5, "label": 1}],
        },
    ).json()["job_id"]
    w.tick()
    assert c.get("/v1/jobs/" + j).json()["error"]["code"] == "INFERENCE_OOM"


def test_write_failure_does_not_publish_partial_output(setup, monkeypatch):
    c, w, _ = setup
    id = upload(c, w)
    j = create(c, id).json()["job_id"]
    real = w.storage.write
    count = 0

    def diskfull(key, data):
        nonlocal count
        if key.startswith("jobs/"):
            count += 1
            if count == 3:
                raise OSError(28, "disk full")
        return real(key, data)

    monkeypatch.setattr(w.storage, "write", diskfull)
    w.tick()
    assert c.get("/v1/jobs/" + j).json()["status"] == "failed"
    assert w.db.one("SELECT COUNT(*) n FROM assets WHERE role='foreground'")["n"] == 0


def test_expiry_never_resurrects_asset(setup):
    from cleaner.lifecycle import maintain

    c, w, _ = setup
    id = upload(c, w)
    with w.db.tx() as con:
        con.execute(text("UPDATE assets SET expires=:n WHERE id=:id"), {"n": now() - 1, "id": id})
    assert c.get("/v1/assets/" + id).status_code == 404
    maintain(w.db, w.storage, force=True)
    assert w.db.one("SELECT deleted FROM assets WHERE id=:id", id=id)["deleted"]
    assert c.get("/v1/assets/" + id).status_code == 404


def test_transient_retry_is_bounded(setup):
    c, w, _ = setup
    id = upload(c, w)
    j = create(c, id).json()["job_id"]

    def unavailable(*args, **kwargs):
        raise OSError(5, "temporary IO")

    w.runner = unavailable
    for attempt in range(3):
        assert w.tick()
        with w.db.tx() as con:
            con.execute(text("UPDATE jobs SET available=0 WHERE id=:id"), {"id": j})
    assert c.get("/v1/jobs/" + j).json()["status"] == "failed"
    assert w.db.job(j)["attempt"] == 3


def test_sse_resume(setup):
    c, w, _ = setup
    id = upload(c, w)
    j = create(c, id).json()["job_id"]
    w.tick()
    rows = w.db.all("SELECT * FROM events WHERE job_id=:j ORDER BY id", j=j)
    r = c.get(f"/v1/jobs/{j}/events", headers={"Last-Event-ID": str(rows[-2]["id"])})
    assert r.status_code == 200
    assert "succeeded" in r.text and f"id: {rows[0]['id']}\n" not in r.text
