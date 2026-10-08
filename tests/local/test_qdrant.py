import os

import pytest
from conftest import upload
from test_app import create

from cleaner.memory import MemoryIndex


@pytest.mark.skipif(
    not os.getenv("QDRANT_TEST_URL"), reason="Set QDRANT_TEST_URL to a disposable loopback Qdrant service"
)
def test_live_projection_revocation_rebuild(setup):
    c, w, a = setup
    w.memory = MemoryIndex(w.db, os.environ["QDRANT_TEST_URL"])
    id = upload(c, w)
    c.put("/v1/consent", json={"enabled": True})
    job = create(c, id, memory={"allow_indexing": True}).json()["job_id"]
    w.tick()
    assert c.post(f"/v1/jobs/{job}/feedback", json={"accepted": True, "revision": 1}).status_code == 200
    w.memory.sync()
    example = w.db.one("SELECT * FROM examples")
    import json

    embedding = w.db.one("SELECT * FROM embeddings")
    features = json.loads(w.db.one("SELECT features FROM analyses")["features"])
    result, warnings = w.memory.retrieve(
        {"vector": json.loads(embedding["vector"]), "manifest": embedding["manifest"]}, features, "background_remove"
    )
    assert not warnings
    assert any(r["example_id"] == example["id"] for r in result)
    assert w.memory.rebuild() == 1
    c.put("/v1/consent", json={"enabled": False})
    # Index intentionally stale: DB must reject revoked memory before the delete projection.
    result, warnings = w.memory.retrieve(
        {"vector": json.loads(embedding["vector"]), "manifest": embedding["manifest"]}, features, "background_remove"
    )
    assert not result
    w.memory.sync()
    assert w.db.one("SELECT COUNT(*) n FROM outbox WHERE delivered IS NULL")["n"] == 0
