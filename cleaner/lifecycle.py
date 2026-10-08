import json
import shutil
import sqlite3
from pathlib import Path

from sqlalchemy import text

from .config import TENANT
from .db import dumps, now, uid
from .locks import lifecycle_lock


def delete_asset(db, storage, id, *, expired=False):
    if not expired:
        db.asset(id)
    with db.tx() as c:
        ids = [
            r[0]
            for r in c.execute(
                text("""WITH RECURSIVE tree(id) AS (SELECT id FROM assets WHERE id=:id AND tenant_id=:t
 UNION ALL SELECT a.id FROM assets a JOIN tree ON a.parent_id=tree.id) SELECT id FROM tree"""),
                {"id": id, "t": TENANT},
            )
        ]
        for aid in ids:
            c.execute(text("UPDATE assets SET deleted=:n,state='deleted' WHERE id=:id"), {"n": now(), "id": aid})
            c.execute(
                text(
                    "UPDATE jobs SET canceled=1,status=CASE WHEN status IN ('queued','failed') THEN 'canceled' ELSE status END WHERE asset_id=:id"
                ),
                {"id": aid},
            )
            c.execute(text("DELETE FROM embeddings WHERE asset_id=:id"), {"id": aid})
            c.execute(text("DELETE FROM analyses WHERE asset_id=:id"), {"id": aid})
            rows = c.execute(text("SELECT id,revision FROM examples WHERE asset_id=:id"), {"id": aid}).mappings().all()
            for row in rows:
                c.execute(
                    text("UPDATE examples SET approved=0,deleted=:n,revision=revision+1 WHERE id=:id"),
                    {"n": now(), "id": row["id"]},
                )
                c.execute(
                    text("INSERT OR IGNORE INTO outbox VALUES (:id,:e,:r,:n,NULL)"),
                    {"id": uid(), "e": row["id"], "r": row["revision"] + 1, "n": now()},
                )
            c.execute(text("INSERT INTO audit VALUES (:id,'delete',:a,:n)"), {"id": uid(), "a": aid, "n": now()})
        # Independent immutable records survive restoring an older database.
        storage.write(f"tombstones/{uid()}.json", dumps({"assets": ids, "created": now()}).encode())


def consent(db, enabled):
    with db.tx() as c:
        c.execute(text("UPDATE tenants SET consent=:v,epoch=epoch+1 WHERE id=:t"), {"v": int(enabled), "t": TENANT})
        rows = c.execute(text("SELECT id,revision FROM examples WHERE tenant_id=:t"), {"t": TENANT}).mappings().all()
        for row in rows:
            c.execute(
                text("UPDATE examples SET approved=0,deleted=:n,revision=revision+1 WHERE id=:id"),
                {"n": now(), "id": row["id"]},
            )
            c.execute(
                text("INSERT OR IGNORE INTO outbox VALUES (:id,:e,:r,:n,NULL)"),
                {"id": uid(), "e": row["id"], "r": row["revision"] + 1, "n": now()},
            )


_last = 0


def maintain(db, storage, force=False):
    with lifecycle_lock(storage, exclusive=True):
        _maintain(db, storage, force)


def _maintain(db, storage, force=False):
    global _last
    if not force and now() - _last < 60:
        return
    _last = now()
    for row in db.all("SELECT id FROM assets WHERE deleted IS NULL AND expires<=:n", n=now()):
        delete_asset(db, storage, row["id"], expired=True)
    for row in db.all("SELECT path,canonical_path FROM assets WHERE deleted IS NOT NULL"):
        for key in (row["path"], row["canonical_path"]):
            if key:
                storage.path(key).unlink(missing_ok=True)
    with db.tx() as c:
        for table in ("sessions", "tokens", "idempotency"):
            c.execute(text(f"DELETE FROM {table} WHERE expires<:n"), {"n": now()})
        c.execute(text("DELETE FROM audit WHERE created<:n"), {"n": now() - 90 * 86400})
        c.execute(text("DELETE FROM events WHERE created<:n"), {"n": now() - 30 * 86400})
    referenced = {
        r[k]
        for r in db.all("SELECT path,canonical_path FROM assets WHERE deleted IS NULL")
        for k in ("path", "canonical_path")
        if r[k]
    }
    for folder in ("jobs", "quarantine"):
        directory = storage.path(folder)
        if not directory.exists():
            continue
        for file in directory.rglob("*"):
            if (
                file.is_file()
                and now() - file.stat().st_mtime > 86400
                and str(file.relative_to(storage.root)) not in referenced
            ):
                file.unlink()


def backup(db, storage, destination):
    with lifecycle_lock(storage):
        return _backup(db, storage, destination)


def _backup(db, storage, destination):
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=False)
    target = sqlite3.connect(dest / "library.sqlite3")
    source = sqlite3.connect(db.path)
    try:
        source.backup(target)
    finally:
        source.close()
        target.close()
    snapshot = sqlite3.connect(dest / "library.sqlite3")
    try:
        rows = snapshot.execute("SELECT path,canonical_path FROM assets WHERE deleted IS NULL").fetchall()
        for table in ("sessions", "tokens", "worker_status"):
            snapshot.execute(f"DELETE FROM {table}")
        snapshot.commit()
    finally:
        snapshot.close()
    for row in rows:
        for key in row:
            if key:
                src = storage.path(key)
                if not src.exists():
                    raise RuntimeError(f"Backup aborted: referenced asset missing: {key}")
                out = dest / key
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, out)
    if storage.path("tombstones").exists():
        shutil.copytree(storage.path("tombstones"), dest / "tombstones")
    (dest / "backup.json").write_text(
        dumps({"created": now(), "models": "reprovision separately", "qdrant": "rebuild from database"})
    )
    return dest


def replay(db, storage):
    for file in storage.path("tombstones").glob("*.json"):
        for id in json.loads(file.read_text())["assets"]:
            with db.tx() as c:
                c.execute(text("UPDATE assets SET deleted=:n,state='deleted' WHERE id=:id"), {"n": now(), "id": id})
                c.execute(text("DELETE FROM embeddings WHERE asset_id=:id"), {"id": id})
                c.execute(text("DELETE FROM analyses WHERE asset_id=:id"), {"id": id})
                c.execute(text("UPDATE examples SET approved=0,deleted=:n WHERE asset_id=:id"), {"n": now(), "id": id})
                c.execute(
                    text(
                        "UPDATE jobs SET canceled=1,status='canceled' WHERE asset_id=:id AND status NOT IN ('succeeded','failed','canceled')"
                    ),
                    {"id": id},
                )
    maintain(db, storage, force=True)
