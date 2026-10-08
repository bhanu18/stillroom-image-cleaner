"""SQLite is authoritative. Every publication checks its current lease and ownership."""

import hashlib
import json
import time
from contextlib import contextmanager
from uuid import uuid4

from sqlalchemy import create_engine, event, text

from .config import TENANT
from .domain import Problem


def uid():
    return str(uuid4())


def now():
    return time.time()


def dumps(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def unpack(row):
    return dict(row._mapping) if row is not None else None


SCHEMA = [
    "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)",
    "CREATE TABLE IF NOT EXISTS tenants (id TEXT PRIMARY KEY, consent INTEGER NOT NULL DEFAULT 0, epoch INTEGER NOT NULL DEFAULT 1)",
    """CREATE TABLE IF NOT EXISTS assets (id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, role TEXT NOT NULL,
 state TEXT NOT NULL, filename TEXT NOT NULL, mime TEXT NOT NULL, bytes INTEGER NOT NULL, sha256 TEXT NOT NULL,
 parent_id TEXT, path TEXT, canonical_path TEXT, width INTEGER, height INTEGER, transform TEXT,
 created REAL NOT NULL, expires REAL NOT NULL, deleted REAL, error TEXT, UNIQUE(tenant_id,id))""",
    """CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, asset_id TEXT,
 kind TEXT NOT NULL, request TEXT NOT NULL, status TEXT NOT NULL, stage TEXT NOT NULL DEFAULT 'queued',
 created REAL NOT NULL, deadline REAL NOT NULL, token INTEGER NOT NULL DEFAULT 0, lease REAL,
 owner TEXT, attempt INTEGER NOT NULL DEFAULT 0, available REAL NOT NULL DEFAULT 0,
 canceled INTEGER NOT NULL DEFAULT 0, plan TEXT, result TEXT, error TEXT, warnings TEXT NOT NULL DEFAULT '[]',
 UNIQUE(tenant_id,id), FOREIGN KEY(tenant_id,asset_id) REFERENCES assets(tenant_id,id))""",
    """CREATE TABLE IF NOT EXISTS attempts (id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id),
 token INTEGER NOT NULL, stage TEXT NOT NULL, started REAL NOT NULL, ended REAL, diagnostics TEXT,
 UNIQUE(job_id,token,stage))""",
    """CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id TEXT NOT NULL,
 job_id TEXT NOT NULL REFERENCES jobs(id), payload TEXT NOT NULL, created REAL NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS idempotency (tenant_id TEXT NOT NULL, endpoint TEXT NOT NULL, key TEXT NOT NULL,
 hash TEXT NOT NULL, result TEXT NOT NULL, expires REAL NOT NULL, PRIMARY KEY(tenant_id,endpoint,key))""",
    """CREATE TABLE IF NOT EXISTS embeddings (asset_id TEXT NOT NULL, tenant_id TEXT NOT NULL,
 manifest TEXT NOT NULL, vector TEXT NOT NULL, PRIMARY KEY(asset_id,manifest),
 FOREIGN KEY(tenant_id,asset_id) REFERENCES assets(tenant_id,id))""",
    """CREATE TABLE IF NOT EXISTS analyses (asset_id TEXT PRIMARY KEY REFERENCES assets(id), version TEXT NOT NULL, features TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS feedback (job_id TEXT PRIMARY KEY REFERENCES jobs(id), accepted INTEGER NOT NULL,
 revision INTEGER NOT NULL, reason TEXT NOT NULL, created REAL NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS examples (id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, job_id TEXT NOT NULL UNIQUE REFERENCES jobs(id),
 asset_id TEXT NOT NULL REFERENCES assets(id), workflow TEXT NOT NULL, manifest TEXT NOT NULL, epoch INTEGER NOT NULL,
 approved INTEGER NOT NULL, revision INTEGER NOT NULL, deleted REAL, UNIQUE(tenant_id,asset_id,workflow,manifest))""",
    """CREATE TABLE IF NOT EXISTS outbox (id TEXT PRIMARY KEY, example_id TEXT NOT NULL, revision INTEGER NOT NULL,
 created REAL NOT NULL, delivered REAL, UNIQUE(example_id,revision))""",
    """CREATE TABLE IF NOT EXISTS sessions (hash TEXT PRIMARY KEY, csrf TEXT NOT NULL, expires REAL NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS tokens (hash TEXT PRIMARY KEY, asset_id TEXT NOT NULL REFERENCES assets(id),
 purpose TEXT NOT NULL, expires REAL NOT NULL)""",
    "CREATE TABLE IF NOT EXISTS audit (id TEXT PRIMARY KEY, action TEXT NOT NULL, object_id TEXT NOT NULL, created REAL NOT NULL)",
    "CREATE TABLE IF NOT EXISTS worker_status (id INTEGER PRIMARY KEY, heartbeat REAL NOT NULL, pid INTEGER NOT NULL)",
    "CREATE INDEX IF NOT EXISTS jobs_pending ON jobs(status,available,created)",
    "CREATE INDEX IF NOT EXISTS assets_owner ON assets(tenant_id,created)",
    "CREATE INDEX IF NOT EXISTS events_job ON events(job_id,id)",
]


class Database:
    def __init__(self, path, migrate=True):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{path}", connect_args={"timeout": 15})

        @event.listens_for(self.engine, "connect")
        def pragmas(db, _):
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA busy_timeout=15000")

        if migrate:
            with self.tx() as c:
                from alembic import command
                from alembic.config import Config

                from .config import ROOT

                config = Config(str(ROOT / "alembic.ini"))
                config.set_main_option("script_location", str(ROOT / "migrations"))
                config.attributes["connection"] = c
                command.upgrade(config, "head")
                c.execute(text("INSERT OR IGNORE INTO schema_version VALUES (1)"))
                c.execute(text("INSERT OR IGNORE INTO tenants(id) VALUES (:id)"), {"id": TENANT})

    @contextmanager
    def tx(self):
        with self.engine.connect() as c:
            c.exec_driver_sql("BEGIN IMMEDIATE")
            try:
                yield c
                c.commit()
            except BaseException:
                c.rollback()
                raise

    def one(self, sql, **args):
        with self.engine.connect() as c:
            return unpack(c.execute(text(sql), args).first())

    def all(self, sql, **args):
        with self.engine.connect() as c:
            return [unpack(r) for r in c.execute(text(sql), args)]

    def asset(self, id, tenant=TENANT):
        row = self.one(
            "SELECT * FROM assets WHERE id=:id AND tenant_id=:t AND deleted IS NULL AND expires>:n",
            id=str(id),
            t=tenant,
            n=now(),
        )
        if not row:
            raise Problem("NOT_FOUND", "Asset not found", 404)
        return row

    def job(self, id, tenant=TENANT):
        row = self.one("SELECT * FROM jobs WHERE id=:id AND tenant_id=:t", id=str(id), t=tenant)
        if not row:
            raise Problem("NOT_FOUND", "Job not found", 404)
        if row["asset_id"]:
            self.asset(row["asset_id"], tenant)
        return row

    def event(self, c, job, status, **extra):
        stages = ["analyzing", "planning", "processing", "evaluating", "exporting"]
        extra.setdefault("total_stages", len(stages))
        extra.setdefault(
            "completed_stages",
            len(stages) if status == "succeeded" else stages.index(status) if status in stages else 0,
        )
        c.execute(
            text("INSERT INTO events(tenant_id,job_id,payload,created) VALUES (:t,:j,:p,:n)"),
            {"t": TENANT, "j": job, "p": dumps({"job_id": job, "status": status, **extra}), "n": now()},
        )

    def enqueue(self, kind, request, asset_id=None, key=None, limit=3, deadline=1800):
        payload = dumps(request)
        with self.tx() as c:
            if key:
                old = unpack(
                    c.execute(
                        text("SELECT * FROM idempotency WHERE tenant_id=:t AND endpoint=:e AND key=:k AND expires>:n"),
                        {"t": TENANT, "e": kind, "k": key, "n": now()},
                    ).first()
                )
                if old:
                    if old["hash"] != digest(request):
                        raise Problem("IDEMPOTENCY_CONFLICT", "Key already used for another request", 409)
                    return old["result"]
            count = c.execute(text("SELECT COUNT(*) FROM jobs WHERE status='queued'")).scalar_one()
            if count >= limit:
                raise Problem("QUOTA_EXCEEDED", "Waiting queue is full", 429)
            if asset_id:
                asset = c.execute(
                    text("SELECT id FROM assets WHERE id=:id AND deleted IS NULL AND expires>:n"),
                    {"id": asset_id, "n": now()},
                ).first()
                if not asset:
                    raise Problem("NOT_FOUND", "Asset unavailable", 404)
            id = uid()
            c.execute(
                text(
                    "INSERT INTO jobs(id,tenant_id,asset_id,kind,request,status,created,deadline) VALUES (:id,:t,:a,:k,:r,'queued',:n,:d)"
                ),
                {"id": id, "t": TENANT, "a": asset_id, "k": kind, "r": payload, "n": now(), "d": now() + deadline},
            )
            if key:
                c.execute(
                    text("INSERT OR REPLACE INTO idempotency VALUES (:t,:e,:k,:h,:r,:x)"),
                    {"t": TENANT, "e": kind, "k": key, "h": digest(request), "r": id, "x": now() + 86400},
                )
            self.event(c, id, "queued", stage="queued")
            return id

    def claim(self, owner, lease=60):
        with self.tx() as c:
            # Crashed attempts cannot publish after their fencing token is replaced.
            c.execute(
                text(
                    "UPDATE jobs SET status='queued',owner=NULL,lease=NULL WHERE lease<:n AND status NOT IN ('succeeded','failed','canceled','queued') AND canceled=0"
                ),
                {"n": now()},
            )
            c.execute(
                text(
                    "UPDATE jobs SET status='canceled',lease=NULL WHERE canceled=1 AND status NOT IN ('succeeded','failed','canceled') AND (lease IS NULL OR lease<:n)"
                ),
                {"n": now()},
            )
            c.execute(
                text(
                    "UPDATE jobs SET status='failed',error=:e,lease=NULL WHERE deadline<:n AND status NOT IN ('succeeded','failed','canceled')"
                ),
                {"n": now(), "e": dumps({"code": "DEADLINE_EXCEEDED", "detail": "Job deadline expired"})},
            )
            if c.execute(
                text("SELECT id FROM jobs WHERE lease>:n AND status NOT IN ('succeeded','failed','canceled')"),
                {"n": now()},
            ).first():
                return None
            row = unpack(
                c.execute(
                    text(
                        "SELECT * FROM jobs WHERE status='queued' AND canceled=0 AND available<=:n ORDER BY created LIMIT 1"
                    ),
                    {"n": now()},
                ).first()
            )
            if not row:
                return None
            row.update(token=row["token"] + 1, attempt=row["attempt"] + 1, owner=owner)
            c.execute(
                text(
                    "UPDATE jobs SET status='analyzing',token=:token,attempt=:attempt,owner=:owner,lease=:lease WHERE id=:id"
                ),
                {**row, "lease": now() + lease},
            )
            self.event(c, row["id"], "analyzing", stage="analyzing")
            return row

    def guard(self, c, job):
        row = unpack(c.execute(text("SELECT * FROM jobs WHERE id=:id"), {"id": job["id"]}).first())
        if (
            not row
            or row["token"] != job["token"]
            or row["owner"] != job["owner"]
            or row["status"] in ("succeeded", "failed", "canceled")
            or row["canceled"]
            or not row["lease"]
            or row["lease"] < now()
        ):
            raise Problem("LEASE_LOST", "Job canceled or lease lost", 409)
        if row["deadline"] < now():
            raise Problem("DEADLINE_EXCEEDED", "Job deadline expired", 409)
        if row["asset_id"]:
            request = json.loads(row["request"])
            dependencies = [
                request.get("segmentation", {}).get("refinement_mask_asset_id"),
                (request.get("inpaint") or {}).get("erase_mask_asset_id"),
            ]
            for dependency in filter(None, dependencies):
                if not c.execute(
                    text("SELECT id FROM assets WHERE id=:id AND tenant_id=:t AND deleted IS NULL AND expires>:n"),
                    {"id": dependency, "t": TENANT, "n": now()},
                ).first():
                    raise Problem("LEASE_LOST", "A required mask was deleted", 409)
            asset = c.execute(
                text("SELECT id FROM assets WHERE id=:id AND deleted IS NULL AND expires>:n"),
                {"id": row["asset_id"], "n": now()},
            ).first()
            if not asset:
                raise Problem("LEASE_LOST", "Input deleted", 409)
        return row

    def stage(self, job, stage):
        with self.tx() as c:
            self.guard(c, job)
            c.execute(
                text("UPDATE attempts SET ended=:n WHERE job_id=:j AND token=:t AND ended IS NULL"),
                {"n": now(), "j": job["id"], "t": job["token"]},
            )
            c.execute(text("UPDATE jobs SET stage=:s,status=:s WHERE id=:id"), {"s": stage, "id": job["id"]})
            c.execute(
                text("INSERT OR IGNORE INTO attempts VALUES (:id,:j,:t,:s,:n,NULL,NULL)"),
                {"id": uid(), "j": job["id"], "t": job["token"], "s": stage, "n": now()},
            )
            self.event(c, job["id"], stage, stage=stage)
        job["stage"] = stage

    def heartbeat(self, job, lease=60):
        with self.tx() as c:
            self.guard(c, job)
            import os

            c.execute(text("INSERT OR REPLACE INTO worker_status VALUES (1,:n,:p)"), {"n": now(), "p": os.getpid()})
            c.execute(text("UPDATE jobs SET lease=:l WHERE id=:id"), {"l": now() + lease, "id": job["id"]})
