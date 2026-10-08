import asyncio
import hashlib
import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from fastapi import FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from .config import ROOT, TENANT, Settings
from .db import Database, dumps, now, uid
from .domain import Feedback, JobRequest, Problem, SearchRequest, Upload
from .lifecycle import consent, delete_asset, replay
from .memory import feedback
from .models import Registry
from .storage import Storage


def hash_token(value):
    return hashlib.sha256(value.encode()).hexdigest()


def create_app(settings=None):
    settings = settings or Settings.env()
    storage = Storage(settings.data)
    db = Database(settings.data / "library.sqlite3")
    replay(db, storage)
    app = FastAPI(title="Image Cleaning", version="1.0.0")
    app.state.db = db
    app.state.settings = settings
    app.state.storage = storage
    secret_path = settings.data / "bootstrap.secret"
    if not secret_path.exists():
        fd = os.open(secret_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_urlsafe(32))
    secret = secret_path.read_text().strip()
    origin = settings.origin
    host = urlparse(origin).netloc

    def problem(code, detail, status, request_id=""):
        return JSONResponse(
            {
                "type": f"urn:image-cleaning:{code}",
                "title": code,
                "status": status,
                "detail": detail,
                "code": code,
                "request_id": request_id,
            },
            status_code=status,
            media_type="application/problem+json",
            headers={"Retry-After": "5"} if status == 429 else {},
        )

    @app.exception_handler(Problem)
    async def handle(request, e):
        return problem(e.code, e.detail, e.status, getattr(request.state, "request_id", ""))

    @app.exception_handler(RequestValidationError)
    async def validation(request, e):
        return problem(
            "VALIDATION_ERROR", "; ".join(x["msg"] for x in e.errors()), 422, getattr(request.state, "request_id", "")
        )

    @app.middleware("http")
    async def secure(request, call_next):
        request.state.request_id = str(uuid4())
        if request.headers.get("host") != host:
            return problem("INVALID_HOST", "Use the configured loopback address", 403)
        supplied_origin = request.headers.get("origin")
        if supplied_origin and supplied_origin != origin:
            return problem("INVALID_ORIGIN", "Origin is not allowed", 403)
        path = request.url.path
        public = path in ("/health/live", "/health/ready", "/v1/session/bootstrap") or not path.startswith("/v1/")
        transport = path.startswith("/v1/transport/")
        if not public and not transport:
            session = db.one(
                "SELECT * FROM sessions WHERE hash=:h AND expires>:n",
                h=hash_token(request.cookies.get("session", "")),
                n=now(),
            )
            if not session:
                return problem("UNAUTHENTICATED", "Start a local session", 401)
            if request.method not in ("GET", "HEAD", "OPTIONS"):
                if supplied_origin != origin or not secrets.compare_digest(
                    request.headers.get("x-csrf-token", ""), session["csrf"]
                ):
                    return problem("CSRF_FAILED", "Invalid request origin or CSRF token", 403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'"
        )
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    def transport_token(asset, purpose):
        token = secrets.token_urlsafe(32)
        with db.tx() as c:
            c.execute(
                text("INSERT INTO tokens VALUES (:h,:a,:p,:e)"),
                {"h": hash_token(token), "a": asset, "p": purpose, "e": now() + 300},
            )
        return f"/v1/transport/{asset}/{purpose}?token={token}"

    def authorize_token(id, purpose, token):
        row = db.one(
            "SELECT * FROM tokens WHERE hash=:h AND asset_id=:a AND purpose=:p AND expires>:n",
            h=hash_token(token),
            a=id,
            p=purpose,
            n=now(),
        )
        if not row:
            raise Problem("NOT_FOUND", "Transport token expired or invalid", 404)
        return db.asset(id)

    def asset_public(row):
        return {
            k: row[k]
            for k in (
                "id",
                "role",
                "state",
                "filename",
                "mime",
                "bytes",
                "width",
                "height",
                "created",
                "expires",
                "error",
                "parent_id",
            )
        }

    def job_public(row):
        result = json.loads(row["result"]) if row["result"] else None
        if row["kind"] == "search" and result:
            live_results = []
            for item in result.get("results", []):
                try:
                    db.asset(item["asset_id"])
                except Problem:
                    continue
                live_results.append(item)
            result["results"] = live_results
        return {
            "job_id": row["id"],
            "asset_id": row["asset_id"],
            "kind": row["kind"],
            "status": row["status"],
            "stage": row["stage"],
            "created_at": datetime.fromtimestamp(row["created"], timezone.utc).isoformat(),
            "deadline": datetime.fromtimestamp(row["deadline"], timezone.utc).isoformat(),
            "warnings": json.loads(row["warnings"]),
            "error": json.loads(row["error"]) if row["error"] else None,
            "result": result,
            "status_url": f"/v1/jobs/{row['id']}",
            "events_url": f"/v1/jobs/{row['id']}/events",
        }

    @app.get("/health/live")
    def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    def ready():
        heartbeat = db.one("SELECT heartbeat FROM worker_status WHERE id=1")
        ok = bool(heartbeat and now() - heartbeat["heartbeat"] < 90 and os.access(settings.data, os.W_OK))
        return JSONResponse(
            {"ready": ok, "worker": bool(heartbeat and now() - heartbeat["heartbeat"] < 90)},
            status_code=200 if ok else 503,
        )

    @app.post("/v1/session/bootstrap")
    async def bootstrap(request: Request):
        if request.headers.get("origin") != origin:
            raise Problem("INVALID_ORIGIN", "Use the local application", 403)
        body = await request.json()
        if not isinstance(body.get("secret"), str) or not secrets.compare_digest(body["secret"], secret):
            await asyncio.sleep(0.3)
            raise Problem("UNAUTHENTICATED", "Invalid bootstrap secret", 401)
        token = secrets.token_urlsafe(32)
        csrf = secrets.token_urlsafe(24)
        with db.tx() as c:
            c.execute(
                text("INSERT INTO sessions VALUES (:h,:c,:e)"),
                {"h": hash_token(token), "c": csrf, "e": now() + 8 * 3600},
            )
        response = JSONResponse({"csrf": csrf})
        response.set_cookie("session", token, httponly=True, samesite="strict", max_age=8 * 3600)
        return response

    @app.get("/v1/session")
    def session(request: Request):
        row = db.one("SELECT csrf FROM sessions WHERE hash=:h", h=hash_token(request.cookies["session"]))
        return {
            "csrf": row["csrf"],
            "consent": bool(db.one("SELECT consent FROM tenants WHERE id=:t", t=TENANT)["consent"]),
        }

    @app.put("/v1/consent")
    async def set_consent(request: Request):
        body = await request.json()
        if type(body.get("enabled")) is not bool:
            raise Problem("VALIDATION_ERROR", "enabled must be boolean")
        consent(db, body["enabled"])
        return {"enabled": body["enabled"]}

    @app.get("/v1/operations")
    def operations():
        return {
            "profile": "local_m4_16gb",
            "models": Registry(settings.data).capabilities(),
            "operations": [
                "segment",
                "denoise_classical",
                "tone",
                "inpaint_classical",
                "upscale_lanczos",
                "compose",
                "encode",
            ],
            "disabled": ["deblur_restormer", "denoise_restormer", "upscale_realesrgan", "inpaint_lama"],
            "limits": {
                "input_bytes": settings.max_bytes,
                "input_pixels": settings.max_pixels,
                "max_axis": settings.max_axis,
                "output_pixels": settings.max_output_pixels,
                "scales": [1, 2],
                "waiting_jobs": 3,
            },
            "retrieval": "baseline until calibrated",
            "retention_days": settings.retention_days,
        }

    @app.post("/v1/uploads", status_code=201)
    def reserve(body: Upload):
        if body.parent_asset_id:
            db.asset(body.parent_asset_id)
        id = uid()
        with db.tx() as c:
            c.execute(
                text("""INSERT INTO assets(id,tenant_id,role,state,filename,mime,bytes,sha256,parent_id,created,expires)
 VALUES (:id,:t,:r,'reserved',:f,:m,:b,:s,:p,:n,:e)"""),
                {
                    "id": id,
                    "t": TENANT,
                    "r": body.role,
                    "f": Path(body.filename.replace("\\", "/")).name,
                    "m": body.mime,
                    "b": body.bytes,
                    "s": body.sha256,
                    "p": str(body.parent_asset_id) if body.parent_asset_id else None,
                    "n": now(),
                    "e": now() + 86400,
                },
            )
        return {
            "asset_id": id,
            "upload_url": transport_token(id, "upload"),
            "required_headers": {"Content-Type": body.mime},
            "expires_in": 300,
        }

    @app.put("/v1/transport/{id}/upload")
    async def put(id: str, request: Request, token: str):
        asset = authorize_token(id, "upload", token)
        if asset["state"] != "reserved":
            raise Problem("UPLOAD_COMPLETE", "Upload is already complete", 409)
        data = bytearray()
        async for block in request.stream():
            data.extend(block)
            if len(data) > min(settings.max_bytes, asset["bytes"]):
                raise Problem("BYTE_LIMIT", "Upload exceeds reserved byte size", 413)
        if len(data) != asset["bytes"] or hashlib.sha256(data).hexdigest() != asset["sha256"]:
            raise Problem("HASH_MISMATCH", "Upload differs from reservation")
        key = f"quarantine/{id}/original"
        storage.write(key, data)
        with db.tx() as c:
            c.execute(
                text("UPDATE assets SET path=:p WHERE id=:id AND state='reserved' AND deleted IS NULL"),
                {"p": key, "id": id},
            )
        return {"asset_id": id, "uploaded": True}

    @app.post("/v1/uploads/{id}/complete", status_code=202)
    def complete(id: str):
        asset = db.asset(id)
        if asset["state"] in ("validating", "ready"):
            return {"asset_id": id, "state": asset["state"]}
        if not asset["path"]:
            raise Problem("UPLOAD_MISSING", "Upload bytes before completion", 409)
        job = db.enqueue("validate", {"asset_id": id}, id, key="upload:" + id, limit=settings.max_waiting)
        with db.tx() as c:
            c.execute(
                text(
                    "UPDATE assets SET state='validating',expires=:e WHERE id=:id AND deleted IS NULL AND state!='ready'"
                ),
                {"e": now() + settings.retention_days * 86400, "id": id},
            )
        return {"asset_id": id, "state": "validating", "job_id": job}

    @app.get("/v1/assets")
    def assets(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
        rows = db.all(
            "SELECT * FROM assets WHERE tenant_id=:t AND role='original' AND deleted IS NULL AND expires>:n ORDER BY created DESC LIMIT :l OFFSET :o",
            t=TENANT,
            n=now(),
            l=limit,
            o=offset,
        )
        return [asset_public(r) for r in rows]

    @app.get("/v1/assets/{id}")
    def asset(id: str):
        return asset_public(db.asset(id))

    @app.get("/v1/assets/{id}/content")
    def content(id: str, representation: str = "canonical"):
        row = db.asset(id)
        key = (row["canonical_path"] or row["path"]) if representation == "canonical" else row["path"]
        if representation not in ("canonical", "original"):
            raise Problem("VALIDATION_ERROR", "Unknown representation")
        if not key or row["state"] != "ready":
            raise Problem("ASSET_NOT_READY", "Asset is not ready", 409)
        return FileResponse(
            storage.path(key),
            media_type="image/png" if representation == "canonical" and row["canonical_path"] else row["mime"],
        )

    @app.post("/v1/assets/{id}/download")
    def download(id: str):
        db.asset(id)
        return {"url": transport_token(id, "download"), "expires_in": 300}

    @app.get("/v1/transport/{id}/download")
    def get_download(id: str, token: str):
        row = authorize_token(id, "download", token)
        if not row["path"]:
            raise Problem("ASSET_NOT_READY", "Asset not ready", 409)
        return FileResponse(storage.path(row["path"]), media_type=row["mime"], filename=row["filename"])

    @app.delete("/v1/assets/{id}", status_code=202)
    def delete(id: str):
        delete_asset(db, storage, id)
        return {"deleted": id}

    def check_request(body):
        asset = db.asset(body.input_asset_id)
        if asset["state"] != "ready" or asset["role"] != "original":
            raise Problem("ASSET_NOT_READY", "Select a ready original", 409)
        scale = body.upscale.scale
        if (
            asset["width"] * asset["height"] * scale * scale > settings.max_output_pixels
            or max(asset["width"], asset["height"]) * scale > settings.max_axis
        ):
            raise Problem("PIXEL_LIMIT", "Requested output exceeds local limits")
        for id, role in [
            (body.segmentation.refinement_mask_asset_id, "refinement_mask"),
            (body.inpaint.erase_mask_asset_id if body.inpaint else None, "erase_mask"),
        ]:
            if id:
                mask = db.asset(id)
                if (
                    mask["role"] != role
                    or mask["parent_id"] != asset["id"]
                    or mask["state"] != "ready"
                    or (mask["width"], mask["height"]) != (asset["width"], asset["height"])
                ):
                    raise Problem("MASK_DIMENSION_MISMATCH", "Mask must have the correct role, source, and dimensions")
        return asset

    @app.post("/v1/jobs", status_code=202)
    def create_job(body: JobRequest, idempotency_key: str = Header(min_length=1, max_length=200)):
        asset = check_request(body)
        id = db.enqueue(
            "image",
            body.model_dump(mode="json"),
            asset["id"],
            idempotency_key,
            settings.max_waiting,
            settings.deadline_seconds,
        )
        return job_public(db.job(id))

    @app.get("/v1/jobs")
    def jobs(asset_id: str | None = None):
        if asset_id:
            db.asset(asset_id)
        rows = db.all(
            """SELECT j.* FROM jobs j LEFT JOIN assets a ON a.id=j.asset_id WHERE j.tenant_id=:t
 AND (j.asset_id IS NULL OR (a.deleted IS NULL AND a.expires>:n)) AND (:a IS NULL OR j.asset_id=:a) ORDER BY j.created DESC LIMIT 100""",
            t=TENANT,
            n=now(),
            a=asset_id,
        )
        return [job_public(r) for r in rows]

    @app.get("/v1/jobs/{id}")
    def get_job(id: str):
        return job_public(db.job(id))

    @app.post("/v1/jobs/{id}/cancel", status_code=202)
    def cancel(id: str):
        db.job(id)
        with db.tx() as c:
            c.execute(
                text(
                    "UPDATE jobs SET canceled=1,status=CASE WHEN status='queued' THEN 'canceled' ELSE 'cancel_requested' END WHERE id=:id AND status NOT IN ('succeeded','failed','canceled')"
                ),
                {"id": id},
            )
            db.event(c, id, "cancel_requested", stage="cancel_requested")
        return job_public(db.job(id))

    @app.post("/v1/jobs/{id}/retry", status_code=202)
    def retry(id: str):
        job = db.job(id)
        if job["status"] != "failed":
            raise Problem("JOB_NOT_FAILED", "Only failed jobs can be retried", 409)
        if job["kind"] == "image":
            check_request(JobRequest.model_validate_json(job["request"]))
        return job_public(
            db.job(
                db.enqueue(job["kind"], json.loads(job["request"]), job["asset_id"], deadline=settings.deadline_seconds)
            )
        )

    @app.get("/v1/jobs/{id}/plan")
    def get_plan(id: str):
        row = db.job(id)
        return json.loads(row["plan"]) if row["plan"] else {"state": "pending"}

    @app.get("/v1/jobs/{id}/feedback")
    def get_feedback(id: str):
        db.job(id)
        row = db.one("SELECT accepted,revision,reason FROM feedback WHERE job_id=:id", id=id)
        return row or {"accepted": None, "revision": 0, "reason": ""}

    @app.post("/v1/jobs/{id}/feedback")
    def set_feedback(id: str, body: Feedback):
        feedback(db, db.job(id), body.accepted, body.revision, body.reason)
        return {"saved": True, "revision": body.revision}

    @app.get("/v1/jobs/{id}/events")
    async def events(id: str, request: Request, last_event_id: str = Header(default="0")):
        db.job(id)
        try:
            start = int(last_event_id)
        except ValueError:
            raise Problem("VALIDATION_ERROR", "Invalid event cursor")

        async def stream():
            cursor = start
            while not await request.is_disconnected():
                try:
                    job = db.job(id)
                except Problem:
                    return
                for row in db.all(
                    "SELECT * FROM events WHERE job_id=:j AND id>:i ORDER BY id LIMIT 100", j=id, i=cursor
                ):
                    cursor = row["id"]
                    payload = json.loads(row["payload"])
                    payload["event_id"] = cursor
                    yield f"id: {cursor}\ndata: {dumps(payload)}\n\n"
                if job["status"] in ("succeeded", "failed", "canceled"):
                    return
                yield ": heartbeat\n\n"
                await asyncio.sleep(1)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/v1/searches", status_code=202)
    def search(body: SearchRequest):
        return job_public(db.job(db.enqueue("search", body.model_dump())))

    @app.get("/v1/searches/{id}")
    def search_result(id: str):
        row = db.job(id)
        if row["kind"] != "search":
            raise Problem("NOT_FOUND", "Search not found", 404)
        return job_public(row)

    @app.post("/v1/assets/{id}/index", status_code=202)
    def index(id: str):
        a = db.asset(id)
        if a["state"] != "ready" or a["role"] != "original":
            raise Problem("ASSET_NOT_READY", "Original is not ready", 409)
        return job_public(db.job(db.enqueue("index", {}, id)))

    dist = ROOT / "apps/web/dist"
    if dist.exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="web-assets")

        @app.get("/")
        def home():
            return FileResponse(dist / "index.html")
    else:

        @app.get("/")
        def home():
            return {"message": "Build the UI with npm run build in apps/web"}

    return app
