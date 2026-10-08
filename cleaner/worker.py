"""Serial durable worker; inference and image decode happen in disposable children."""

import json
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import psutil
from PIL import Image
from sqlalchemy import text

from .config import TENANT, Settings
from .db import Database, digest, dumps, now, uid
from .domain import Problem
from .images import analyze, compose, decode, exports, png, restore
from .memory import MemoryIndex, plan
from .models import Models, Registry
from .storage import Storage

log = logging.getLogger(__name__)


def child_call(conn, root, operation, args):
    try:
        from .sandbox import restrict_child

        restrict_child(operation)
        if operation == "decode":
            image, transform = decode(Path(args["path"]).read_bytes())
            result = {"png": png(image), "transform": transform, "size": image.size}
        else:
            model = Models(root)
            image = Image.open(args["path"]).convert("RGBA") if args.get("path") else None
            if operation == "embed":
                result = model.embed(image=image, query=args.get("query"))
            else:
                result = model.segment(
                    image, mode=args["mode"], prompts=args.get("prompts"), lightweight=args.get("lightweight", False)
                )
        conn.send(("ok", result))
    except Problem as e:
        conn.send(("error", {"code": e.code, "detail": e.detail, "status": e.status}))
    except Exception as e:
        code = "INFERENCE_OOM" if "out of memory" in str(e).lower() else "MODEL_UNAVAILABLE"
        conn.send(("error", {"code": code, "detail": f"{type(e).__name__}: {str(e)[:300]}", "status": 503}))
    finally:
        conn.close()


class Worker:
    def __init__(self, settings=None, runner=None):
        self.settings = settings or Settings.env()
        self.db = Database(self.settings.data / "library.sqlite3")
        self.storage = Storage(self.settings.data)
        from .logging_setup import configure

        configure(self.settings.data)
        self.owner = uid()
        self.runner = runner
        self.memory = MemoryIndex(self.db, self.settings.qdrant_url)

    def run_model(self, job, operation, **args):
        if self.runner:
            return self.runner(operation, **args)
        model_name = (
            "siglip"
            if operation == "embed"
            else "sam2"
            if args.get("mode") == "interactive"
            else "u2net"
            if args.get("lightweight")
            else "birefnet"
        )
        minimum = (
            {"siglip": 2, "sam2": 2, "u2net": 1, "birefnet": 5}.get(model_name, 1) * 1024**3
            if operation != "decode"
            else 512 * 1024**2
        )
        if psutil.virtual_memory().available < minimum:
            raise Problem(
                "RESOURCE_PRESSURE",
                f"{model_name} needs at least {minimum / 1024**3:g} GiB available; close other applications or permit lightweight fallback",
                503,
            )
        ctx = mp.get_context("spawn")
        read, write = ctx.Pipe(duplex=False)
        process = ctx.Process(target=child_call, args=(write, self.settings.data, operation, args))
        process.start()
        write.close()
        started = time.monotonic()
        starting_swap = psutil.swap_memory().used
        peak_rss = 0
        try:
            while not read.poll(0.3):
                try:
                    peak_rss = max(peak_rss, psutil.Process(process.pid).memory_info().rss)
                except psutil.NoSuchProcess:
                    pass
                self.db.heartbeat(job)
                if (
                    psutil.virtual_memory().available < 512 * 1024**2
                    or psutil.swap_memory().used - starting_swap > 512 * 1024**2
                ):
                    raise Problem("RESOURCE_PRESSURE", "Inference stopped to preserve system memory", 503)
                if not process.is_alive():
                    raise Problem("MODEL_UNAVAILABLE", "Inference process exited unexpectedly", 503)
                if time.monotonic() - started > self.settings.deadline_seconds:
                    raise Problem("DEADLINE_EXCEEDED", "Inference timed out")
            status, value = read.recv()
            if status == "error":
                raise Problem(**value)
            self.db.heartbeat(job)
            diagnostics = {
                "seconds": time.monotonic() - started,
                "peak_rss": peak_rss,
                "operation": operation,
                **value.get("diagnostics", {}),
            }
            value["diagnostics"] = diagnostics
            log.info("inference_completed", extra={"job_id": job["id"], **diagnostics})
            with self.db.tx() as c:
                self.db.guard(c, job)
                c.execute(
                    text("UPDATE attempts SET diagnostics=:d WHERE job_id=:j AND token=:t AND stage=:s"),
                    {"d": dumps(diagnostics), "j": job["id"], "t": job["token"], "s": job["stage"]},
                )
            return value
        finally:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
            read.close()

    def publish_asset(self, c, job, role, data, mime, parent, prepared, width=None, height=None):
        id, key, sha = prepared
        self.db.guard(c, job)
        c.execute(
            text("""INSERT INTO assets(id,tenant_id,role,state,filename,mime,bytes,sha256,parent_id,path,width,height,created,expires)
 VALUES (:id,:t,:r,'ready',:f,:m,:b,:s,:p,:path,:w,:h,:n,:x)"""),
            {
                "id": id,
                "t": TENANT,
                "r": role,
                "f": role + (".jpg" if mime == "image/jpeg" else ".json" if mime == "application/json" else ".png"),
                "m": mime,
                "b": len(data),
                "s": sha,
                "p": parent,
                "path": key,
                "w": width,
                "h": height,
                "n": now(),
                "x": min(now() + self.settings.retention_days * 86400, self.db.asset(parent)["expires"]),
            },
        )
        return id

    def finish(self, job, result, warnings=None):
        with self.db.tx() as c:
            self.db.guard(c, job)
            c.execute(
                text("UPDATE attempts SET ended=:n WHERE job_id=:j AND token=:t AND ended IS NULL"),
                {"n": now(), "j": job["id"], "t": job["token"]},
            )
            c.execute(
                text(
                    "UPDATE jobs SET status='succeeded',stage='succeeded',result=:r,warnings=:w,lease=NULL WHERE id=:id"
                ),
                {"r": dumps(result), "w": dumps(warnings or []), "id": job["id"]},
            )
            self.db.event(c, job["id"], "succeeded", stage="succeeded", warnings=warnings or [])

    def validate(self, job):
        asset = self.db.asset(job["asset_id"])
        result = self.run_model(job, "decode", path=str(self.storage.path(asset["path"])))
        w, h = result["size"]
        if result["transform"].get("source_mime") != asset["mime"]:
            raise Problem("INVALID_IMAGE", "Declared MIME type does not match decoded bytes")
        parent = asset["parent_id"]
        if parent:
            source = self.db.asset(parent)
            if (w, h) != (source["width"], source["height"]):
                raise Problem("MASK_DIMENSION_MISMATCH", "Mask must match canonical image dimensions")
        key = f"assets/{asset['id']}/canonical.png"
        self.storage.write(key, result["png"])
        with self.db.tx() as c:
            self.db.guard(c, job)
            c.execute(
                text("UPDATE assets SET state='ready',canonical_path=:p,width=:w,height=:h,transform=:t WHERE id=:id"),
                {"p": key, "w": w, "h": h, "t": dumps(result["transform"]), "id": asset["id"]},
            )
        self.finish(job, {"asset_id": asset["id"]})

    def embed_asset(self, job, asset):
        result = self.run_model(job, "embed", path=str(self.storage.path(asset["canonical_path"])))
        with self.db.tx() as c:
            self.db.guard(c, job)
            c.execute(
                text("INSERT OR REPLACE INTO embeddings VALUES (:a,:t,:m,:v)"),
                {"a": asset["id"], "t": TENANT, "m": result["manifest"], "v": dumps(result["vector"])},
            )
        return result

    def image(self, job):
        request = json.loads(job["request"])
        asset = self.db.asset(job["asset_id"])
        source = Image.open(self.storage.path(asset["canonical_path"])).convert("RGBA")
        features = analyze(source)
        warnings = []
        with self.db.tx() as c:
            self.db.guard(c, job)
            c.execute(
                text("INSERT OR REPLACE INTO analyses VALUES (:a,:v,:f)"),
                {"a": asset["id"], "v": "quality_v1", "f": dumps(features)},
            )
        try:
            embedding = self.embed_asset(job, asset)
        except Problem as e:
            if e.code in ("LEASE_LOST", "DEADLINE_EXCEEDED", "MODEL_INTEGRITY"):
                raise
            embedding = None
            warnings.append("search_index_pending")
        self.db.stage(job, "planning")
        if job.get("plan"):
            resolved = json.loads(job["plan"])
        else:
            resolved = plan(request, embedding, features, self.memory)
            resolved["model_manifests"] = Registry(self.settings.data).capabilities()
            resolved.pop("hash", None)
            resolved["hash"] = digest(resolved)
            with self.db.tx() as c:
                self.db.guard(c, job)
                c.execute(
                    text("UPDATE jobs SET plan=:p WHERE id=:id AND plan IS NULL"),
                    {"p": dumps(resolved), "id": job["id"]},
                )
        warnings += resolved["warnings"]
        self.db.stage(job, "processing")
        request = json.loads(dumps(request))
        if request["restoration"]["denoise"] != "off":
            request["restoration"]["strength"] = resolved.get("parameters", {}).get(
                "denoise_strength", request["restoration"]["strength"]
            )
        erase = None
        if request.get("inpaint"):
            er = self.db.asset(request["inpaint"]["erase_mask_asset_id"])
            erase = Image.open(self.storage.path(er["canonical_path"])).convert("L")
        candidate, changes = restore(source, request, erase) if request.get("inpaint") else (source, {})
        segmentation = request["segmentation"]
        input_image = candidate if request.get("inpaint") else source
        temp = f"jobs/{job['id']}/{job['token']}/model-input.png"
        self.storage.write(temp, png(input_image))
        model_info = {}
        refinement = segmentation.get("refinement_mask_asset_id")
        if segmentation["mode"] == "none":
            mask = np.ones((source.height, source.width), np.float32)
        elif refinement and not segmentation["points"] and not segmentation["box"]:
            ref = self.db.asset(refinement)
            mask = np.asarray(Image.open(self.storage.path(ref["canonical_path"])).convert("L"), dtype=np.float32) / 255
        else:
            try:
                result = self.run_model(
                    job, "segment", path=str(self.storage.path(temp)), mode=segmentation["mode"], prompts=segmentation
                )
            except Problem as e:
                if (
                    e.code not in ("MODEL_UNAVAILABLE", "INFERENCE_OOM", "RESOURCE_PRESSURE")
                    or segmentation["mode"] != "automatic"
                    or not segmentation["allow_lightweight_fallback"]
                ):
                    raise
                result = self.run_model(
                    job, "segment", path=str(self.storage.path(temp)), mode="automatic", lightweight=True
                )
                warnings.append("lightweight_fallback")
            mask = result["mask"]
            model_info = result["diagnostics"]
            if refinement:
                ref = self.db.asset(refinement)
                mask *= (
                    np.asarray(Image.open(self.storage.path(ref["canonical_path"])).convert("L"), dtype=np.float32)
                    / 255
                )
        if not request.get("inpaint"):
            candidate, changes = restore(source, request)
        output = compose(candidate, mask, request["upscale"]["scale"])
        conservative = compose(source, mask, 1)
        self.db.stage(job, "evaluating")
        needs_review = float((mask > 0.5).mean()) < 0.005 or float((mask > 0.5).mean()) > 0.995
        if needs_review:
            warnings.append("mask_review_recommended")
        if segmentation["mode"] == "interactive":
            for p in segmentation["points"]:
                if bool(mask[round(p["y"] * (source.height - 1)), round(p["x"] * (source.width - 1))] > 0.5) != bool(
                    p["label"]
                ):
                    raise Problem("SELECTION_UNSATISFIED", "Refinement contradicts a selected point")
        self.db.stage(job, "exporting")
        files = exports(output, request["output"])
        files["conservative"] = (png(conservative), "image/png")
        provenance = {
            "plan": resolved,
            "models": model_info,
            "transform": json.loads(asset["transform"]),
            "evaluation": {"needs_review": needs_review, **changes},
            "warnings": warnings,
        }
        files["provenance"] = (dumps(provenance).encode(), "application/json")
        # Write immutable files before a short metadata publication transaction.
        prepared = {}
        for role, (data, mime) in files.items():
            id = uid()
            key = f"jobs/{job['id']}/{job['token']}/{id}"
            prepared[role] = (id, key, self.storage.write(key, data))
        self.db.heartbeat(job)
        ids = {}
        with self.db.tx() as c:
            self.db.guard(c, job)
            for role, (data, mime) in files.items():
                if mime.startswith("image/"):
                    import io

                    with Image.open(io.BytesIO(data)) as exported:
                        width, height = exported.size
                else:
                    width = height = None
                ids[role] = self.publish_asset(c, job, role, data, mime, asset["id"], prepared[role], width, height)
            self.db.guard(c, job)
            result = {"artifacts": ids, "needs_review": needs_review, "evaluation": changes}
            c.execute(
                text(
                    "UPDATE jobs SET status='succeeded',stage='succeeded',result=:r,warnings=:w,lease=NULL WHERE id=:id"
                ),
                {"r": dumps(result), "w": dumps(warnings), "id": job["id"]},
            )
            self.db.event(c, job["id"], "succeeded", stage="succeeded", warnings=warnings)

    def search(self, job):
        req = json.loads(job["request"])
        embedding = self.run_model(job, "embed", query=req["query"])
        rows = self.db.all(
            """SELECT e.asset_id,e.vector FROM embeddings e JOIN assets a ON a.id=e.asset_id
 WHERE e.tenant_id=:t AND e.manifest=:m AND a.deleted IS NULL AND a.expires>:n AND a.role='original' """,
            t=TENANT,
            m=embedding["manifest"],
            n=now(),
        )
        vector = np.array(embedding["vector"])
        ranked = sorted(
            [{"asset_id": r["asset_id"], "score": float(np.dot(vector, json.loads(r["vector"])))} for r in rows],
            key=lambda r: r["score"],
            reverse=True,
        )
        total = self.db.one(
            "SELECT COUNT(*) n FROM assets WHERE tenant_id=:t AND role='original' AND state='ready' AND deleted IS NULL AND expires>:n",
            t=TENANT,
            n=now(),
        )["n"]
        self.finish(job, {"results": ranked[: req["limit"]], "indexed": len(rows), "total": total})

    def tick(self):
        with self.db.tx() as c:
            c.execute(text("INSERT OR REPLACE INTO worker_status VALUES (1,:n,:p)"), {"n": now(), "p": os.getpid()})
        job = self.db.claim(self.owner, self.settings.lease_seconds)
        if not job:
            return False
        try:
            self.db.stage(job, "analyzing")
            if job["kind"] == "validate":
                self.validate(job)
            elif job["kind"] == "search":
                self.search(job)
            elif job["kind"] == "index":
                self.finish(job, self.embed_asset(job, self.db.asset(job["asset_id"])))
            else:
                self.image(job)
        except Exception as e:
            problem = (
                e
                if isinstance(e, Problem)
                else Problem("INTERNAL_ERROR", "Processing failed; see local diagnostics", 500)
            )
            if not isinstance(e, Problem):
                log.exception("job_failed job_id=%s", job["id"])
            with self.db.tx() as c:
                current = (
                    c.execute(text("SELECT token,canceled FROM jobs WHERE id=:id"), {"id": job["id"]})
                    .mappings()
                    .first()
                )
                if current and current["token"] == job["token"]:
                    status = "canceled" if current["canceled"] or problem.code == "LEASE_LOST" else "failed"
                    if (
                        not current["canceled"]
                        and isinstance(e, (OSError, TimeoutError))
                        and getattr(e, "errno", None) != 28
                        and job["attempt"] < 3
                    ):
                        status = "queued"
                        c.execute(
                            text("UPDATE jobs SET available=:n WHERE id=:id"),
                            {"n": now() + 2 ** job["attempt"], "id": job["id"]},
                        )
                    c.execute(
                        text("UPDATE jobs SET status=:s,error=:e,lease=NULL WHERE id=:id"),
                        {"s": status, "e": dumps({"code": problem.code, "detail": problem.detail}), "id": job["id"]},
                    )
                    if job["kind"] == "validate" and status == "failed":
                        c.execute(
                            text("UPDATE assets SET state='failed',error=:e WHERE id=:id AND deleted IS NULL"),
                            {"e": problem.detail, "id": job["asset_id"]},
                        )
                    self.db.event(c, job["id"], status, stage=job["stage"])
        return True

    def run(self):
        import fcntl

        lock = (self.settings.data / "worker.lock").open("w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A worker already owns this library")
        from .lifecycle import maintain

        while True:
            busy = self.tick()
            if not busy:
                try:
                    self.memory.sync()
                except Exception:
                    log.debug("Qdrant unavailable")
                maintain(self.db, self.storage)
                time.sleep(0.5)
