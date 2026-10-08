import json
import math

import httpx
from sqlalchemy import text

from .config import TENANT
from .db import digest, now, uid
from .workflows import WorkflowRegistry

COLLECTION = "workflow_examples_siglip_v1"


class MemoryIndex:
    def __init__(self, db, url):
        self.db, self.url = db, url.rstrip("/")

    def call(self, method, path, body=None):
        with httpx.Client(timeout=1.0, trust_env=False) as client:
            r = client.request(method, self.url + path, json=body)
            r.raise_for_status()
            return r.json()

    @staticmethod
    def collection(manifest):
        return COLLECTION + "_" + digest(manifest)[:12]

    def ensure(self, manifest):
        collection = self.collection(manifest)
        try:
            self.call("GET", f"/collections/{collection}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404:
                raise
            self.call("PUT", f"/collections/{collection}", {"vectors": {"image": {"size": 768, "distance": "Cosine"}}})
            for name, schema in [
                ("tenant_id", "keyword"),
                ("approved", "bool"),
                ("manifest", "keyword"),
                ("task", "keyword"),
                ("expires", "float"),
            ]:
                self.call("PUT", f"/collections/{collection}/index", {"field_name": name, "field_schema": schema})

    def live_example(self, id):
        return self.db.one(
            """SELECT e.*,a.expires,b.vector,n.features FROM examples e JOIN assets a ON a.id=e.asset_id
 JOIN tenants t ON t.id=e.tenant_id JOIN embeddings b ON b.asset_id=e.asset_id AND b.manifest=e.manifest
 JOIN analyses n ON n.asset_id=e.asset_id WHERE e.id=:id AND e.tenant_id=:t AND e.approved=1
 AND e.deleted IS NULL AND a.deleted IS NULL AND a.expires>:n AND t.consent=1 AND e.epoch=t.epoch""",
            id=id,
            t=TENANT,
            n=now(),
        )

    def retrieve(self, embedding, features, task):
        collection = self.collection(embedding["manifest"])
        try:
            hits = self.call(
                "POST",
                f"/collections/{collection}/points/query",
                {
                    "query": embedding["vector"],
                    "using": "image",
                    "limit": 50,
                    "with_payload": True,
                    "filter": {
                        "must": [
                            {"key": "tenant_id", "match": {"value": TENANT}},
                            {"key": "approved", "match": {"value": True}},
                            {"key": "manifest", "match": {"value": embedding["manifest"]}},
                            {"key": "task", "match": {"value": task}},
                            {"key": "expires", "range": {"gt": now()}},
                        ]
                    },
                },
            )["result"]["points"]
            evidence = []
            for hit in hits:
                live = self.live_example(str(hit["id"]))
                if not live or live["manifest"] != embedding["manifest"]:
                    continue
                if not WorkflowRegistry().get(live["workflow"]):
                    continue
                other = json.loads(live["features"])
                distances = []
                for key, scale in [("blur", 5), ("noise", 20), ("exposure", 1)]:
                    if features.get(key) is not None and other.get(key) is not None:
                        distances.append(abs(features[key] - other[key]) / scale)
                quality = math.exp(-sum(distances) / len(distances)) if distances else 0
                score = 0.5 * ((float(hit["score"]) + 1) / 2) + 0.3 * quality + 0.2
                evidence.append(
                    {
                        "example_id": live["id"],
                        "workflow": live["workflow"],
                        "score": score,
                        "feature_coverage": len(distances) / 3,
                    }
                )
            return sorted(evidence, key=lambda e: e["score"], reverse=True)[:5], []
        except (httpx.HTTPError, KeyError, ValueError):
            return [], ["retrieval_unavailable"]

    def sync(self):
        pending = self.db.all("SELECT * FROM outbox WHERE delivered IS NULL ORDER BY created LIMIT 20")
        if not pending:
            return
        for event in pending:
            example = self.db.one("SELECT manifest FROM examples WHERE id=:id", id=event["example_id"])
            if not example:
                with self.db.tx() as c:
                    c.execute(text("UPDATE outbox SET delivered=:n WHERE id=:id"), {"n": now(), "id": event["id"]})
                continue
            self.ensure(example["manifest"])
            collection = self.collection(example["manifest"])
            live = self.live_example(event["example_id"])
            if live:
                self.call(
                    "PUT",
                    f"/collections/{collection}/points?wait=true",
                    {
                        "points": [
                            {
                                "id": live["id"],
                                "vector": {"image": json.loads(live["vector"])},
                                "payload": {
                                    "tenant_id": TENANT,
                                    "approved": True,
                                    "manifest": live["manifest"],
                                    "task": "background_remove",
                                    "expires": live["expires"],
                                    "workflow": live["workflow"],
                                    "epoch": live["epoch"],
                                    "revision": live["revision"],
                                },
                            }
                        ]
                    },
                )
            else:
                self.call(
                    "POST", f"/collections/{collection}/points/delete?wait=true", {"points": [event["example_id"]]}
                )
            with self.db.tx() as c:
                c.execute(text("UPDATE outbox SET delivered=:n WHERE id=:id"), {"n": now(), "id": event["id"]})

    def rebuild(self):
        # Current-state revalidation on every projection makes stale events harmless.
        rows = self.db.all("SELECT id,revision FROM examples")
        with self.db.tx() as c:
            for row in rows:
                c.execute(
                    text("INSERT OR REPLACE INTO outbox VALUES (:id,:e,:r,:n,NULL)"),
                    {"id": uid(), "e": row["id"], "r": row["revision"], "n": now()},
                )
        while self.db.one("SELECT id FROM outbox WHERE delivered IS NULL LIMIT 1"):
            self.sync()
        return len(rows)


def plan(request, embedding, features, memory):
    registry = WorkflowRegistry()
    recipe = registry.get("baseline_v1")
    if not recipe:
        raise RuntimeError("Approved baseline recipe is missing")
    evidence, warnings = (
        memory.retrieve(embedding, features, "background_remove") if embedding else ([], ["embedding_unavailable"])
    )
    selected_reason = "deterministic baseline; learned thresholds not yet calibrated"
    calibration = registry.calibration()
    if calibration:
        groups = {}
        for item in evidence:
            groups.setdefault(item["workflow"], []).append(item)
        eligible = []
        for workflow, items in groups.items():
            candidate = registry.get(workflow)
            if (
                candidate
                and len(items) >= calibration["minimum_support"]
                and all(x["feature_coverage"] >= 2 / 3 for x in items)
            ):
                score = sum(x["score"] for x in items) / len(items)
                if score >= calibration["minimum_score"]:
                    eligible.append((score, candidate))
        if eligible:
            recipe = max(eligible, key=lambda x: x[0])[1]
            selected_reason = "approved recipe selected using calibrated, authorized evidence"
    operations = []
    restoring = []
    if request["restoration"]["denoise"] != "off":
        restoring.append("denoise_classical")
    if request["restoration"]["tone"] != "off":
        restoring.append("tone")
    if request.get("inpaint"):
        operations.extend(restoring + ["inpaint_classical"])
    if request["segmentation"]["mode"] != "none":
        operations.append("segment")
    if not request.get("inpaint"):
        operations.extend(restoring)
    if request["upscale"]["scale"] > 1:
        operations.append("upscale_lanczos")
    operations += ["refine_mask", "compose", "evaluate", "encode"]
    result = {
        "revision": 1,
        "workflow": recipe["id"],
        "policy": "policy_v1",
        "profile": "local_m4_16gb",
        "operations": operations,
        "request": request,
        "evidence": evidence,
        "reason": selected_reason,
        "parameters": recipe["parameters"],
        "embedding_manifest": embedding["manifest"] if embedding else None,
        "warnings": warnings,
    }
    result["nodes"] = [
        {
            "id": f"step_{i}",
            "operation": operation,
            "depends_on": [f"step_{i - 1}"] if i else [],
            "parameters": recipe["parameters"] if operation == "denoise_classical" else {},
        }
        for i, operation in enumerate(operations)
    ]
    result["hash"] = digest(result)
    return result


def feedback(db, job, accepted, revision, reason):
    if job["status"] != "succeeded" or job["kind"] != "image":
        from .domain import Problem

        raise Problem("JOB_NOT_READY", "Feedback requires a completed image job", 409)
    with db.tx() as c:
        live_asset = c.execute(
            text("SELECT id FROM assets WHERE id=:id AND tenant_id=:t AND deleted IS NULL AND expires>:n"),
            {"id": job["asset_id"], "t": TENANT, "n": now()},
        ).first()
        if not live_asset:
            from .domain import Problem

            raise Problem("NOT_FOUND", "Asset was deleted", 404)
        old = c.execute(text("SELECT * FROM feedback WHERE job_id=:id"), {"id": job["id"]}).mappings().first()
        if old and revision <= old["revision"]:
            if revision == old["revision"] and bool(old["accepted"]) == accepted and old["reason"] == reason:
                return
            from .domain import Problem

            raise Problem("REVISION_CONFLICT", "Feedback revision must increase", 409)
        c.execute(
            text("INSERT OR REPLACE INTO feedback VALUES (:j,:a,:r,:s,:n)"),
            {"j": job["id"], "a": int(accepted), "r": revision, "s": reason, "n": now()},
        )
        tenant = c.execute(text("SELECT * FROM tenants WHERE id=:id"), {"id": TENANT}).mappings().one()
        request = json.loads(job["request"])
        p = json.loads(job["plan"])
        result = json.loads(job["result"])
        active = bool(
            accepted
            and tenant["consent"]
            and request["memory"]["allow_indexing"]
            and p["embedding_manifest"]
            and not result.get("needs_review")
        )
        existing = (
            c.execute(text("SELECT id,revision FROM examples WHERE job_id=:j"), {"j": job["id"]}).mappings().first()
        )
        if existing:
            id = existing["id"]
            rev = existing["revision"] + 1
            c.execute(
                text("UPDATE examples SET approved=:a,revision=:r,epoch=:e,deleted=:d WHERE id=:id"),
                {"a": int(active), "r": rev, "e": tenant["epoch"], "d": None if active else now(), "id": id},
            )
        elif active:
            id = uid()
            rev = 1
            c.execute(
                text("INSERT OR IGNORE INTO examples VALUES (:id,:t,:j,:a,:w,:m,:e,1,1,NULL)"),
                {
                    "id": id,
                    "t": TENANT,
                    "j": job["id"],
                    "a": job["asset_id"],
                    "w": p["workflow"],
                    "m": p["embedding_manifest"],
                    "e": tenant["epoch"],
                },
            )
        else:
            return
        c.execute(
            text("INSERT OR IGNORE INTO outbox VALUES (:id,:e,:r,:n,NULL)"),
            {"id": uid(), "e": id, "r": rev, "n": now()},
        )
