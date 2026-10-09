"""Reconcile checked-in license evidence; never infer a license or approve release.

Run with Python 3.11+ from the repository root. No downloads or installed metadata.
"""
import hashlib
import json
import re
import tomllib
from pathlib import Path


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def generate():
    paths = ["uv.lock", "apps/web/package-lock.json", "licenses/dependency-inventory.json",
             "models/artifacts.lock.json", "cleaner/qdrant_local.py", "infra/compose/qdrant.yaml"]
    observed = json.loads(Path(paths[2]).read_text())["packages"]
    metadata = {(normalize(p["name"]), p["version"]): p for p in observed}
    python = []
    for p in tomllib.loads(Path(paths[0]).read_text())["package"]:
        evidence = metadata.get((normalize(p["name"]), p["version"]))
        python.append({"name": p["name"], "version": p["version"], "source": p["source"],
                       "dependencies": p.get("dependencies", []),
                       "artifacts": {k: p[k] for k in ("sdist", "wheels") if k in p},
                       "installed_metadata": evidence,
                       "redistribution_review": "pending"})
    npm = []
    for path, p in json.loads(Path(paths[1]).read_text())["packages"].items():
        if path:
            npm.append({"path": path, "name": path.split("node_modules/")[-1], **p,
                        "redistribution_review": "pending"})
    models = []
    for name, p in json.loads(Path(paths[3]).read_text()).items():
        evidence = []
        for filename in ("README.md", "LICENSE"):
            path = Path("licenses") / name / filename
            if path.exists():
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                evidence.append({"path": str(path), "sha256": sha,
                                 "matches_artifact_lock": sha == p["files"].get(filename)})
        models.append({"name": name, "manifest": p, "checked_in_evidence": evidence,
                       "redistribution_review": "pending"})
    return {"schema_version": 1, "reviewed": False,
            "scope": "All lock entries, not an installed or shipped SBOM; includes platform and dev variants.",
            "evidence_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
            "python": python, "npm": npm, "models": models,
            "external_runtime": [{"name": "qdrant", "version": "1.13.6",
                "native_source": paths[4], "container_source": paths[5],
                "license_evidence": None, "redistribution_review": "pending"}]}


if __name__ == "__main__":
    Path("licenses/redistribution-inventory.json").write_text(json.dumps(generate(), indent=2) + "\n")
