"""Explicit network-enabled provisioning; inference never downloads code or weights."""

import json
import re
import shutil
from pathlib import Path

from .db import digest
from .domain import Problem
from .models import Registry, hash_file

SOURCES = {
    "siglip": ("google/siglip-base-patch16-224", "Apache-2.0"),
    "birefnet": ("ZhengPeng7/BiRefNet", "MIT"),
    "sam2": ("facebook/sam2.1-hiera-tiny", "Apache-2.0"),
}


def existing_installation(root, name, source, revision, adapter):
    if not (Path(root) / "models" / name).exists():
        return None
    manifest = Registry(root).manifest(name)
    if (manifest.get("source"), manifest.get("revision"), manifest.get("adapter")) != (source, revision, adapter):
        raise Problem("MODEL_EXISTS", "Installed model differs from the requested version; existing files were preserved")
    if name == "birefnet" and not manifest.get("custom_code_reviewed"):
        raise Problem("CODE_REVIEW_REQUIRED", "Installed BiRefNet manifest lacks custom-code review approval")
    return manifest


def provision(root, name, revision, reviewed=False):
    if not re.fullmatch("[0-9a-f]{40}", revision):
        raise Problem("INVALID_REVISION", "Supply the full immutable upstream commit SHA")
    if name == "birefnet" and not reviewed:
        raise Problem("CODE_REVIEW_REQUIRED", "Review custom Python at this revision and use --approve-reviewed-code")
    source, license = SOURCES[name]
    target = Path(root) / "models" / name
    existing = existing_installation(root, name, source, revision, name + "_v1")
    if existing:
        return existing
    from huggingface_hub import snapshot_download

    patterns = ["*.json", "*.txt", "*.model", "*.safetensors", "README.md", "LICENSE*"]
    if name == "birefnet":
        patterns += ["*.py"]
    if name == "sam2":
        patterns += ["*.yaml"]
    snapshot = snapshot_download(source, revision=revision, allow_patterns=patterns)
    shutil.copytree(snapshot, target, symlinks=False)
    files = {str(p.relative_to(target)): hash_file(p) for p in target.rglob("*") if p.is_file()}
    if not any(k.endswith((".safetensors", ".pt")) for k in files):
        raise Problem("MODEL_UNAVAILABLE", "Snapshot lacks weights")
    manifest = {
        "source": source,
        "revision": revision,
        "files": files,
        "license": license,
        "approved": True,
        "custom_code_reviewed": reviewed,
        "adapter": name + "_v1",
        "qualified_devices": [],
    }
    manifest["id"] = name + "_" + digest(manifest)[:24]
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def provision_u2net(root):
    revision = "ac7e1c817ecab7c7dff5ce6b1abba61cd213ff29"
    existing = existing_installation(root, "u2net", "https://github.com/xuebinqin/U-2-Net", revision, "u2netp_v1")
    if existing:
        return existing

    import tempfile

    import httpx
    import torch
    from safetensors.torch import save_file

    target = Path(root) / "models/u2net"
    target.mkdir(parents=True)
    base = f"https://raw.githubusercontent.com/xuebinqin/U-2-Net/{revision}/"
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for source, dest in [("model/u2net.py", "u2net.py"), ("LICENSE", "LICENSE"), ("README.md", "README.md")]:
            r = client.get(base + source)
            r.raise_for_status()
            (target / dest).write_bytes(r.content)
        url = "https://drive.usercontent.google.com/download?id=1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy&export=download&confirm=t"
        r = client.get(url)
        r.raise_for_status()
        if "html" in r.headers.get("content-type", ""):
            raise Problem("MODEL_UNAVAILABLE", "Official checkpoint returned a download interstitial")
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "u2netp.pth"
        checkpoint.write_bytes(r.content)
        parent_hash = hash_file(checkpoint)
        if parent_hash != "e7567cde013fb64813973ce6e1ecc25a80c05c3ca7adbc5a54f3c3d90991b854":
            raise Problem("MODEL_INTEGRITY", "Official U2Net checkpoint differs from the pinned artifact")
        if hash_file(target / "u2net.py") != "96dd7a19c7de4f13520ccfc1075ded3350ff4946493be3561b9918a46218f415":
            raise Problem("MODEL_INTEGRITY", "U2Net code differs from reviewed source")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        save_file({k: v.contiguous() for k, v in state.items()}, str(target / "model.safetensors"))
    files = {str(p.relative_to(target)): hash_file(p) for p in target.iterdir() if p.is_file()}
    m = {
        "source": "https://github.com/xuebinqin/U-2-Net",
        "revision": revision,
        "files": files,
        "license": "Apache-2.0",
        "checkpoint_source": url,
        "parent_checkpoint_sha256": parent_hash,
        "converter": "torch.weights_only+safetensors_v1",
        "approved": True,
        "adapter": "u2netp_v1",
        "qualified_devices": [],
    }
    m["id"] = "u2net_" + digest(m)[:24]
    (target / "manifest.json").write_text(json.dumps(m, indent=2))
    return m
