"""Optional pinned native Qdrant provisioning and launcher ownership."""

import hashlib
import io
import os
import platform
import subprocess
import tarfile

import httpx

URL = "https://github.com/qdrant/qdrant/releases/download/v1.13.6/qdrant-aarch64-apple-darwin.tar.gz"
SHA256 = "e706cea3a2fffe03549f7b9fd202a769ca31dad73f3c57cdf1ae42d538b43f86"


def provision(root):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Native binary supports ARM64 macOS only")
    destination = root / "bin"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "qdrant").exists():
        raise RuntimeError("Native Qdrant already exists; no overwrite performed")
    response = httpx.get(URL, follow_redirects=True, timeout=120)
    response.raise_for_status()
    if hashlib.sha256(response.content).hexdigest() != SHA256:
        raise RuntimeError("Qdrant archive integrity failure")
    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as archive:
        archive.extractall(destination, filter="data")
    return str(destination / "qdrant")


def start_if_provisioned(settings):
    binary = settings.data / "bin/qdrant"
    if not binary.exists() or settings.qdrant_url != "http://127.0.0.1:6333":
        return None
    try:
        response = httpx.get(settings.qdrant_url + "/healthz", timeout=1, trust_env=False)
        if response.is_success:
            return None
    except httpx.HTTPError:
        pass
    env = {
        **os.environ,
        "QDRANT__SERVICE__HOST": "127.0.0.1",
        "QDRANT__STORAGE__STORAGE_PATH": str(settings.data / "qdrant"),
        "QDRANT__TELEMETRY_DISABLED": "true",
        "QDRANT__SERVICE__MAX_WORKERS": "2",
    }
    logdir = settings.data / "logs"
    logdir.mkdir(exist_ok=True)
    with (logdir / "qdrant.log").open("ab") as log:
        return subprocess.Popen(
            [str(binary)], cwd=settings.data, env=env, stdout=log, stderr=log, start_new_session=True
        )
