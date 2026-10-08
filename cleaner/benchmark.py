"""Isolated CPU/MPS measurements. Compatibility reports are not quality approval."""

import json
import multiprocessing as mp
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
import psutil
from PIL import Image

from .domain import Problem
from .models import Models, Registry, hash_file


def infer_child(root, name, path, prompts, device, destination):
    model = Models(root, device=device, qualification=True)
    image = Image.open(path).convert("RGBA")
    if name == "siglip":
        value = np.asarray(model.embed(image=image)["vector"])
    else:
        value = model.segment(
            image, mode="interactive" if name == "sam2" else "automatic", prompts=prompts, lightweight=name == "u2net"
        )["mask"]
    np.save(destination, value)


def measurement(root, name, path, prompts, device):
    with tempfile.TemporaryDirectory(prefix="stillroom-measure-") as temp:
        output = Path(temp) / "result.npy"
        process = mp.get_context("spawn").Process(target=infer_child, args=(root, name, path, prompts, device, output))
        start = time.monotonic()
        swap_start = psutil.swap_memory().used
        peak = 0
        minimum_available = psutil.virtual_memory().available
        process.start()
        try:
            while process.is_alive():
                try:
                    peak = max(peak, psutil.Process(process.pid).memory_info().rss)
                except psutil.NoSuchProcess:
                    pass
                available = psutil.virtual_memory().available
                minimum_available = min(minimum_available, available)
                if available < 512 * 1024**2 or time.monotonic() - start > 1800:
                    process.terminate()
                    raise Problem("RESOURCE_PRESSURE", "Benchmark exceeded safe resource bounds")
                process.join(0.2)
            if process.exitcode != 0 or not output.exists():
                raise Problem("BENCHMARK_FAILED", f"{name} failed on {device}; inspect diagnostic output")
            return np.load(output), {
                "seconds": time.monotonic() - start,
                "peak_rss": peak,
                "minimum_system_available": minimum_available,
                "swap_delta": psutil.swap_memory().used - swap_start,
                "device": device,
            }
        finally:
            if process.is_alive():
                process.terminate()
            process.join()


def qualify(root, name, fixtures, device):
    manifest = json.loads(fixtures.read_text())
    rows = manifest["images"]
    if not rows:
        raise Problem("NO_FIXTURES", "Fixture manifest is empty")
    registry = Registry(root)
    m = registry.manifest(name)
    reports = []
    for row in rows:
        path = (fixtures.parent / row["path"]).resolve()
        if row.get("sha256") and hash_file(path) != row["sha256"]:
            raise Problem("FIXTURE_INTEGRITY", "Fixture hash mismatch")
        baseline, cpu = measurement(root, name, path, row.get("prompts"), "cpu")
        record = {
            "fixture": row["path"],
            "cpu": cpu,
            "finite": bool(np.isfinite(baseline).all()),
            "shape": list(baseline.shape),
        }
        if device == "mps":
            candidate, mps = measurement(root, name, path, row.get("prompts"), "mps")
            if name == "siglip":
                parity = float(np.dot(baseline, candidate) / (np.linalg.norm(baseline) * np.linalg.norm(candidate)))
                passed = parity >= 0.999
            else:
                a, b = baseline > 0.5, candidate > 0.5
                parity = float((a & b).sum() / max(1, (a | b).sum()))
                passed = parity >= 0.98
            record.update(mps=mps, parity=parity, parity_passed=passed)
        reports.append(record)
    report = {
        "model": m["id"],
        "requested_device": device,
        "host": platform.platform(),
        "architecture": platform.machine(),
        "memory": psutil.virtual_memory().total,
        "fixtures": reports,
        "qualification": "compatibility_measurement",
        "release_approved": False,
        "remaining": ["200-image held-out quality review", "sustained model-switching workload"],
    }
    dest = Path(root) / "benchmarks"
    dest.mkdir(exist_ok=True)
    (dest / f"{name}-{int(time.time())}.json").write_text(json.dumps(report, indent=2))
    return report
