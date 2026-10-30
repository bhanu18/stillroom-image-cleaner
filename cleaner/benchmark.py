"""Isolated CPU/MPS/CUDA measurements. Compatibility reports are not quality approval."""

import fcntl
import json
import multiprocessing as mp
import os
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
    destination.with_suffix(".json").write_text(json.dumps(model.diagnostics()))


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
                **json.loads(output.with_suffix(".json").read_text()),
                "seconds": time.monotonic() - start,
                "peak_rss": peak,
                "minimum_system_available": minimum_available,
                "swap_delta": psutil.swap_memory().used - swap_start,
                "device": device,
            }
        finally:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()


def parity_check(name, baseline, candidate):
    if baseline.shape != candidate.shape or not baseline.size:
        return 0.0, False
    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        return 0.0, False
    if name == "siglip":
        a, b = baseline.ravel(), candidate.ravel()
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        parity = float(np.dot(a, b) / denominator) if denominator else 0.0
        return parity, parity >= 0.999
    a, b = baseline > 0.5, candidate > 0.5
    union = (a | b).sum()
    parity = float((a & b).sum() / union) if union else 1.0
    return parity, parity >= 0.98


def approve(root, name, original, device, report, report_path):
    registry = Registry(root)
    target = registry.root / name / "manifest.json"
    with target.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = registry.manifest(name)
        if any(current[key] != original[key] for key in ("id", "files", "revision")):
            raise Problem("MODEL_INTEGRITY", "Model changed during qualification")
        runtime = report["fixtures"][0][device]["runtime"]
        if any(row[device]["runtime"] != runtime for row in report["fixtures"]):
            raise Problem("BENCHMARK_FAILED", "Runtime changed during qualification")
        current["qualified_devices"] = sorted(set(current.get("qualified_devices", [])) | {device})
        current.setdefault("device_qualifications", {})[device] = {
            "model": current["id"], "runtime": runtime, "report": str(report_path),
        }
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(current, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if temporary and temporary.exists():
                temporary.unlink()


def qualify(root, name, fixtures, device, approve_device=False):
    if device not in ("cpu", "mps", "cuda"):
        raise ValueError("Qualification device must be cpu, mps or cuda")
    if approve_device and device == "cpu":
        raise ValueError("CPU does not require device approval")
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
        record["parity_passed"] = record["finite"] and bool(baseline.size)
        if device != "cpu":
            candidate, accelerator = measurement(root, name, path, row.get("prompts"), device)
            parity, passed = parity_check(name, baseline, candidate)
            record.update({device: accelerator, "parity": parity, "parity_passed": passed})
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
    report["passed"] = all(row["parity_passed"] for row in reports)
    report_path = dest / f"{name}-{device}-{time.time_ns()}.json"
    report_path.write_text(json.dumps(report, indent=2))
    if not report["passed"]:
        raise Problem("BENCHMARK_FAILED", f"Compatibility checks failed; see {report_path}")
    if approve_device:
        approve(root, name, m, device, report, report_path)
    return report
