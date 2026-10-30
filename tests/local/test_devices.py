import json
from types import SimpleNamespace

import numpy as np
import pytest

from cleaner import benchmark, devices
from cleaner.config import Settings
from cleaner.domain import Problem
from cleaner.models import hash_file


def fake_torch(cuda=True, mps=True):
    return SimpleNamespace(
        __version__="test", version=SimpleNamespace(cuda="test-cuda"),
        cuda=SimpleNamespace(is_available=lambda: cuda, get_device_name=lambda _: "Test GPU",
                             get_device_capability=lambda _: (8, 0)),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


def approved(torch, device):
    return {"id": "test-model", "qualified_devices": [device], "device_qualifications": {
        device: {"model": "test-model", "runtime": devices.identity(torch, device)},
    }}


@pytest.mark.parametrize("system,arch,expected", [("Linux", "x86_64", "cuda"), ("Darwin", "arm64", "mps")])
def test_auto_and_explicit_selection(monkeypatch, system, arch, expected):
    monkeypatch.setattr(devices.platform, "system", lambda: system)
    monkeypatch.setattr(devices.platform, "machine", lambda: arch)
    torch = fake_torch()
    manifest = approved(torch, expected)
    assert devices.select(torch, manifest, "auto") == expected
    assert devices.select(torch, manifest, expected) == expected
    assert devices.select(torch, manifest, "cpu") == "cpu"
    manifest["device_qualifications"][expected]["runtime"]["host"] = "different-host"
    assert devices.select(torch, manifest, "auto") == "cpu"
    with pytest.raises(Problem, match="qualify") as error:
        devices.select(torch, manifest, expected)
    assert error.value.code == "DEVICE_UNQUALIFIED"
    assert devices.select(torch, manifest, expected, qualification=True) == expected


def test_unavailable_and_legacy_approval(monkeypatch):
    monkeypatch.setattr(devices.platform, "system", lambda: "Linux")
    manifest = {"id": "test", "qualified_devices": ["cuda"]}
    assert devices.select(fake_torch(), manifest, "auto") == "cpu"
    with pytest.raises(Problem) as error:
        devices.select(fake_torch(cuda=False), manifest, "cuda")
    assert error.value.code == "ACCELERATOR_UNAVAILABLE"
    assert devices.select(fake_torch(cuda=False), manifest, "auto") == "cpu"


def test_config(monkeypatch):
    monkeypatch.setenv("IMAGE_CLEANING_DEVICE", "cuda")
    assert Settings.env().device == "cuda"
    with pytest.raises(ValueError):
        Settings(device="bogus")


@pytest.mark.parametrize("name,a,b,passed", [
    ("siglip", [1., 0.], [1., 0.], True),
    ("siglip", [1., 0.], [0., 1.], False),
    ("siglip", [0., 0.], [0., 0.], False),
    ("siglip", [1.], [1., 0.], False),
    ("siglip", [1., 0.], [float("nan"), 0.], False),
    ("sam2", [0., 1.], [0., 1.], True),
    ("sam2", [0., 1.], [1., 0.], False),
    ("sam2", [0., 0.], [0., 0.], True),
    ("sam2", [float("inf")], [1.], False),
    ("sam2", [], [], False),
])
def test_parity(name, a, b, passed):
    assert benchmark.parity_check(name, np.array(a), np.array(b))[1] == passed


def fixture_model(tmp_path):
    model = tmp_path / "models/siglip"
    model.mkdir(parents=True)
    weights = model / "weights"
    weights.write_bytes(b"fixture")
    manifest = {"id": "siglip_fixture", "revision": "test", "approved": True,
                "files": {"weights": hash_file(weights)}, "qualified_devices": []}
    target = model / "manifest.json"
    target.write_text(json.dumps(manifest))
    fixtures = tmp_path / "fixtures.json"
    fixtures.write_text(json.dumps({"images": [{"path": "unused.png"}]}))
    return target, fixtures


def test_qualification_approvals_independent_and_failure_preserves(monkeypatch, tmp_path):
    target, fixtures = fixture_model(tmp_path)

    def measure(root, name, path, prompts, device):
        return np.array([1., 0.]), {"runtime": {"device": device}, "device": device}

    monkeypatch.setattr(benchmark, "measurement", measure)
    benchmark.qualify(tmp_path, "siglip", fixtures, "cuda")
    assert not json.loads(target.read_text())["qualified_devices"]
    for device in ("cuda", "mps"):
        benchmark.qualify(tmp_path, "siglip", fixtures, device, True)
    manifest = json.loads(target.read_text())
    assert manifest["qualified_devices"] == ["cuda", "mps"]
    assert manifest["id"] == "siglip_fixture"
    original = target.read_bytes()

    def fail(root, name, path, prompts, device):
        return np.array([1., 0.] if device == "cpu" else [0., 1.]), {"runtime": {}, "device": device}

    monkeypatch.setattr(benchmark, "measurement", fail)
    with pytest.raises(Problem, match="Compatibility checks failed"):
        benchmark.qualify(tmp_path, "siglip", fixtures, "cuda", True)
    assert target.read_bytes() == original
    assert not list(target.parent.glob("tmp*"))


def test_corrupted_model_never_approved(monkeypatch, tmp_path):
    target, fixtures = fixture_model(tmp_path)
    original = target.read_bytes()
    (target.parent / "weights").write_bytes(b"corrupted")
    with pytest.raises(Problem) as error:
        benchmark.qualify(tmp_path, "siglip", fixtures, "cuda", True)
    assert error.value.code == "MODEL_INTEGRITY"
    assert target.read_bytes() == original


def test_only_accelerator_oom_is_classified(monkeypatch):
    import sys

    class CudaOOM(RuntimeError):
        pass

    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=CudaOOM)))
    assert devices.accelerator_oom(CudaOOM('allocation failed'), 'cuda')
    assert not devices.accelerator_oom(RuntimeError('CUDA invalid argument'), 'cuda')
    assert devices.accelerator_oom(RuntimeError('MPS backend out of memory'), 'mps')
    assert not devices.accelerator_oom(RuntimeError('out of memory'), 'cpu')
    assert not devices.accelerator_oom(RuntimeError('arbitrary model error'), 'mps')


def test_atomic_approval_failure_preserves_manifest(monkeypatch, tmp_path):
    target, fixtures = fixture_model(tmp_path)
    original = target.read_bytes()
    monkeypatch.setattr(benchmark, 'measurement', lambda *args: (np.array([1., 0.]), {'runtime': {}}))

    def fail_replace(*args):
        raise OSError('simulated write failure')

    monkeypatch.setattr(benchmark.os, 'replace', fail_replace)
    with pytest.raises(OSError):
        benchmark.qualify(tmp_path, 'siglip', fixtures, 'cuda', True)
    assert target.read_bytes() == original
    assert not list(target.parent.glob('tmp*'))
