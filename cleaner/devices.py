"""Accelerator checks. Call torch-dependent functions only in inference children."""

import platform

from .domain import Problem

DEVICES = ("auto", "cpu", "mps", "cuda")


def validate_device(device):
    if device not in DEVICES:
        raise ValueError(f"IMAGE_CLEANING_DEVICE must be one of {', '.join(DEVICES)}")
    return device


def available(torch, device):
    if device == "cuda":
        return platform.system() == "Linux" and torch.cuda.is_available()
    if device == "mps":
        return platform.system() == "Darwin" and platform.machine() == "arm64" and torch.backends.mps.is_available()
    return device == "cpu"


def identity(torch, device):
    result = {"host": platform.node(), "system": platform.system(), "architecture": platform.machine(),
              "os": platform.platform(), "torch": str(torch.__version__), "device": device}
    if device == "cuda":
        result.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                      capability=list(torch.cuda.get_device_capability(0)))
    return result


def select(torch, manifest, requested, qualification=False):
    validate_device(requested)

    def qualified(device):
        evidence = manifest.get("device_qualifications", {}).get(device, {})
        return (device in manifest.get("qualified_devices", []) and evidence.get("model") == manifest["id"]
                and evidence.get("runtime") == identity(torch, device))

    if requested == "auto":
        for device in ("cuda", "mps"):
            if available(torch, device) and qualified(device):
                return device
        return "cpu"
    if requested == "cpu":
        return "cpu"
    if not available(torch, requested):
        raise Problem("ACCELERATOR_UNAVAILABLE", f"{requested} is unavailable on this host", 503)
    if not qualification and not qualified(requested):
        raise Problem("DEVICE_UNQUALIFIED", f"Run qualify --device {requested} --approve-device for this model/host", 503)
    return requested


def accelerator_oom(error, device):
    # CUDA has a typed OOM; MPS currently exposes a RuntimeError with a specific message.
    if device == "cuda":
        import torch

        return isinstance(error, torch.cuda.OutOfMemoryError)
    return device == "mps" and isinstance(error, RuntimeError) and "mps backend out of memory" in str(error).lower()
