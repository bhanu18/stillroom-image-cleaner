"""Fail setup unless the requested accelerator can execute and synchronize a tensor."""

import json
import platform
import sys

import torch

from cleaner.devices import available, identity


def main():
    device = sys.argv[1]
    if platform.python_version_tuple()[:2] != ("3", "12"):
        raise SystemExit("Python 3.12 is required")
    if not available(torch, device):
        raise SystemExit(f"{device} unavailable: verify architecture, PyTorch build and GPU/driver access")
    try:
        a = torch.ones((8, 8), device=device, dtype=torch.float32)
        assert (a @ a).cpu().eq(8).all().item()
        if device == "cuda":
            torch.cuda.synchronize()
        else:
            torch.mps.synchronize()
    except Exception as error:
        raise SystemExit(f"{device} execution failed; check driver/runtime compatibility: {error}") from error
    print(json.dumps(identity(torch, device), indent=2))


if __name__ == "__main__":
    main()
