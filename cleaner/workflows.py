"""Only immutable local registry recipes can influence execution."""

import json
from pathlib import Path

from .config import ROOT

OPERATIONS = {
    "segment",
    "denoise_classical",
    "tone",
    "inpaint_classical",
    "upscale_lanczos",
    "refine_mask",
    "compose",
    "evaluate",
    "encode",
}


class WorkflowRegistry:
    def __init__(self, path=None):
        self.path = Path(path or ROOT / "workflows")

    def get(self, id):
        for file in self.path.glob("*.json"):
            recipe = json.loads(file.read_text())
            if recipe.get("id") != id:
                continue
            if not recipe.get("approved") or recipe.get("profile") != "local_m4_16gb":
                return None
            if not set(recipe.get("operations", [])) <= OPERATIONS:
                return None
            strength = recipe.get("parameters", {}).get("denoise_strength", 0.2)
            if not isinstance(strength, (int, float)) or not 0 <= strength <= 0.5:
                return None
            if id != "baseline_v1":
                evidence = recipe.get("benchmark")
                if not evidence or not evidence.get("held_out_passed") or not evidence.get("report_sha256"):
                    return None
            return recipe
        return None

    def calibration(self):
        path = self.path / "retrieval-calibration.json"
        if not path.exists():
            return None
        value = json.loads(path.read_text())
        if not value.get("held_out_passed") or not value.get("report_sha256"):
            return None
        if not 0 <= value.get("minimum_score", -1) <= 1 or value.get("minimum_support", 0) < 3:
            return None
        return value
