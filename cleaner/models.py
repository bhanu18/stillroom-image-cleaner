"""Offline adapters. One subprocess call owns one model and releases it on exit."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .domain import Problem
from .images import neutral, normalize


def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Registry:
    def __init__(self, root):
        self.root = Path(root) / "models"

    def manifest(self, name, verify=True):
        path = self.root / name / "manifest.json"
        if not path.is_file():
            raise Problem("MODEL_UNAVAILABLE", f"{name} is not provisioned", 503)
        m = json.loads(path.read_text())
        if not m.get("approved") or not m.get("revision") or not m.get("files"):
            raise Problem("MODEL_UNAVAILABLE", f"{name} awaits artifact approval", 503)
        if verify:
            for key, expected in m["files"].items():
                file = (path.parent / key).resolve()
                if not file.is_relative_to(path.parent.resolve()) or not file.is_file() or hash_file(file) != expected:
                    raise Problem("MODEL_INTEGRITY", f"{name} artifact hash mismatch", 503)
        return m

    def capabilities(self):
        result = {}
        for name in ("siglip", "birefnet", "sam2", "u2net"):
            try:
                m = self.manifest(name, verify=False)
                result[name] = {
                    "available": True,
                    "manifest": m["id"],
                    "qualified_devices": m.get("qualified_devices", []),
                }
            except Problem as e:
                result[name] = {"available": False, "reason": e.detail}
        return result


class Models:
    def __init__(self, root, device="auto", qualification=False):
        self.registry = Registry(root)
        self.device = device
        self.last = {}
        self.qualification = qualification

    def load(self, name):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        m = self.registry.manifest(name)
        device = self.device
        if device == "auto":
            import torch

            device = "mps" if "mps" in m.get("qualified_devices", []) and torch.backends.mps.is_available() else "cpu"
        if device != "cpu" and device not in m.get("qualified_devices", []) and not self.qualification:
            raise Problem("MODEL_UNAVAILABLE", f"{name} has not passed {device} qualification", 503)
        self.last = {"manifest": m["id"], "device": device, "dtype": "float32"}
        return self.registry.root / name, m, device

    def embed(self, image=None, query=None):
        import torch
        from transformers import AutoModel, AutoProcessor

        path, m, device = self.load("siglip")
        model = AutoModel.from_pretrained(path, local_files_only=True).eval().to(device)
        processor = AutoProcessor.from_pretrained(path, local_files_only=True, use_fast=False)
        with torch.inference_mode():
            if image is not None:
                inputs = processor(images=neutral(image), return_tensors="pt").to(device)
                features = model.get_image_features(**inputs)
            else:
                inputs = processor(text=[query], padding="max_length", truncation=True, return_tensors="pt").to(device)
                features = model.get_text_features(**inputs)
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output
        return {"vector": normalize(features[0].float().cpu().numpy()), "manifest": m["id"], "diagnostics": self.last}

    def segment(self, image, mode="automatic", prompts=None, lightweight=False):
        import torch
        from PIL import Image

        name = "sam2" if mode == "interactive" else ("u2net" if lightweight else "birefnet")
        path, m, device = self.load(name)
        with torch.inference_mode():
            if name == "birefnet":
                from torchvision import transforms
                from transformers import AutoModelForImageSegmentation

                model = (
                    AutoModelForImageSegmentation.from_pretrained(path, trust_remote_code=True, local_files_only=True)
                    .eval()
                    .to(device)
                )
                tensor = (
                    transforms.Compose(
                        [
                            transforms.Resize((1024, 1024)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                        ]
                    )(neutral(image))
                    .unsqueeze(0)
                    .to(device)
                )
                mask = model(tensor)[-1].sigmoid()[0, 0].float().cpu().numpy()
            elif name == "sam2":
                from transformers import Sam2Model, Sam2Processor

                model = Sam2Model.from_pretrained(path, local_files_only=True).eval().to(device)
                processor = Sam2Processor.from_pretrained(path, local_files_only=True)
                points = (prompts or {}).get("points", [])
                box = (prompts or {}).get("box")
                kwargs = {}
                if points:
                    kwargs["input_points"] = [
                        [[[p["x"] * (image.width - 1), p["y"] * (image.height - 1)] for p in points]]
                    ]
                    kwargs["input_labels"] = [[[p["label"] for p in points]]]
                if box:
                    kwargs["input_boxes"] = [
                        [
                            [
                                box["x_min"] * (image.width - 1),
                                box["y_min"] * (image.height - 1),
                                box["x_max"] * (image.width - 1),
                                box["y_max"] * (image.height - 1),
                            ]
                        ]
                    ]
                inputs = processor(images=neutral(image), return_tensors="pt", **kwargs).to(device)
                prediction = model(**inputs, multimask_output=True)
                masks = processor.post_process_masks(prediction.pred_masks.cpu(), inputs["original_sizes"])[0][
                    0
                ].numpy()
                scores = prediction.iou_scores[0, 0].float().cpu().numpy()
                eligible = []
                for i, candidate in enumerate(masks):
                    if all(
                        bool(candidate[round(p["y"] * (image.height - 1)), round(p["x"] * (image.width - 1))])
                        == bool(p["label"])
                        for p in points
                    ):
                        eligible.append(i)
                if not eligible:
                    raise Problem("SELECTION_UNSATISFIED", "No mask satisfies the supplied points")
                mask = masks[max(eligible, key=lambda i: scores[i])].astype(np.float32)
            else:
                import importlib.util

                from safetensors.torch import load_file

                spec = importlib.util.spec_from_file_location("approved_u2net", path / "u2net.py")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                model = module.U2NETP(3, 1).eval().to(device)
                model.load_state_dict(load_file(str(path / "model.safetensors")))
                a = np.asarray(neutral(image).resize((320, 320)), dtype=np.float32) / 255
                a = (a - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                tensor = torch.from_numpy(a.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0).to(device)
                mask = model(tensor)[0][0, 0].float().cpu().numpy()
                mask = (mask - mask.min()) / max(float(mask.max() - mask.min()), 1e-8)
        mask = np.asarray(Image.fromarray(mask.astype(np.float32), "F").resize(image.size, Image.Resampling.BILINEAR))
        return {"mask": np.clip(mask, 0, 1), "diagnostics": self.last}
