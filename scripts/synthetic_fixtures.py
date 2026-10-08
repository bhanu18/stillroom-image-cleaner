"""Generate original synthetic smoke fixtures, explicitly not a natural-image quality corpus."""

import json
from pathlib import Path

from PIL import Image, ImageDraw

from cleaner.models import hash_file

root = Path("/tmp/stillroom-model-check/fixtures")
root.mkdir(parents=True, exist_ok=True)
im = Image.new("RGB", (256, 256), "#dedfd9")
d = ImageDraw.Draw(im)
d.rounded_rectangle((65, 45, 195, 225), radius=15, fill="#be2838")
d.arc((90, 10, 170, 90), 180, 360, fill="#861c2d", width=12)
im.save(root / "red-bag.png")
(root / "smoke.json").write_text(
    json.dumps(
        {
            "split": "synthetic-smoke",
            "images": [
                {
                    "path": "red-bag.png",
                    "sha256": hash_file(root / "red-bag.png"),
                    "license": "CC0-1.0",
                    "source": "generated geometric fixture",
                    "prompts": {"points": [{"x": 0.5, "y": 0.5, "label": 1}, {"x": 0.05, "y": 0.05, "label": 0}]},
                }
            ],
        },
        indent=2,
    )
)
