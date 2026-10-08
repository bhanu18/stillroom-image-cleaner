"""Inventory installed distribution metadata without asserting license approval."""

import importlib.metadata
import json
from pathlib import Path

rows = []
for dist in importlib.metadata.distributions():
    rows.append(
        {
            "name": dist.metadata["Name"],
            "version": dist.version,
            "license_expression": dist.metadata.get("License-Expression"),
            "license_metadata": dist.metadata.get("License"),
            "homepage": dist.metadata.get("Home-page"),
        }
    )
Path("licenses/dependency-inventory.json").write_text(
    json.dumps({"reviewed": False, "packages": sorted(rows, key=lambda r: r["name"].lower())}, indent=2) + "\n"
)
