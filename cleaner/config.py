import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TENANT = "00000000-0000-4000-8000-000000000001"


@dataclass
class Settings:
    data: Path = Path.home() / "Library/Application Support/ImageCleaning"
    origin: str = "http://127.0.0.1:8000"
    max_bytes: int = 25 * 1024 * 1024
    max_pixels: int = 12_000_000
    max_axis: int = 6000
    max_output_pixels: int = 16_000_000
    deadline_seconds: int = 1800
    max_waiting: int = 3
    lease_seconds: int = 60
    qdrant_url: str = "http://127.0.0.1:6333"
    retention_days: int = 30

    @classmethod
    def env(cls):
        return cls(
            data=Path(os.getenv("IMAGE_CLEANING_DATA", str(cls.data))),
            origin=os.getenv("IMAGE_CLEANING_ORIGIN", cls.origin),
            qdrant_url=os.getenv("QDRANT_URL", cls.qdrant_url),
        )
