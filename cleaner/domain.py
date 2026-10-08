from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Problem(Exception):
    def __init__(self, code, detail, status=422):
        self.code, self.detail, self.status = code, detail, status
        super().__init__(detail)


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Upload(Strict):
    filename: str = Field(default="image", max_length=200)
    mime: Literal["image/png", "image/jpeg", "image/webp"]
    bytes: int = Field(gt=0, le=25 * 1024 * 1024)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    role: Literal["original", "refinement_mask", "erase_mask"] = "original"
    parent_asset_id: UUID | None = None

    @model_validator(mode="after")
    def parent(self):
        if self.role != "original" and self.parent_asset_id is None:
            raise ValueError("Masks require a parent asset")
        return self


class Point(Strict):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    label: Literal[0, 1]


class Box(Strict):
    x_min: float = Field(ge=0, le=1)
    y_min: float = Field(ge=0, le=1)
    x_max: float = Field(ge=0, le=1)
    y_max: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def ordered(self):
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("Box coordinates must be ordered")
        return self


class Segmentation(Strict):
    mode: Literal["automatic", "interactive", "none"] = "automatic"
    allow_lightweight_fallback: bool = False
    coordinate_space: Literal["canonical_normalized"] = "canonical_normalized"
    points: list[Point] = Field(default_factory=list, max_length=32)
    box: Box | None = None
    refinement_mask_asset_id: UUID | None = None

    @model_validator(mode="after")
    def prompts(self):
        has = bool(self.points or self.box or self.refinement_mask_asset_id)
        if self.mode == "interactive" and not (
            any(p.label == 1 for p in self.points) or self.box or self.refinement_mask_asset_id
        ):
            raise ValueError("Interactive selection requires a positive point, box, or refinement mask")
        if self.mode != "interactive" and has:
            raise ValueError("Prompts require interactive segmentation")
        return self


class Restoration(Strict):
    denoise: Literal["off", "classical"] = "off"
    deblur: Literal["off"] = "off"
    tone: Literal["off", "classical"] = "off"
    strength: float = Field(default=0.2, ge=0, le=0.5)


class Upscale(Strict):
    scale: Literal[1, 2] = 1
    method: Literal["lanczos"] = "lanczos"


class Output(Strict):
    format: Literal["png", "jpeg"] = "png"
    background: str = "transparent"

    @model_validator(mode="after")
    def background_valid(self):
        import re

        if self.background != "transparent" and not re.fullmatch(r"#[0-9a-fA-F]{6}", self.background):
            raise ValueError("Background must be transparent or #RRGGBB")
        if self.format == "jpeg" and self.background == "transparent":
            raise ValueError("JPEG requires an explicit background")
        return self


class Memory(Strict):
    allow_indexing: bool = False


class Inpaint(Strict):
    erase_mask_asset_id: UUID
    method: Literal["classical"] = "classical"
    feather_px: int = Field(default=2, ge=0, le=8)


class JobRequest(Strict):
    input_asset_id: UUID
    segmentation: Segmentation = Field(default_factory=Segmentation)
    restoration: Restoration = Field(default_factory=Restoration)
    upscale: Upscale = Field(default_factory=Upscale)
    output: Output = Field(default_factory=Output)
    memory: Memory = Field(default_factory=Memory)
    inpaint: Inpaint | None = None


class Feedback(Strict):
    accepted: bool
    revision: int = Field(ge=1)
    reason: str = Field(default="", max_length=500)


class SearchRequest(Strict):
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=24, ge=1, le=100)

    @model_validator(mode="after")
    def nonempty(self):
        self.query = self.query.strip()
        if not self.query:
            raise ValueError("Enter a search query")
        return self
