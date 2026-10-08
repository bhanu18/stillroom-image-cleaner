"""Bounded decoding, explicit restoration, and straight-alpha exports."""

import io
import warnings

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError

from .domain import Problem


def decode(raw, max_pixels=12_000_000, max_axis=6000):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as im:
                if im.format not in ("PNG", "JPEG", "WEBP") or getattr(im, "n_frames", 1) != 1:
                    raise Problem("INVALID_IMAGE", "Only JPEG, PNG and static WebP are accepted")
                if im.width * im.height > max_pixels or max(im.size) > max_axis:
                    raise Problem("PIXEL_LIMIT", "Image exceeds pixel or axis limits")
                if len(im.info.get("icc_profile", b"")) > 1_000_000 or len(im.info.get("exif", b"")) > 1_000_000:
                    raise Problem("INVALID_IMAGE", "Image metadata is too large")
                original_size = im.size
                orientation = im.getexif().get(274, 1)
                icc = im.info.get("icc_profile")
                image = ImageOps.exif_transpose(im).convert("RGBA")
                if icc:
                    rgb = ImageCms.profileToProfile(
                        image.convert("RGB"),
                        ImageCms.ImageCmsProfile(io.BytesIO(icc)),
                        ImageCms.createProfile("sRGB"),
                        outputMode="RGB",
                    )
                    rgb.putalpha(image.getchannel("A"))
                    image = rgb
                image.info.clear()
                return image, {
                    "original_size": original_size,
                    "canonical_size": image.size,
                    "exif_orientation": orientation,
                    "color_space": "sRGB",
                    "source_mime": Image.MIME[im.format],
                }
    except Problem:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:
        raise Problem("INVALID_IMAGE", "Invalid image or color profile") from exc


def png(im):
    out = io.BytesIO()
    im.save(out, format="PNG")
    return out.getvalue()


def normalize(vector):
    a = np.asarray(vector, dtype=np.float32)
    if a.shape != (768,) or not np.isfinite(a).all() or np.linalg.norm(a) < 1e-8:
        raise Problem("MODEL_OUTPUT_INVALID", "Embedding must contain 768 finite nonzero values")
    return (a / np.linalg.norm(a)).tolist()


def neutral(image):
    bg = Image.new("RGBA", image.size, (128, 128, 128, 255))
    bg.alpha_composite(image)
    return bg.convert("RGB")


def analyze(image):
    thumb = image.copy()
    thumb.thumbnail((512, 512))
    rgba = np.asarray(thumb)
    lum = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
    visible = rgba[:, :, 3] > 0
    values = lum[visible]
    if not len(values):
        return {"version": "quality_v1", "missing": ["blur", "noise", "exposure"], "resolution": list(image.size)}
    lap = cv2.Laplacian(lum, cv2.CV_32F)
    gradients = cv2.magnitude(cv2.Sobel(lum, cv2.CV_32F, 1, 0), cv2.Sobel(lum, cv2.CV_32F, 0, 1))
    flat = visible & (gradients < np.percentile(gradients[visible], 25))
    noise = float(np.median(np.abs(lap[flat])) / 0.6745) if flat.sum() > 100 else None
    return {
        "version": "quality_v1",
        "resolution": list(image.size),
        "analysis_resolution": list(thumb.size),
        "blur": float(np.log1p(lap[visible].var())),
        "noise": noise,
        "missing": ["noise"] if noise is None else [],
        "exposure": float(values.mean() / 255),
        "clipping": float(((values < 2) | (values > 253)).mean()),
        "contrast": float((np.percentile(values, 95) - np.percentile(values, 5)) / 255),
        "tenengrad": float(np.mean(gradients[visible] ** 2)),
        "shadow_fraction": float((values < 2).mean()),
        "highlight_fraction": float((values > 253).mean()),
        "channel_clipping": [float((rgba[:, :, i][visible] >= 253).mean()) for i in range(3)],
        "applicability": {"noise": noise is not None, "blur": "content_dependent", "color_cast": False},
        "roi": "visible_pixels",
    }


def restore(image, request, erase=None):
    rgb = np.asarray(image.convert("RGB")).copy()
    original = rgb.copy()
    r = request["restoration"]
    strength = r["strength"]
    if r["denoise"] == "classical":
        filtered = cv2.bilateralFilter(rgb, 5, 20, 5)
        rgb = np.rint(rgb * (1 - strength) + filtered * strength).astype(np.uint8)
    if r["tone"] == "classical":
        rgb = np.clip(rgb.astype(np.float32) * (1 + strength * 0.25), 0, 255).astype(np.uint8)
    if request.get("inpaint"):
        if erase is None:
            raise Problem("MASK_REQUIRED", "An erase mask is required")
        mask = np.asarray(erase.convert("L"))
        if mask.shape != rgb.shape[:2]:
            raise Problem("MASK_DIMENSION_MISMATCH", "Erase mask dimensions differ")
        mask = (mask > 127).astype(np.uint8) * 255
        if not mask.any():
            raise Problem("INVALID_MASK", "Erase mask is empty")
        before = rgb.copy()
        fill = cv2.inpaint(rgb, mask, 3, cv2.INPAINT_TELEA)
        feather = request["inpaint"]["feather_px"]
        if feather:
            band = cv2.dilate(mask, np.ones((2 * feather + 1, 2 * feather + 1), np.uint8))
            weight = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), max(0.5, feather / 2))
            rgb = np.rint(fill * weight[:, :, None] + before * (1 - weight[:, :, None])).astype(np.uint8)
            rgb[band == 0] = before[band == 0]
        else:
            rgb = np.where(mask[:, :, None] > 0, fill, before)
    output = Image.fromarray(rgb)
    output.putalpha(image.getchannel("A"))
    return output, {"changed_fraction": float(np.any(original != rgb, axis=2).mean())}


def compose(image, mask, scale=1):
    a = np.asarray(image, dtype=np.float32) / 255
    m = np.asarray(mask, dtype=np.float32)
    if m.shape != a.shape[:2] or not np.isfinite(m).all():
        raise Problem("MODEL_OUTPUT_INVALID", "Invalid mask shape or values")
    a[:, :, 3] *= np.clip(m, 0, 1)
    if a[:, :, 3].max() == 0:
        raise Problem("EMPTY_MASK", "No foreground selected")
    if scale != 1:
        # Premultiply while resampling, then safely export straight alpha.
        premult = a[:, :, :3] * a[:, :, 3:4]
        size = (image.width * scale, image.height * scale)
        alpha = np.clip(cv2.resize(a[:, :, 3], size, interpolation=cv2.INTER_LANCZOS4), 0, 1)
        rgb = cv2.resize(premult, size, interpolation=cv2.INTER_LANCZOS4)
        a = np.dstack(
            (np.divide(rgb, alpha[:, :, None], out=np.zeros_like(rgb), where=alpha[:, :, None] > 1e-6), alpha)
        )
    return Image.fromarray(np.rint(np.clip(a, 0, 1) * 255).astype(np.uint8), "RGBA")


def flatten(image, color):
    rgba = np.asarray(image, dtype=np.float32) / 255
    background = np.array([int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)])

    def linear(x):
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

    rgb = linear(rgba[:, :, :3]) * rgba[:, :, 3:4] + linear(background) * (1 - rgba[:, :, 3:4])
    srgb = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.maximum(rgb, 0) ** (1 / 2.4) - 0.055)
    return Image.fromarray(np.rint(np.clip(srgb, 0, 1) * 255).astype(np.uint8))


def exports(image, output):
    preview = flatten(image, "#ffffff")
    preview.thumbnail((1600, 1600))
    result = {
        "foreground": (png(image), "image/png"),
        "foreground_mask": (png(image.getchannel("A")), "image/png"),
        "preview": (png(preview), "image/png"),
    }
    final = image if output["background"] == "transparent" else flatten(image, output["background"])
    out = io.BytesIO()
    final.save(
        out,
        format="JPEG" if output["format"] == "jpeg" else "PNG",
        **({"quality": 95} if output["format"] == "jpeg" else {}),
    )
    result["export"] = (out.getvalue(), "image/jpeg" if output["format"] == "jpeg" else "image/png")
    return result
