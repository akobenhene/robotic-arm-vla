"""Visualization helpers: HUD overlays and side-by-side policy comparison frames."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int = 16) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Load a readable TrueType font with a safe default fallback."""
    for name in ("arial.ttf", "DejaVuSans.ttf", "calibri.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def overlay_hud(
    frame: np.ndarray,
    *,
    title: str,
    seed: int,
    step: int,
    reward: float,
    max_reward: float,
    success: bool,
) -> np.ndarray:
    """Burn a compact status HUD onto an RGB frame.

    Parameters
    ----------
    frame:
        uint8 array, shape ``(H, W, 3)``.

    Returns
    -------
    out:
        uint8 array, same shape as ``frame``.
    """
    img = Image.fromarray(np.ascontiguousarray(frame), mode="RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    font = _font(15)
    status = "SUCCESS" if success else "RUNNING"
    color = (40, 200, 90, 230) if success else (240, 200, 60, 230)
    lines = [
        f"{title}",
        f"seed={seed}  step={step}  r={reward:.0f}  max={max_reward:.0f}",
        f"status={status}",
    ]
    pad = 6
    text = "\n".join(lines)
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=2)
    box_w = bbox[2] - bbox[0] + 2 * pad
    box_h = bbox[3] - bbox[1] + 2 * pad
    draw.rectangle((8, 8, 8 + box_w, 8 + box_h), fill=(0, 0, 0, 160))
    draw.multiline_text((8 + pad, 8 + pad), text, font=font, fill=color, spacing=2)
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def resize_frame(frame: np.ndarray, height: int = 240) -> np.ndarray:
    """Resize HWC uint8 frame to a target height, preserving aspect ratio."""
    h, w = frame.shape[:2]
    if h == height:
        return frame
    new_w = max(1, int(round(w * (height / float(h)))))
    img = Image.fromarray(frame).resize((new_w, height), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def side_by_side(
    left: np.ndarray,
    right: np.ndarray,
    *,
    gap: int = 8,
    gap_color: tuple[int, int, int] = (20, 20, 20),
) -> np.ndarray:
    """Horizontally concatenate two RGB frames (resized to matching height)."""
    left_r = resize_frame(left, 240)
    right_r = resize_frame(right, 240)
    h = max(left_r.shape[0], right_r.shape[0])
    spacer = np.full((h, gap, 3), gap_color, dtype=np.uint8)
    return np.concatenate([left_r, spacer, right_r], axis=1)
