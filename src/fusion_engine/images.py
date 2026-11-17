"""Explicit RGB8 working baseline and lossless image export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from fusion_engine.errors import ConfigError


def image_info(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as source:
            source.load()
            oriented = ImageOps.exif_transpose(source)
            return {
                "path": str(path),
                "width": oriented.width,
                "height": oriented.height,
                "source_mode": source.mode,
                "source_format": source.format,
                "exif_orientation": source.getexif().get(274),
                "has_icc_profile": bool(source.info.get("icc_profile")),
            }
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ConfigError(f"无法读取图像：{path.name}") from exc


def load_rgb(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as source:
            source.load()
            source = ImageOps.exif_transpose(source)
            if "A" in source.getbands() and source.getchannel("A").getextrema()[0] != 255:
                raise ConfigError(f"场景/候选暂不支持非不透明 alpha：{path.name}")
            if source.mode == "P" and "transparency" in source.info:
                raise ConfigError(f"场景/候选暂不支持索引透明图：{path.name}")
            return np.array(source.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ConfigError(f"无法读取 RGB 图像：{path.name}") from exc


def save_png(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, format="PNG")
