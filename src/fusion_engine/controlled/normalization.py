"""One explicit, audited decode/alpha/registration policy for API and imported candidates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from fusion_engine.errors import AdapterError


def normalize_candidate(
    path: Path, target_size: tuple[int, int], settings: dict
) -> tuple[np.ndarray, dict]:
    with Image.open(path) as source:
        oriented = ImageOps.exif_transpose(source)
        record = {"source_mode": oriented.mode, "provider_size": list(oriented.size)}
        if oriented.mode == "P" and "transparency" in oriented.info:
            oriented = oriented.convert("RGBA")
        if "A" in oriented.getbands():
            alpha = np.asarray(oriented.getchannel("A"))
            record["alpha"] = {
                "min": int(alpha.min()),
                "max": int(alpha.max()),
                "nonopaque_fraction": float((alpha < 255).mean()),
                "policy": settings.get("alpha_policy", "reject"),
            }
            if alpha.min() < 255:
                minimum = settings.get("near_opaque_min_alpha", 251)
                outlier_minimum = settings.get("near_opaque_outlier_min_alpha", minimum)
                outlier_fraction = float((alpha < minimum).mean())
                maximum_outliers = settings.get("near_opaque_max_outlier_fraction", 0)
                if (
                    settings.get("alpha_policy", "reject") != "near_opaque_unassociated_rgb"
                    or alpha.min() < outlier_minimum
                    or outlier_fraction > maximum_outliers
                ):
                    raise AdapterError("候选透明度超出显式 Alpha 策略；原始文件已保留，未隐式铺底")
                record["alpha"]["operation"] = "retain_unassociated_RGB_set_alpha_opaque"
                record["alpha"]["min_allowed"] = minimum
                record["alpha"].update(
                    {
                        "outlier_min_allowed": outlier_minimum,
                        "below_bulk_minimum_pixels": int((alpha < minimum).sum()),
                        "below_bulk_minimum_fraction": outlier_fraction,
                        "maximum_outlier_fraction": maximum_outliers,
                    }
                )
                record["alpha"]["assumption"] = (
                    "显式假设近不透明通道为非预乘RGB；没有反推预乘或用背景铺底"
                )
            else:
                record["alpha"]["operation"] = "already_opaque"
        image = oriented.convert("RGB")
        resized = image.size != target_size
        if resized:
            if settings["output_registration"] != "resize_same_aspect":
                raise AdapterError("候选尺寸不同；strict 配准策略禁止缩放")
            ratio_error = abs((image.width / image.height) / (target_size[0] / target_size[1]) - 1)
            if ratio_error > 0.025:
                raise AdapterError("候选长宽比偏差超过 2.5%，禁止拉伸")
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        record.update(
            {
                "working_size": list(target_size),
                "resized_to_background": resized,
                "resampling": "LANCZOS" if resized else "none",
                "registration_limits": "只统一画布尺寸；未声称完成相机/局部特征配准",
            }
        )
        return np.asarray(image).copy(), record
