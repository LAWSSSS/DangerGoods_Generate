"""Explicit context crop and invertible original/request coordinate mapping."""

from __future__ import annotations

import numpy as np

from fusion_engine.controlled.geometry import obb_vertices
from fusion_engine.controlled.job import ControlledJob
from fusion_engine.controlled.masks import MaskBundle, read_mask
from fusion_engine.errors import ConfigError


def request_frame(job: ControlledJob, masks: MaskBundle, shape: tuple[int, int]) -> dict:
    height, width = shape
    if job.data["generator"].get("preserve_full_frame") and job.data["generator"].get(
        "context_crop"
    ):
        raise ConfigError("preserve_full_frame禁止上下文裁切；请使用完整原图输入")
    if not job.data["generator"].get("context_crop", False):
        return {
            "kind": "original",
            "rect_pixels": [0, 0, width, height],
            "size": [width, height],
            "original_size": [width, height],
            "field_of_view_preserved": True,
            "cropped": False,
        }
    if job.data.get("prompt_version") != "2.2":
        raise ConfigError("局部裁切必须配合2.2提示词进行坐标映射")
    if (
        job.data["generator"]["route"] != "guided_generation_then_restore"
        or job.data["constraints"]["background_protection"] != "strict"
    ):
        raise ConfigError("局部裁切要求显式受控恢复路线与严格背景保护")
    if not masks.arrays["obj_full"].any():
        raise ConfigError("局部裁切需要非空完整形状代理；不能以全场概念画布自动退化")
    selected = masks.arrays["edit"].copy()
    for anchor in job.data["harmonization"].get("paired_anchors", []):
        selected |= read_mask(job, anchor["mask"], shape)[0]
    ys, xs = np.where(selected)
    low, high = np.array([xs.min(), ys.min()], float), np.array([xs.max() + 1, ys.max() + 1], float)
    # Include complete OBBs, even when a foreground hides part of a requested object.
    for item in job.data["objects"]:
        points = obb_vertices(item["placement"], width, height)
        low, high = np.minimum(low, points.min(axis=0)), np.maximum(high, points.max(axis=0))
    padding = job.data["generator"].get("context_padding_ratio", 0.3)
    center = (low + high) / 2
    size = (high - low) * (1 + 2 * padding)
    # Preserve original aspect: no hidden crop, stretch or change of camera projection.
    size = np.array(
        [max(size[0], size[1] * width / height), max(size[1], size[0] * height / width)]
    )
    if size[0] >= width or size[1] >= height:
        return {
            "kind": "original",
            "rect_pixels": [0, 0, width, height],
            "size": [width, height],
            "original_size": [width, height],
            "reason": "上下文已覆盖全图",
        }
    integer_size = np.ceil(size).astype(int)
    origin = np.rint(center - integer_size / 2).astype(int)
    origin = np.clip(origin, [0, 0], np.array([width, height]) - integer_size)
    end = origin + integer_size
    return {
        "kind": "context_crop",
        "rect_pixels": [*origin.tolist(), *end.tolist()],
        "size": integer_size.tolist(),
        "original_size": [width, height],
        "padding_ratio": padding,
        "reassembly": "resize_returned_crop_to_this_size_then_replace_same_original_rect",
        "binary_mask_submitted": False,
    }


def map_point(point: list[float], frame: dict | None) -> list[float]:
    if not frame:
        return point
    x0, y0, _, _ = frame["rect_pixels"]
    value = (np.asarray(point) * frame["original_size"] - [x0, y0]) / frame["size"]
    return value.tolist()


def map_placement(placement: dict, frame: dict | None) -> dict:
    if not frame:
        return placement
    width, height = (
        np.asarray([placement["width"], placement["height"]])
        * frame["original_size"]
        / frame["size"]
    )
    return {
        **placement,
        "center": map_point(placement["center"], frame),
        "width": float(width),
        "height": float(height),
    }


def reassemble(scene: np.ndarray, crop: np.ndarray, frame: dict) -> np.ndarray:
    x0, y0, x1, y1 = frame["rect_pixels"]
    if crop.shape != (y1 - y0, x1 - x0, 3):
        raise ConfigError("局部候选与声明裁切尺寸不一致，禁止错位回填")
    result = scene.copy()
    result[y0:y1, x0:x1] = crop
    return result
