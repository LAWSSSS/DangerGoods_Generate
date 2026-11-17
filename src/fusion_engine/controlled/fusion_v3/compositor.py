"""Strict, region-scoped, linear-light opaque-object compositing.

Adapted from the user's V3 reference design. No depth, matting, relighting or
physical shadow estimator is implied by this numerical compositor.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import cv2
import numpy as np

from fusion_engine.controlled.photometry import encode_rgb, linear_rgb
from fusion_engine.errors import ConfigError


@dataclass
class Layers:
    object: np.ndarray
    core: np.ndarray
    edge: np.ndarray
    front: np.ndarray
    contact: np.ndarray
    cast: np.ndarray
    edit: np.ndarray
    protect: np.ndarray

    @property
    def authorized(self) -> np.ndarray:
        return (self.object | self.contact | self.cast) & self.edit & ~self.protect & ~self.front


def binary_partition(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # This is a morphological partition of M2 evidence, not a confidence upgrade.
    core = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return core, mask & ~core


def validate_layers(background: np.ndarray, foreground: np.ndarray, layers: Layers) -> None:
    if background.ndim != 3 or background.shape[2] != 3 or background.dtype != np.uint8:
        raise ConfigError("V3 background 必须为 H×W×3 RGB8")
    if foreground.shape != background.shape or foreground.dtype != np.uint8:
        raise ConfigError("V3 foreground 必须与原图同尺寸、同 RGB8 类型，禁止隐式缩放")
    for field in fields(layers):
        value = getattr(layers, field.name)
        if value.shape != background.shape[:2] or value.dtype != np.bool_:
            raise ConfigError(f"V3 {field.name} 必须为原图尺寸 bool Mask")
    if not layers.object.any():
        raise ConfigError("V3 不接受空物体 Mask；不能用计划框替代候选实体")
    if np.any(layers.edit & layers.protect):
        raise ConfigError("V3 edit/protect 冲突，必须修订授权范围")
    if np.any(layers.core & layers.edge) or not np.array_equal(
        layers.core | layers.edge, layers.object
    ):
        raise ConfigError("V3 core/edge 必须互斥且完整覆盖实体")
    if np.any((layers.object & ~layers.front) & (~layers.edit | layers.protect)):
        raise ConfigError("V3 物体越过编辑范围或进入保护区，禁止裁断主体")
    interactions = layers.contact | layers.cast
    if np.any(interactions & (~layers.edit | layers.protect | layers.front | layers.object)):
        raise ConfigError("V3 阴影接收区必须位于合法、可见的原背景表面")


def _transmission(value: np.ndarray | None, mask: np.ndarray, name: str) -> np.ndarray:
    if value is None:
        if mask.any():
            raise ConfigError(f"V3 {name} 有接收区但缺少透过率证据")
        return np.ones(mask.shape, np.float32)
    if value.shape != mask.shape or not np.issubdtype(value.dtype, np.floating):
        raise ConfigError(f"V3 {name} 透过率必须为原图尺寸浮点数组")
    if not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
        raise ConfigError(f"V3 {name} 透过率必须有限且处于[0,1]")
    if np.any(value[~mask] != 1):
        raise ConfigError(f"V3 {name} 不得在接收区外衰减")
    return value


def compose_layers(
    background: np.ndarray,
    foreground: np.ndarray,
    layers: Layers,
    *,
    cast_transmission: np.ndarray | None = None,
    contact_transmission: np.ndarray | None = None,
    object_alpha: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    validate_layers(background, foreground, layers)
    alpha = layers.object.astype(np.float32) if object_alpha is None else object_alpha
    if (
        alpha.shape != layers.object.shape
        or not np.issubdtype(alpha.dtype, np.floating)
        or not np.isfinite(alpha).all()
        or np.any((alpha < 0) | (alpha > 1))
        or np.any(alpha[~layers.object] != 0)
        or np.any(alpha[layers.core] != 1)
    ):
        raise ConfigError("V3 alpha 只能表示边缘真实覆盖，实体内部必须完全不透明")
    cast = _transmission(cast_transmission, layers.cast, "cast")
    contact = _transmission(contact_transmission, layers.contact, "contact")
    # Both estimates can describe the same occlusion. Do not multiply twice.
    transmission = np.minimum(cast, contact)
    authorized = layers.authorized
    yy, xx = np.where(authorized)
    if not len(xx):
        raise ConfigError("V3 没有可授权合成像素")
    x0, x1, y0, y1 = int(xx.min()), int(xx.max()) + 1, int(yy.min()), int(yy.max()) + 1
    roi = np.s_[y0:y1, x0:x1]
    coverage = alpha[roi] * (~layers.front[roi] & ~layers.protect[roi])
    backing = linear_rgb(background[roi]) * transmission[roi][..., None]
    mixed = linear_rgb(foreground[roi]) * coverage[..., None] + backing * (1 - coverage[..., None])
    result = background.copy()
    rendered = encode_rgb(mixed)
    result_roi = result[roi]
    result_roi[authorized[roi]] = rendered[authorized[roi]]
    result[layers.front] = background[layers.front]
    result[layers.protect] = background[layers.protect]
    changed = np.any(result != background, axis=2)
    return result, {
        "operation": "v3_linear_light_layer_compositor",
        "status": "APPLIED",
        "canvas": list(background.shape[1::-1]),
        "cropped": False,
        "working_roi_pixels": [x0, y0, x1, y1],
        "working_roi_is_not_output_crop": True,
        "mask_pixels": {f.name: int(getattr(layers, f.name).sum()) for f in fields(layers)},
        "alpha_policy": "binary_opaque_M2_estimate_no_uncertainty_feather"
        if object_alpha is None
        else "supplied_coverage_alpha_core_opaque",
        "shadow_overlap_policy": "stronger_attenuation_not_double_multiplication",
        "changed_outside_authorized": int((changed & ~authorized).sum()),
        "changed_protected_before_save": int((changed & layers.protect).sum()),
        "front_restored_pixels": int(layers.front.sum()),
        "foreground_restoration_present": bool(layers.front.any()),
        "annotation_eligible": False,
        "physical_light_transport_performed": False,
    }
