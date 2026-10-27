"""Explicit user-requested appearance corrections on current visible object estimates."""

from __future__ import annotations

import math

import cv2
import numpy as np

from fusion_engine.controlled.job import ControlledJob
from fusion_engine.controlled.masks import MaskBundle
from fusion_engine.errors import ConfigError


def apply_user_appearance_feedback(
    candidate: np.ndarray,
    job: ControlledJob,
    masks: MaskBundle,
    target: np.ndarray,
    estimated_visible: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, list[dict]]:
    """Adjust colour/detail, never alpha, geometry, holes, contacts or background.

    Parameters are bounded manual corrections, not an estimate of camera MTF,
    material reflectance, illumination or physical support. No planned mask is
    substituted when a current candidate mask is unavailable.
    """
    settings = job.data.get("harmonization", {}).get("feedback_correction", {})
    if not settings.get("enabled", False):
        return candidate, []
    if estimated_visible is None:
        return candidate, [
            {
                "operation": "user_requested_local_appearance",
                "status": "SKIPPED_NO_CURRENT_VISIBLE_ESTIMATE",
                "reason": "没有当前候选可见轮廓；不使用计划区域代替实体",
            }
        ]
    result, records = candidate.copy(), []
    chroma_scale = settings["chroma_scale"]
    sigma = settings["detail_sigma_px"]
    excluded = masks.arrays["protect"] | masks.arrays["fg_occ"]
    for oid in settings["object_ids"]:
        actual = estimated_visible.get(oid)
        if actual is None or not actual.any():
            records.append(
                {
                    "operation": "user_requested_local_appearance",
                    "object_id": oid,
                    "status": "SKIPPED_NO_CURRENT_VISIBLE_ESTIMATE",
                }
            )
            continue
        if actual.shape != candidate.shape[:2] or actual.dtype != np.bool_:
            raise ConfigError("外观反馈需要同画布的布尔可见轮廓")
        region = actual & target & masks.objects[oid].arrays["edit"] & ~excluded
        if not region.any():
            records.append(
                {
                    "operation": "user_requested_local_appearance",
                    "object_id": oid,
                    "status": "SKIPPED_EMPTY_AUTHORIZED_OBJECT",
                }
            )
            continue
        ys, xs = np.where(region)
        pad = max(1, math.ceil(3 * sigma))
        y0, y1 = max(0, int(ys.min()) - pad), min(candidate.shape[0], int(ys.max()) + pad + 1)
        x0, x1 = max(0, int(xs.min()) - pad), min(candidate.shape[1], int(xs.max()) + pad + 1)
        selected = region[y0:y1, x0:x1]
        patch = result[y0:y1, x0:x1].astype(np.float32)
        before = patch[selected].copy()
        lab = cv2.cvtColor((before / 255).reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
        input_chroma = np.linalg.norm(lab[:, 1:], axis=1)
        if chroma_scale != 1:
            lab[:, 1:] *= chroma_scale
            patch[selected] = cv2.cvtColor(
                lab.reshape(-1, 1, 3), cv2.COLOR_LAB2RGB
            ).reshape(-1, 3) * 255
        if sigma:
            # Normalize by actual object coverage: do not pull steel/background
            # through real holes or introduce dark/transparent cutout borders.
            weights = selected.astype(np.float32)
            denominator = cv2.GaussianBlur(
                weights, (0, 0), sigma, borderType=cv2.BORDER_CONSTANT
            )
            numerator = cv2.GaussianBlur(
                patch * weights[..., None], (0, 0), sigma, borderType=cv2.BORDER_CONSTANT
            )
            patch[selected] = (numerator / np.maximum(denominator[..., None], 1e-6))[selected]
        view = result[y0:y1, x0:x1]
        view[selected] = np.clip(np.rint(patch[selected]), 0, 255).astype(np.uint8)
        output_lab = cv2.cvtColor(
            (view[selected].astype(np.float32) / 255).reshape(-1, 1, 3), cv2.COLOR_RGB2LAB
        ).reshape(-1, 3)
        records.append(
            {
                "operation": "user_requested_local_appearance",
                "object_id": oid,
                "status": "APPLIED",
                "reason": settings["reason"],
                "pixels": int(region.sum()),
                "changed_pixels": int(np.any(view[selected] != before, axis=1).sum()),
                "chroma_scale": chroma_scale,
                "detail_sigma_px": sigma,
                "input_chroma_p50_p90": np.quantile(input_chroma, [0.5, 0.9]).tolist(),
                "output_chroma_p50_p90": np.quantile(
                    np.linalg.norm(output_lab[:, 1:], axis=1), [0.5, 0.9]
                ).tolist(),
                "mask_source": "当前候选可见轮廓估计，不是分割真值",
                "alpha_or_geometry_modified": False,
                "canvas_resized_or_cropped": False,
                "noise_added": False,
                "claim": "用户明确反馈支持的局部有界校正，不是相机标定或自然度验收",
            }
        )
    return result, records
