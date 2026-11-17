"""Machine gates only: no invented realism, geometry or material scores."""

from typing import Any

import numpy as np


def protected_delta(
    original: np.ndarray, candidate: np.ndarray, protect: np.ndarray
) -> dict[str, Any]:
    difference = np.abs(original.astype(np.int16) - candidate.astype(np.int16))
    changed = np.any(difference != 0, axis=2) & protect
    count = int(protect.sum())
    return {
        "protected_pixel_count": count,
        "changed_pixel_count": int(changed.sum()),
        "changed_fraction": float(changed.sum() / count) if count else 0.0,
        "mean_absolute_error": float(difference[protect].mean()) if count else 0.0,
        "max_channel_delta": int(difference[protect].max()) if count else 0,
        "scope": "exif_normalized_decoded_rgb8_lossless_master",
        "applicable": count > 0,
    }


def evaluate_candidate(
    original: np.ndarray,
    candidate: np.ndarray,
    protect: np.ndarray,
    editable: np.ndarray,
    max_changed_fraction: float,
) -> dict[str, Any]:
    dimensions_pass = original.shape == candidate.shape
    if not dimensions_pass:
        return {"machine_pass": False, "reasons": ["dimensions_mismatch"], "dimensions_pass": False}
    background = protected_delta(original, candidate, protect)
    background_pass = background["changed_fraction"] <= max_changed_fraction
    visible_change_count = int((np.any(original != candidate, axis=2) & editable).sum())
    reasons = []
    if not background_pass:
        reasons.append("protected_background_changed")
    if not visible_change_count:
        reasons.append("no_change_in_editable_area")
    return {
        "machine_pass": background_pass and visible_change_count > 0,
        "dimensions_pass": True,
        "background": background,
        "editable_changed_pixel_count": visible_change_count,
        "reasons": reasons,
        "visual_realism": "not_evaluated",
        "geometry": "not_evaluated",
        "material": "not_evaluated",
        "annotation_correctness": "not_evaluated",
    }
