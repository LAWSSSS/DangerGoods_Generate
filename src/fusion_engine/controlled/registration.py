"""Fit small background camera drift; preserve the original full canvas and field of view."""

from __future__ import annotations

import cv2
import numpy as np

from fusion_engine.controlled.job import ControlledJob
from fusion_engine.controlled.masks import MaskBundle


def register_background(
    scene: np.ndarray, candidate: np.ndarray, job: ControlledJob, masks: MaskBundle
) -> tuple[np.ndarray, dict]:
    settings = job.data.get("registration")
    if not settings:
        return candidate.copy(), {"status": "NOT_REQUESTED", "canvas": list(scene.shape[1::-1])}
    height, width = scene.shape[:2]
    ratio = min(1.0, settings.get("max_side", 1800) / max(height, width))
    size = round(width * ratio), round(height * ratio)
    original = cv2.resize(
        cv2.cvtColor(scene, cv2.COLOR_RGB2GRAY), size, interpolation=cv2.INTER_AREA
    )
    generated = cv2.resize(
        cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY), size, interpolation=cv2.INTER_AREA
    )
    allowed = cv2.resize(
        (~masks.arrays["edit"]).astype(np.uint8) * 255, size, interpolation=cv2.INTER_NEAREST
    )
    allowed = cv2.erode(allowed, np.ones((15, 15), np.uint8))
    detector = cv2.SIFT_create(nfeatures=7000, contrastThreshold=0.025)
    ref_points, ref_features = detector.detectAndCompute(original, allowed)
    src_points, src_features = detector.detectAndCompute(generated, allowed)
    record = {
        "method": "SIFT_ratio_RANSAC_similarity",
        "work_size": list(size),
        "canvas": [width, height],
        "cropped": False,
    }
    if ref_features is None or src_features is None:
        return candidate.copy(), {
            **record,
            "status": "UNKNOWN",
            "reason": "背景特征不足，未假装配准成功",
        }
    pairs = cv2.BFMatcher().knnMatch(src_features, ref_features, k=2)
    matches = [
        pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance
    ]
    if len(matches) < settings.get("min_inliers", 35):
        return candidate.copy(), {
            **record,
            "status": "UNKNOWN",
            "matches": len(matches),
            "reason": "可靠特征匹配不足",
        }
    src = np.float32([src_points[m.queryIdx].pt for m in matches])
    ref = np.float32([ref_points[m.trainIdx].pt for m in matches])
    cv2.setRNGSeed(42)
    matrix, inlier = cv2.estimateAffinePartial2D(
        src, ref, method=cv2.RANSAC, ransacReprojThreshold=2, maxIters=4000, confidence=0.999
    )
    if matrix is None or inlier is None:
        return candidate.copy(), {**record, "status": "UNKNOWN", "reason": "背景相似变换拟合失败"}
    selected = inlier.ravel().astype(bool)
    error = np.linalg.norm(src @ matrix[:, :2].T + matrix[:, 2] - ref, axis=1)
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    angle = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    matrix[:, 2] /= ratio
    translation = np.linalg.norm(matrix[:, 2] / [width, height])
    spatial = src[selected] / np.array(size)
    span = np.ptp(spatial, axis=0) if len(spatial) else np.zeros(2)
    valid = (
        int(selected.sum()) >= settings.get("min_inliers", 35)
        and selected.mean() >= 0.25
        and np.min(span) >= 0.35
        and abs(angle) <= settings.get("max_rotation_deg", 2)
        and abs(np.log(scale)) <= settings.get("max_scale_log", 0.04)
        and translation <= settings.get("max_translation_ratio", 0.025)
        and float(np.median(error[selected])) / ratio <= settings.get("max_reprojection_px", 4)
    )
    record.update(
        {
            "matches": len(matches),
            "inliers": int(selected.sum()),
            "inlier_fraction": float(selected.mean()),
            "rotation_deg": angle,
            "scale": scale,
            "matrix_generated_to_original": matrix.tolist(),
            "median_reprojection_px": float(np.median(error[selected])) / ratio,
            "inlier_span": span.tolist(),
        }
    )
    if not valid:
        return candidate.copy(), {
            **record,
            "status": "UNKNOWN",
            "reason": "拟合超出小幅漂移约束或背景覆盖不足",
        }
    warped = cv2.warpAffine(
        candidate, matrix, (width, height), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT
    )
    coverage = cv2.warpAffine(
        np.ones((height, width), np.uint8), matrix, (width, height), flags=cv2.INTER_NEAREST
    ).astype(bool)
    warped[~coverage] = scene[~coverage]
    return warped, {
        **record,
        "status": "REGISTERED",
        "original_edge_pixels_filled": int((~coverage).sum()),
        "claim": "仅校正背景小幅二维漂移；不改变相机视野，不把物体移动到支点，不证明三维接触",
    }
