"""Image-space geometry and explicit calibrated projection, never inferred metric depth."""

from __future__ import annotations

import math

import cv2
import numpy as np

from fusion_engine.errors import ConfigError


def obb_vertices(placement: dict, width: int, height: int) -> np.ndarray:
    cx, cy = np.array(placement["center"]) * [width, height]
    half_w, half_h = placement["width"] * width / 2, placement["height"] * height / 2
    angle = math.radians(placement["rotation_deg"])
    # x right, y down: a positive image rotation is clockwise.
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    local = np.array([[-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_w, half_h]])
    return local @ rotation.T + [cx, cy]


def reference_projection(data: dict) -> dict:
    baseline = data["reference_pixels"] * data["object_length"] / data["reference_length"]
    missing = [
        key for key in ("depth_ratio_range", "foreshortening_ratio_range") if key not in data
    ]
    if missing:
        return {
            "status": "UNKNOWN",
            "missing": missing,
            "unadjusted_baseline_pixels": baseline,
            "reason": "缺少深度比或缩短比区间；未用 1.0 代替未知项",
        }
    depth, gamma = data["depth_ratio_range"], data["foreshortening_ratio_range"]
    if depth[0] > depth[1] or gamma[0] > gamma[1]:
        raise ConfigError("尺度比区间必须递增")
    return {
        "status": "ESTIMATE",
        "pixel_length_range": [baseline * depth[0] * gamma[0], baseline * depth[1] * gamma[1]],
        "formula": "p_ref * L_obj/L_ref * Z_ref/Z_obj * gamma_obj/gamma_ref",
        "metric_calibration_claim": False,
    }


def project_points(camera: dict) -> dict:
    K, R = np.asarray(camera["K"], float), np.asarray(camera["R"], float)
    t, points = np.asarray(camera["t"], float), np.asarray(camera["points_3d"], float)
    if K.shape != (3, 3) or R.shape != (3, 3) or t.shape != (3,) or points.shape[1:] != (3,):
        raise ConfigError("相机矩阵/三维点形状错误")
    if not all(np.isfinite(v).all() for v in (K, R, t, points)):
        raise ConfigError("相机投影输入必须为有限数值")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(R), 1, atol=1e-5
    ):
        raise ConfigError("R 必须是正交且 det=1 的旋转矩阵")
    if not np.allclose(K[2], [0, 0, 1]) or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ConfigError("K 必须是正焦距的针孔内参矩阵")
    camera_points = points @ R.T + t
    if (camera_points[:, 2] <= 0).any():
        raise ConfigError("存在位于相机后方/相机平面的点；禁止直接投影")
    homogeneous = camera_points @ K.T
    pixels = homogeneous[:, :2] / homogeneous[:, 2:]
    return {
        "status": "COMPUTED_FROM_SUPPLIED_CALIBRATION",
        "points_xy": pixels.tolist(),
        "bbox_xyxy": [*pixels.min(axis=0).tolist(), *pixels.max(axis=0).tolist()],
        "assumptions": ["输入已标定且单位一致", "未自动估计畸变；输入应已去畸变"],
        "reconstruction_performed": False,
    }


def scale_plan(scale: dict) -> dict:
    level = scale["level"]
    if level == 0:
        return {"level": 0, "status": "IMAGE_SPACE_ONLY", "metric_calibration_claim": False}
    if level == 1:
        result = (
            reference_projection(scale["reference_projection"])
            if "reference_projection" in scale
            else {"status": "UNKNOWN", "missing": ["reference_projection"]}
        )
    elif level == 2:
        result = (
            project_points(scale["camera"])
            if "camera" in scale
            else {"status": "UNKNOWN", "missing": ["calibrated_camera_and_geometry"]}
        )
    else:
        result = {
            "status": "ESTIMATE" if scale.get("pixel_length_range") else "UNKNOWN",
            "pixel_length_range": scale.get("pixel_length_range"),
            "assumptions": scale.get("assumptions", []),
            "metric_calibration_claim": False,
        }
    return {"level": level, **result, "evidence": scale["evidence"]}


def geometry_metrics(mask: np.ndarray, placement: dict) -> dict:
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return {"status": "UNKNOWN", "reason": "独立完整轮廓不足三个像素"}
    (cx, cy), (w, h), angle = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
    target = placement["rotation_deg"]

    def angle_error(value: float) -> float:
        return abs((value - target + 90) % 180 - 90)

    options = [(w + 1, h + 1, angle), (h + 1, w + 1, angle + 90)]
    w, h, angle = min(options, key=lambda item: angle_error(item[2]))
    tw, th = placement["width"] * width, placement["height"] * height
    tx, ty = np.array(placement["center"]) * [width, height]
    return {
        "status": "COMPUTED",
        "center_error": math.hypot(cx - tx, cy - ty) / math.hypot(width, height),
        "scale_log_error": abs(math.log(w / tw)) + abs(math.log(h / th)),
        "rotation_error_deg": angle_error(angle),
        "observed_obb_xywh_angle": [cx, cy, w, h, angle],
        "basis": "independently_supplied_amodal_mask_not_visible_bbox",
    }


def mask_iou(a: np.ndarray, b: np.ndarray) -> float | None:
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else None


def distance_to_boundary(mask: np.ndarray, point: list[float]) -> float | None:
    if not mask.any():
        return None
    boundary = mask ^ cv2.erode(
        mask.astype(np.uint8),
        np.ones((3, 3), np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    distance = cv2.distanceTransform(
        (~boundary).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    height, width = mask.shape
    x = min(width - 1, round(point[0] * width))
    y = min(height - 1, round(point[1] * height))
    return float(distance[y, x])


def local_depth_order(
    edges: list[tuple[str, str, np.ndarray]], ids: list[str], shape: tuple
) -> dict:
    """A global cycle can be valid only if each pixel's local order is acyclic."""
    signatures: dict[bytes, tuple[list[tuple[str, str]], np.ndarray]] = {}
    if not edges:
        return {"status": "PASS", "regions": 0, "cycles": [], "method": "per_pixel_local_DAG"}
    masks = np.stack([mask for _, _, mask in edges], axis=-1)
    flat = masks.reshape(-1, len(edges))
    patterns = np.unique(flat, axis=0)
    cycles = []
    for pattern in patterns:
        if not pattern.any():
            continue
        active = [(a, b) for enabled, (a, b, _) in zip(pattern, edges) if enabled]
        adjacency = {key: set() for key in ids}
        indegree = {key: 0 for key in ids}
        for front, back in active:
            if back not in adjacency[front]:
                adjacency[front].add(back)
                indegree[back] += 1
        pending = [key for key in ids if indegree[key] == 0]
        visited = 0
        while pending:
            node = pending.pop()
            visited += 1
            for child in adjacency[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    pending.append(child)
        if visited != len(ids):
            region = np.all(flat == pattern, axis=-1).reshape(shape)
            cycles.append({"edges": active, "pixels": int(region.sum())})
        signatures[pattern.tobytes()] = (active, pattern)
    return {
        "status": "FAIL" if cycles else "PASS",
        "regions": len(signatures),
        "cycles": cycles,
        "method": "per_pixel_local_DAG",
        "global_sort_used": False,
    }
