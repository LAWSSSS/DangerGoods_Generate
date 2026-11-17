"""Robust paired-background photometry; material colours are never histogram-matched to steel."""

from __future__ import annotations

import cv2
import numpy as np

from fusion_engine.controlled.job import ControlledJob, evidence_confirmed
from fusion_engine.controlled.masks import MaskBundle, read_mask
from fusion_engine.errors import ConfigError


def linear_rgb(rgb: np.ndarray) -> np.ndarray:
    value = rgb.astype(np.float32) / 255.0
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def encode_rgb(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0, 1)
    srgb = np.where(value <= 0.0031308, 12.92 * value, 1.055 * value ** (1 / 2.4) - 0.055)
    return np.clip(np.rint(srgb * 255), 0, 255).astype(np.uint8)


def _permitted(job: ControlledJob, descriptor: dict) -> bool:
    return evidence_confirmed(descriptor["evidence"]) or (
        job.data["mode"] == "A" and job.data["harmonization"].get("allow_proposed_anchors", False)
    )


def photometric_plan(job: ControlledJob, scene: np.ndarray, masks: MaskBundle) -> dict:
    """Validate scopes before a network call and expose measured scene samples to prompts."""
    settings, anchors = job.data["harmonization"], []
    if settings["mode"] == "paired_local" and not settings.get("paired_anchors"):
        raise ConfigError("paired_local 必须提供原背景中未被编辑的配对采样区")
    for anchor in settings.get("paired_anchors", []):
        region, source = read_mask(job, anchor["mask"], scene.shape[:2])
        if not _permitted(job, anchor["mask"]):
            raise ConfigError("配对采样区未经审核；只有A预览可显式允许建议性ROI")
        if np.any(region & masks.arrays["edit"]) or region.sum() < settings.get(
            "min_anchor_pixels", 128
        ):
            raise ConfigError("配对采样区必须位于编辑区外且达到最小像素数")
        pixels = scene[region]
        ys, xs = np.where(region)
        lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
        anchors.append(
            {
                "id": anchor["id"],
                "description": anchor["description"],
                "source": source,
                "center_normalized": [
                    float(xs.mean() / scene.shape[1]),
                    float(ys.mean() / scene.shape[0]),
                ],
                "pixels": int(region.sum()),
                "rgb_median": np.median(pixels, axis=0).tolist(),
                "lab8_median": np.median(lab, axis=0).tolist(),
                "l_quantiles": np.quantile(lab[:, 0], [0.1, 0.5, 0.9]).tolist(),
                "evidence_confirmed": evidence_confirmed(anchor["mask"]["evidence"]),
            }
        )
    occupied = np.zeros(scene.shape[:2], bool)
    ids = {item["id"] for item in job.data["objects"]}
    for patch in settings.get("surface_adjustments", []):
        if patch["object_id"] not in ids:
            raise ConfigError("表面光色控制引用未知物体")
        region, _ = read_mask(job, patch["region"], scene.shape[:2])
        target = masks.objects[patch["object_id"]].arrays["edit"]
        selected = region & target
        if not _permitted(job, patch["region"]) or not selected.any():
            raise ConfigError("表面光色控制需要有效区域与证据；A可显式允许建议区域")
        if np.any(occupied & selected):
            raise ConfigError("表面光色控制区域重叠，禁止不透明的累积调色")
        occupied |= selected
    return {
        "mode": settings["mode"],
        "anchors": anchors,
        "cct_kelvin_estimated": False,
        "scope": "原图采样是观测值，区域语义/光照假设按各自证据记录；RGB8解码默认sRGB，未做ICC变换",
    }


def paired_local_correction(
    scene: np.ndarray,
    candidate: np.ndarray,
    job: ControlledJob,
    masks: MaskBundle,
    target: np.ndarray,
) -> tuple[np.ndarray, dict]:
    settings = job.data["harmonization"]
    if not target.any():
        raise ConfigError("局部光色校正没有有效物体区域，不能假装已执行")
    anchors, corrections, centers, rejected = [], [], [], []

    def reject(anchor: dict, reason: str, **details) -> None:
        if settings.get("anchor_failure_policy", "stop") == "stop":
            raise ConfigError(reason)
        rejected.append({"id": anchor["id"], "status": "REJECTED", "reason": reason, **details})

    for anchor in settings["paired_anchors"]:
        region, source = read_mask(job, anchor["mask"], scene.shape[:2])
        ref, raw = scene[region], candidate[region]
        valid = (
            (ref.min(axis=1) >= 16)
            & (raw.min(axis=1) >= 16)
            & (ref.max(axis=1) <= 242)
            & (raw.max(axis=1) <= 242)
        )
        indexes = np.flatnonzero(valid)
        minimum = settings.get("min_anchor_pixels", 128)
        if len(indexes) < minimum:
            reject(anchor, "配对采样区可用中间调不足；未用深黑/过曝像素估计白平衡")
            continue
        indexes = indexes[:: max(1, int(np.ceil(len(indexes) / 60000)))]
        a, b = linear_rgb(ref[indexes]), linear_rgb(raw[indexes])
        log_ratio = np.log2(np.maximum(a, 1e-5)) - np.log2(np.maximum(b, 1e-5))
        initial = np.median(log_ratio, axis=0)
        errors = np.max(np.abs(log_ratio - initial), axis=1)
        mad = float(np.median(np.abs(errors - np.median(errors))))
        inliers = errors <= min(0.6, max(0.10, float(np.median(errors)) + 2.5 * 1.4826 * mad))
        fraction = float(inliers.mean())
        if inliers.sum() < minimum or fraction < settings.get("min_pair_inlier_fraction", 0.6):
            reject(
                anchor,
                "配对背景变化/错位过大，无法可靠估计局部曝光与偏色",
                inlier_fraction=fraction,
            )
            continue
        requested = np.median(log_ratio[inliers], axis=0)
        exposure = float(np.mean(requested))
        chromatic = requested - exposure
        chromatic_bound = settings.get("max_chromatic_stops", 0.18)
        # Scale the chromatic vector instead of clipping components: preserve zero mean.
        chromatic *= min(1.0, chromatic_bound / max(float(np.max(np.abs(chromatic))), 1e-8))
        exposure_bound = settings.get("max_exposure_stops", 0.4)
        stops = (np.clip(exposure, -exposure_bound, exposure_bound) + chromatic) * settings.get(
            "strength", 0.75
        )
        residual = float(np.median(np.mean(np.abs(a[inliers] - b[inliers] * 2**requested), axis=1)))
        if residual > settings.get("max_pair_residual", 0.06):
            reject(anchor, "配对背景拟合残差过大；该采样区未参与校色", linear_rgb_residual=residual)
            continue
        ys, xs = np.where(region)
        centers.append([float(xs.mean() / scene.shape[1]), float(ys.mean() / scene.shape[0])])
        corrections.append(stops)
        anchors.append(
            {
                "id": anchor["id"],
                "source": source,
                "usable_pixels": int(len(indexes)),
                "inlier_fraction": fraction,
                "linear_rgb_residual": residual,
                "requested_log2_rgb": requested.tolist(),
                "applied_log2_rgb": stops.tolist(),
                "center_normalized": centers[-1],
                "requested_exposure_stops": exposure,
                "requested_chromatic_stops": (requested - exposure).tolist(),
                "bounded": bool(
                    abs(exposure) > exposure_bound
                    or np.max(np.abs(requested - exposure)) > chromatic_bound
                ),
            }
        )
    if len(anchors) < settings.get("min_reliable_anchors", 1):
        raise ConfigError("可靠配对采样区不足；未放宽残差/内点阈值或伪造光色校正")
    height, width = scene.shape[:2]
    gh, gw = min(height, 96), min(width, 96)
    yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float32)
    points = np.stack(((xx + 0.5) / gw, (yy + 0.5) / gh), axis=-1)
    distances = np.sum((points[..., None, :] - np.asarray(centers)[None, None, ...]) ** 2, axis=-1)
    weights = 1 / np.maximum(distances, 0.015**2)
    weights /= weights.sum(axis=-1, keepdims=True)
    field = weights @ np.asarray(corrections, np.float32)
    field = cv2.resize(field, (width, height), interpolation=cv2.INTER_LINEAR)
    result = candidate.copy()
    adjusted = encode_rgb(linear_rgb(candidate[target]) * 2 ** field[target])
    distance = cv2.distanceTransform(target.astype(np.uint8), cv2.DIST_L2, 5)[target]
    alpha = np.minimum(distance / 3.0, 1.0)
    result[target] = np.clip(
        np.rint(candidate[target] * (1 - alpha[:, None]) + adjusted * alpha[:, None]), 0, 255
    ).astype(np.uint8)
    return result, {
        "operation": "paired_background_local_linear_RGB",
        "anchors": anchors,
        "rejected_anchors": rejected,
        "anchor_failure_policy": settings.get("anchor_failure_policy", "stop"),
        "interpolation": "inverse_distance_squared_on_96_grid",
        "corrected_pixels": int(target.sum()),
        "inward_colour_fade_px": 3,
        "background_corrected": False,
        "claim": "校正生成过程的局部曝光/通道偏移；未重建三维照明，未估计真实开尔文色温",
        "material_histogram_equalization": False,
    }


def adjust_surfaces(
    candidate: np.ndarray,
    job: ControlledJob,
    masks: MaskBundle,
    target: np.ndarray,
    estimated_visible: dict[str, np.ndarray] | None = None,
    estimated_parts: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    result, records = candidate.copy(), []
    if job.data.get("fusion_v3", {}).get("surface_policy") == "paired_only":
        return result, [
            {
                "operation": "surface_response",
                "status": "SKIPPED_FIXED_SURFACE_GRADING",
                "reason": "新路线仅应用有可靠背景配对依据的有界曝光/通道校正，不重复固定压灰与压高光",
            }
        ]
    for patch in job.data["harmonization"].get("surface_adjustments", []):
        region, record = read_mask(job, patch["region"], result.shape[:2])
        if estimated_parts is not None:
            key = patch["object_id"] + "/" + patch["id"]
            region = estimated_parts.get(key, np.zeros(result.shape[:2], bool)).copy()
            record = {
                "source": "candidate_bound_VLM_surface_proposal_with_explicit_unassigned_surface_policy"
                if patch.get("fill_unassigned_visible")
                else "candidate_bound_VLM_surface_proposal_intersected_with_estimated_visible",
                "part_key": key,
                "annotation_eligible": False,
            }
        region &= masks.objects[patch["object_id"]].arrays["edit"] & target
        if estimated_visible is not None:
            region &= estimated_visible.get(patch["object_id"], np.zeros(result.shape[:2], bool))
        if not region.any():
            records.append(
                {
                    "operation": "surface_response",
                    "id": patch["id"],
                    "status": "SKIPPED_EMPTY_ESTIMATED_SURFACE",
                }
            )
            continue
        lab = cv2.cvtColor(
            result[region].astype(np.float32).reshape(-1, 1, 3) / 255, cv2.COLOR_RGB2LAB
        ).reshape(-1, 3)
        chroma = np.linalg.norm(lab[:, 1:], axis=1)
        factor = np.full(len(lab), patch.get("chroma_scale", 1), np.float32)
        if "max_lab_chroma" in patch:
            factor = np.minimum(factor, patch["max_lab_chroma"] / np.maximum(chroma, 1e-6))
        lab[:, 1:] *= factor[:, None]
        knee, slope = patch.get("highlight_knee_l", 100), patch.get("highlight_slope", 1)
        lab[:, 0] = np.where(lab[:, 0] > knee, knee + (lab[:, 0] - knee) * slope, lab[:, 0])
        adjusted = (
            cv2.cvtColor(lab.astype(np.float32).reshape(-1, 1, 3), cv2.COLOR_LAB2RGB).reshape(-1, 3)
            * 255
        )
        # Inward fade applies to the actual selected surface only, without painting adjacent steel.
        distance = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)[region]
        alpha = np.minimum(distance / 3.0, 1.0) * patch.get("strength", 1)
        result[region] = np.clip(
            np.rint(result[region] * (1 - alpha[:, None]) + adjusted * alpha[:, None]), 0, 255
        ).astype(np.uint8)
        records.append(
            {
                "operation": "surface_response",
                "id": patch["id"],
                "pixels": int(region.sum()),
                "region": record,
                "settings": {key: value for key, value in patch.items() if key != "region"},
                "claim": "建议或审核的表面响应约束，不是实测反射率/自动判定材质",
            }
        )
    return result, records
