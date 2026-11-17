"""Conservative illumination transport on original texture; never copy a patch."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from fusion_engine.controlled.grounding import contact_diagnostics
from fusion_engine.controlled.job import ControlledJob
from fusion_engine.controlled.masks import MaskBundle, read_mask
from fusion_engine.controlled.photometry import linear_rgb


@dataclass
class Interaction:
    transmission: np.ndarray
    receiver: np.ndarray
    report: dict


def empty_interaction(shape: tuple[int, int], status: str, reason: str) -> Interaction:
    return Interaction(
        np.ones(shape, np.float32), np.zeros(shape, bool), {"status": status, "reason": reason}
    )


def joint_smooth(
    values: np.ndarray, guide: np.ndarray, valid: np.ndarray, *, radius: int = 3
) -> np.ndarray:
    """Small joint bilateral kernel; invalid samples cannot contribute or receive.

    Component labels and RGB guidance inhibit mixing across authorized boundaries.
    This is a local numerical filter, not a learned shadow or depth model.
    """
    _, labels = cv2.connectedComponents(valid.astype(np.uint8), connectivity=8)
    h, w = values.shape
    guide = guide.astype(np.float32) / 255
    numerator, denominator = np.zeros_like(values), np.zeros_like(values)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            y0, y1, x0, x1 = max(0, -dy), min(h, h - dy), max(0, -dx), min(w, w - dx)
            dst, src = np.s_[y0:y1, x0:x1], np.s_[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
            difference = np.sum((guide[dst] - guide[src]) ** 2, axis=2)
            same = valid[dst] & valid[src] & (labels[dst] == labels[src])
            weight = np.exp(-difference / (2 * 0.08**2) - (dx * dx + dy * dy) / 8) * same
            numerator[dst] += values[src] * weight
            denominator[dst] += weight
    result = np.zeros_like(values)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def extract_shadow(
    scene: np.ndarray,
    candidate: np.ndarray,
    receiver: np.ndarray,
    allowed: np.ndarray,
    object_mask: np.ndarray,
    settings: dict,
    *,
    registration_trusted: bool,
    drift_trusted: bool,
) -> Interaction:
    shape = scene.shape[:2]
    if not registration_trusted or not drift_trusted:
        return empty_interaction(shape, "UNKNOWN", "缺少可信配准/公共背景漂移校正，不提取阴影")
    legal = receiver & allowed & ~object_mask
    if not legal.any():
        return empty_interaction(shape, "UNKNOWN", "没有可见的授权阴影接收面")
    ys, xs = np.where(legal)
    pad = 10
    roi = np.s_[
        max(0, int(ys.min()) - pad) : min(shape[0], int(ys.max()) + pad + 1),
        max(0, int(xs.min()) - pad) : min(shape[1], int(xs.max()) + pad + 1),
    ]
    b, g = linear_rgb(scene[roi]), linear_rgb(candidate[roi])
    yb, yg = b @ np.array([0.2126, 0.7152, 0.0722]), g @ np.array([0.2126, 0.7152, 0.0722])
    yb, yg = yb.astype(np.float32), yg.astype(np.float32)
    ratio = (yg + 1e-5) / (yb + 1e-5)
    cb = b / np.maximum(b.sum(axis=2, keepdims=True), 1e-5)
    cg = g / np.maximum(g.sum(axis=2, keepdims=True), 1e-5)
    chromatic_error = np.max(np.abs(cb - cg), axis=2)
    # Log high-pass texture is approximately invariant to smooth illumination.
    log_b, log_g = np.log(np.maximum(yb, 1e-4)), np.log(np.maximum(yg, 1e-4))
    high_b = log_b - cv2.GaussianBlur(log_b, (0, 0), 1.2)
    high_g = log_g - cv2.GaussianBlur(log_g, (0, 0), 1.2)
    texture_error = cv2.boxFilter(np.abs(high_b - high_g), -1, (5, 5))
    valid = (
        legal[roi]
        & (yb > 0.015)
        & (yb < 0.9)
        & (yg > 0.008)
        & (ratio <= 1.02)
        & (ratio >= 1 - settings.get("max_darkening", 0.22))
        & (chromatic_error <= settings.get("max_chromatic_error", 0.035))
        & (texture_error <= settings.get("max_texture_log_error", 0.12))
    )
    minimum = settings.get("min_valid_pixels", 128)
    count = int(valid.sum())
    fraction = float(count / legal.sum())
    diagnostic = {
        "method": "aligned_linear_luminance_ratio_joint_bilateral_on_original_texture",
        "legal_receiver_pixels": int(legal.sum()),
        "valid_pixels": count,
        "valid_fraction": fraction,
        "rejected_pixels": int(legal.sum()) - count,
        "settings": settings,
        "candidate_texture_copied": False,
        "physical_shadow_verified": False,
        "annotation_eligible": False,
    }
    if count < minimum or fraction < settings.get("min_valid_fraction", 0.35):
        result = empty_interaction(shape, "UNKNOWN", "接收面的颜色/纹理一致性证据不足，未放宽阈值")
        result.report.update(diagnostic)
        return result
    dark = np.clip(1 - ratio, 0, settings.get("max_darkening", 0.22))
    dark[~valid] = 0
    dark = joint_smooth(dark, scene[roi], valid)
    selected = valid & (dark >= settings.get("min_darkening", 0.02))
    count_components, labels, stats, _ = cv2.connectedComponentsWithStats(
        selected.astype(np.uint8), connectivity=8
    )
    keep = np.zeros_like(selected)
    for index in range(1, count_components):
        if stats[index, cv2.CC_STAT_AREA] >= settings.get("min_component_pixels", 16):
            keep |= labels == index
    # Blend illumination at the evidence boundary; never soften the opaque body.
    edge_distance = cv2.distanceTransform(keep.astype(np.uint8), cv2.DIST_L2, 5)
    dark *= np.minimum(edge_distance / 2, 1)
    output = np.ones(shape, np.float32)
    scope = np.zeros(shape, bool)
    output_roi = output[roi]
    output_roi[keep] = 1 - dark[keep]
    scope[roi] = keep
    status = "ESTIMATED" if keep.any() else "UNKNOWN"
    return Interaction(
        output,
        scope,
        {
            **diagnostic,
            "status": status,
            "reason": "仅候选支持的二维照度衰减估计，仍需复核"
            if keep.any()
            else "未发现连贯的合法暗化证据；未虚构投影",
            "attenuated_pixels": int((output < 1).sum()),
            "maximum_darkening": float(1 - output.min()),
        },
    )


def contact_layer(
    job: ControlledJob, masks: MaskBundle, visible: dict[str, np.ndarray]
) -> Interaction:
    shape = masks.arrays["edit"].shape
    settings = job.data.get("contact_rendering", {})
    if not settings.get("enabled"):
        return empty_interaction(shape, "NOT_REQUESTED", "接触 AO 已关闭")
    diagnostic = contact_diagnostics(job, masks, visible)
    all_visible = np.logical_or.reduce(list(visible.values()))
    supports = {
        r["id"]: read_mask(job, r["mask"], shape)[0] for r in job.data["scene"]["support_regions"]
    }
    transmission = np.ones(shape, np.float32)
    receiver_union, records = np.zeros(shape, bool), []
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    for item in job.data["objects"]:
        oid = item["id"]
        evidence = diagnostic["objects"].get(oid, {})
        # A known support failure cannot be hidden by adding one darker contact.
        partial = job.data.get("feature_flags", {}).get("v3_structure_preservation", False)
        if (
            evidence.get("status") == "FAIL"
            or (evidence.get("status") != "PASS" and not partial)
            or oid not in visible
        ):
            records.append({"object_id": oid, "status": "SKIPPED_SUPPORT_NOT_PASS"})
            continue
        distance = cv2.distanceTransform((~visible[oid]).astype(np.uint8), cv2.DIST_L2, 5)
        for index, contact in enumerate(item["support_contacts"]):
            measured = evidence.get("contacts", [])
            sid = contact["support_id"]
            if index >= len(measured) or measured[index]["status"] != "PASS" or sid not in supports:
                continue
            x, y = np.asarray(contact["point"]) * [shape[1], shape[0]]
            local = (xx - x) ** 2 + (yy - y) ** 2 <= settings.get("search_radius_px", 40) ** 2
            sigma = settings.get("sigma_px", 3)
            receiver = (
                local
                & supports[sid]
                & masks.objects[oid].arrays["contact"]
                & masks.objects[oid].arrays["edit"]
                & ~all_visible
                & ~masks.arrays["fg_occ"]
                & ~masks.arrays["protect"]
                & (distance <= 3 * sigma)
            )
            weight = settings.get("strength", 0.12) * np.exp(-0.5 * (distance / sigma) ** 2)
            receiver &= weight > 1e-5
            transmission[receiver] = np.minimum(transmission[receiver], 1 - weight[receiver])
            receiver_union |= receiver
            records.append(
                {
                    "object_id": oid,
                    "support_id": sid,
                    "contact_index": index,
                    "status": "ESTIMATED" if receiver.any() else "NO_EFFECT",
                    "pixels": int(receiver.sum()),
                    "gap_px": measured[index]["gap_px"],
                    "overall_support_status": evidence.get("status", "UNKNOWN"),
                }
            )
    return Interaction(
        transmission,
        receiver_union,
        {
            "status": "ESTIMATED" if receiver_union.any() else "SKIPPED_NO_MEASURED_CONTACT",
            "contacts": records,
            "strength_linear": settings.get("strength", 0.12),
            "maximum_darkening": float(1 - transmission.min()),
            "claim": "二维接触核验后的窄带照度近似，不是物理 AO 或承重证明",
            "partial_contacts_do_not_upgrade_overall_support": True,
        },
    )
