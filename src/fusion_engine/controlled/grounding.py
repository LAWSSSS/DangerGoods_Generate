"""Support constraints and local contact shading; never use a shadow to bridge a geometric gap."""

from __future__ import annotations

import cv2
import numpy as np

from fusion_engine.controlled.job import ControlledJob
from fusion_engine.controlled.masks import MaskBundle, read_mask


def contact_diagnostics(
    job: ControlledJob, masks: MaskBundle, visible: dict[str, np.ndarray]
) -> dict:
    checks, objects = {}, {}
    shape = masks.arrays["edit"].shape
    height, width = shape
    supports = {
        region["id"]: read_mask(job, region["mask"], shape)[0]
        for region in job.data["scene"]["support_regions"]
    }
    for item in job.data["objects"]:
        oid = item["id"]
        actual = visible.get(oid)
        if actual is None or not actual.any():
            objects[oid] = {"status": "UNKNOWN", "reason": "没有可用的候选可见轮廓"}
            checks[f"receiver_contact_{oid}"] = {
                "status": "UNKNOWN",
                "detail": "缺少候选轮廓，无法测量支撑间隙",
            }
            continue
        border = actual & ~cv2.erode(actual.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(
            bool
        )
        distances = cv2.distanceTransform((~border).astype(np.uint8), cv2.DIST_L2, 5)
        contacts = []
        for contact in item["support_contacts"]:
            support = supports.get(contact["support_id"])
            if support is None:
                contacts.append(
                    {
                        "support_id": contact["support_id"],
                        "status": "UNKNOWN",
                        "reason": "不是可观测的原场景接收面；未假装测量物体间三维接触",
                    }
                )
                continue
            # Only the declared local receiver neighbourhood counts, not a distant silhouette.
            x, y = round(contact["point"][0] * width), round(contact["point"][1] * height)
            radius = round(job.data.get("contact_rendering", {}).get("search_radius_px", 40))
            local = np.zeros(shape, bool)
            local[
                max(0, y - radius) : min(height, y + radius + 1),
                max(0, x - radius) : min(width, x + radius + 1),
            ] = True
            receiver = support & local & ~actual & ~masks.arrays["fg_occ"]
            if not receiver.any():
                contacts.append(
                    {
                        "support_id": contact["support_id"],
                        "status": "UNKNOWN",
                        "reason": "承载边界被遮挡或候选覆盖，不能测量",
                    }
                )
                continue
            gap = float(distances[receiver].min())
            contacts.append(
                {
                    "support_id": contact["support_id"],
                    "gap_px": gap,
                    "status": "PASS"
                    if gap <= job.data["evaluation"]["contact_gap_max_px"]
                    else "FAIL",
                }
            )
        failed = any(c["status"] == "FAIL" for c in contacts)
        known = sum(c["status"] == "PASS" for c in contacts)
        status = (
            "FAIL"
            if failed
            else "PASS"
            if known >= item.get("spatial_constraints", {}).get("minimum_contact_count", 1)
            else "UNKNOWN"
        )
        checks[f"receiver_contact_{oid}"] = {
            "status": status,
            "detail": "实际估计边界到指定局部承载面的二维间隙；非三维载荷证明",
            "contacts": contacts,
        }
        objects[oid] = {"status": status, "contacts": contacts}
    return {"hard_checks": checks, "objects": objects, "physics_simulation_performed": False}


def render_contact(
    scene: np.ndarray,
    result: np.ndarray,
    job: ControlledJob,
    masks: MaskBundle,
    visible: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict]:
    settings = job.data.get("contact_rendering", {})
    if not settings.get("enabled", False):
        return result, {"status": "NOT_REQUESTED"}
    output = result.copy()
    records = []
    diagnostic = contact_diagnostics(job, masks, visible)
    height, width = scene.shape[:2]
    supports = {
        r["id"]: read_mask(job, r["mask"], (height, width))[0]
        for r in job.data["scene"]["support_regions"]
    }
    all_visible = (
        np.logical_or.reduce(list(visible.values())) if visible else np.zeros((height, width), bool)
    )
    for item in job.data["objects"]:
        oid = item["id"]
        actual = visible.get(oid)
        if actual is None:
            continue
        allowed = {
            index
            for index, c in enumerate(diagnostic["objects"].get(oid, {}).get("contacts", []))
            if c["status"] == "PASS"
        }
        distance = cv2.distanceTransform((~actual).astype(np.uint8), cv2.DIST_L2, 5)
        for contact_index, contact in enumerate(item["support_contacts"]):
            sid = contact["support_id"]
            if contact_index not in allowed or sid not in supports:
                continue
            x, y = np.asarray(contact["point"]) * [width, height]
            yy, xx = np.ogrid[:height, :width]
            local = (xx - x) ** 2 + (yy - y) ** 2 <= settings.get("search_radius_px", 40) ** 2
            receiver = (
                local
                & supports[sid]
                & masks.arrays["contact"]
                & ~all_visible
                & ~masks.arrays["protect"]
            )
            sigma = settings.get("sigma_px", 3)
            weight = np.exp(-0.5 * (distance / max(sigma, 0.1)) ** 2) * settings.get(
                "strength", 0.12
            )
            receiver &= distance <= 3 * sigma
            # Generated existing shadows reduce the added AO instead of being darkened twice.
            generated_darkening = np.maximum(
                0, 1 - result.mean(axis=2) / np.maximum(scene.mean(axis=2), 1)
            )
            weight = np.maximum(0, weight - generated_darkening)
            output[receiver] = np.rint(output[receiver] * (1 - weight[receiver][:, None])).astype(
                np.uint8
            )
            records.append(
                {
                    "object_id": oid,
                    "support_id": sid,
                    "pixels": int(receiver.sum()),
                    "maximum_added_darkening": float(weight[receiver].max())
                    if receiver.any()
                    else 0,
                }
            )
    return output, {
        "status": "APPLIED" if records else "SKIPPED_NO_MEASURED_CONTACT",
        "contacts": records,
        "claim": "仅在二维边界已接近的承载区补充有限接触遮蔽；不画桥接悬空间隙的黑影",
    }
