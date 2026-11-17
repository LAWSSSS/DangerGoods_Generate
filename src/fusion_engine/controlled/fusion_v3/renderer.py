"""Connect observed entities, bounded H0/H1 appearance and independent interactions."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from fusion_engine.artifacts import sha256
from fusion_engine.controlled.appearance import prepare_foreground, roi_statistics
from fusion_engine.controlled.fusion_v3.compositor import Layers, binary_partition, compose_layers
from fusion_engine.controlled.fusion_v3.interactions import (
    contact_layer,
    empty_interaction,
    extract_shadow,
)
from fusion_engine.controlled.job import ControlledJob, evidence_confirmed
from fusion_engine.controlled.masks import MaskBundle, read_mask
from fusion_engine.controlled.photometry import encode_rgb, linear_rgb
from fusion_engine.errors import ConfigError
from fusion_engine.images import save_png


def validate_plan(job: ControlledJob, masks: MaskBundle) -> list[dict]:
    """Validate authorization before requesting a model; masks describe scopes, not shadows."""
    settings = job.data.get("fusion_v3", {})
    records = []
    ids = {item["id"] for item in job.data["objects"]}
    for value in settings.get("shadow_receivers", []):
        if value["object_id"] not in ids:
            raise ConfigError("V3 阴影区域引用了未知物体")
        receiver, receiver_source = read_mask(job, value["receiver"], masks.arrays["edit"].shape)
        allowed, allowed_source = read_mask(job, value["allowed"], receiver.shape)
        for key in ("receiver", "allowed"):
            if not evidence_confirmed(value[key]["evidence"]) and not (
                job.data["mode"] == "A" and settings.get("allow_proposed_interactions")
            ):
                raise ConfigError("V3 交互区域需审核证据；A预览需显式允许建议区")
        if not receiver.any() or not allowed.any():
            raise ConfigError("V3 声明的阴影接收区与授权区不能是空 Mask")
        if np.any(allowed & (~masks.arrays["edit"] | masks.arrays["protect"])):
            raise ConfigError("V3 阴影授权区越出 edit 或进入 protect；必须修订范围")
        records.append(
            {
                "id": value["id"],
                "object_id": value["object_id"],
                "receiver": receiver_source,
                "allowed": allowed_source,
                "surface_kind": value["surface_kind"],
                "actual_shadow_evidence": "not_yet_estimated",
            }
        )
    if len({v["id"] for v in records}) != len(records):
        raise ConfigError("V3 阴影区域 ID 不可重复")
    return records


def drift_normalized_candidate(
    candidate: np.ndarray, operations: list[dict]
) -> tuple[np.ndarray, dict]:
    """Cancel only a reliable, unbounded-by-limits H0 background fit for shadow analysis.

    Full fitted gain is used here, independent of the appearance blend strength.
    This temporary image is never copied into the original background.
    """
    paired = next(
        (v for v in operations if v["operation"] == "paired_background_local_linear_RGB"), {}
    )
    anchors = paired.get("anchors", [])
    if not anchors or any(v["bounded"] or "center_normalized" not in v for v in anchors):
        return candidate, {"status": "UNKNOWN", "reason": "缺少未触及上限的可靠 H0 配对拟合"}
    h, w = candidate.shape[:2]
    gh, gw = min(h, 96), min(w, 96)
    yy, xx = np.mgrid[:gh, :gw].astype(np.float32)
    points = np.stack(((xx + 0.5) / gw, (yy + 0.5) / gh), axis=-1)
    centers = np.array([v["center_normalized"] for v in anchors], np.float32)
    distance = np.sum((points[..., None, :] - centers) ** 2, axis=-1)
    weights = 1 / np.maximum(distance, 0.015**2)
    weights /= weights.sum(axis=-1, keepdims=True)
    grid = weights @ np.array([v["requested_log2_rgb"] for v in anchors], np.float32)
    gain = cv2.resize(grid, (w, h), interpolation=cv2.INTER_LINEAR)
    result = encode_rgb(linear_rgb(candidate) * 2**gain)
    return result, {
        "status": "ESTIMATED",
        "method": "full_reliable_H0_fit_for_shadow_analysis_only",
        "anchor_ids": [v["id"] for v in anchors],
        "background_pixels_copied": False,
    }


def render(
    scene: np.ndarray,
    candidate: np.ndarray,
    job: ControlledJob,
    masks: MaskBundle,
    visible: dict[str, np.ndarray] | None,
    parts: dict[str, np.ndarray] | None,
    *,
    registration: dict | None,
    diagnostics_dir: Path | None = None,
) -> tuple[np.ndarray, dict]:
    if registration is None or registration.get("status") != "REGISTERED":
        raise ConfigError("V3_REGISTRATION_REQUIRED：配准非 REGISTERED，停止坐标依赖处理")
    if visible is None or set(visible) != {v["id"] for v in job.data["objects"]}:
        raise ConfigError("V3 需要每件物体的候选可见轮廓；计划 OBB/Mask 不可替代")
    if any(v["render_class"] != "opaque_rigid" for v in job.data["objects"]):
        raise ConfigError("V3 当前只实现不透明刚体；透明和软体需要专用合成器")
    for oid, value in visible.items():
        if value.shape != scene.shape[:2] or value.dtype != np.bool_ or not value.any():
            raise ConfigError("V3 候选轮廓必须为非空、原图尺寸二值证据")
        if np.any(value & (~masks.objects[oid].arrays["edit"] | masks.arrays["protect"])):
            raise ConfigError("V3 候选实体越界/进入保护区；禁止裁断后继续合成")
    object_mask = np.logical_or.reduce(list(visible.values()))
    if sum(int(v.sum()) for v in visible.values()) != int(object_mask.sum()):
        raise ConfigError("V3 实例可见 Mask 重叠；先解析前后关系，不静默覆盖")
    plan_records = validate_plan(job, masks)
    foreground, target, operations = prepare_foreground(
        scene, candidate, job, masks, visible, parts, skip_unreliable_photometry=True
    )
    core, edge = binary_partition(object_mask)
    flags = job.data.get("feature_flags", {})
    settings = job.data.get("fusion_v3", {})
    shape = scene.shape[:2]
    cast = empty_interaction(shape, "NOT_REQUESTED", "投射阴影图层未启用")
    shadow_records = []
    drift = {"status": "NOT_REQUESTED"}
    if flags.get("v3_interaction_layers"):
        cast = empty_interaction(shape, "UNKNOWN", "未提供独立阴影接收面，不虚构投影")
        normalized, drift = drift_normalized_candidate(candidate, operations)
        for scope in settings.get("shadow_receivers", []):
            if scope["surface_kind"] != "locally_planar":
                shadow_records.append(
                    {
                        "id": scope["id"],
                        "status": "UNKNOWN",
                        "reason": "复杂交叉/未知接收面需要独立几何证据",
                    }
                )
                continue
            receiver, _ = read_mask(job, scope["receiver"], shape)
            allowed, _ = read_mask(job, scope["allowed"], shape)
            # Plan validation has checked authorization. Now remove hidden pixels.
            allowed &= ~masks.arrays["fg_occ"] & ~object_mask
            estimate = extract_shadow(
                scene,
                normalized,
                receiver,
                allowed,
                object_mask,
                settings.get("shadow", {}),
                registration_trusted=True,
                drift_trusted=drift["status"] == "ESTIMATED",
            )
            cast.transmission = np.minimum(cast.transmission, estimate.transmission)
            cast.receiver |= estimate.receiver
            shadow_records.append(
                {
                    "id": scope["id"],
                    "object_id": scope["object_id"],
                    **estimate.report,
                    "scope_evidence_confirmed": evidence_confirmed(scope["receiver"]["evidence"]),
                }
            )
        cast.report = {
            "status": "ESTIMATED" if cast.receiver.any() else "UNKNOWN",
            "receivers": shadow_records,
            "drift_normalization": drift,
        }
    contact = contact_layer(job, masks, visible)
    layers = Layers(
        object_mask,
        core,
        edge,
        masks.arrays["fg_occ"],
        contact.receiver,
        cast.receiver,
        masks.arrays["edit"],
        masks.arrays["protect"],
    )
    result, compositing = compose_layers(
        scene,
        foreground,
        layers,
        cast_transmission=cast.transmission,
        contact_transmission=contact.transmission,
    )
    operations += [
        {"operation": "v3_candidate_shadow_transport", **cast.report},
        {"operation": "v3_contact_occlusion_linear", **contact.report},
        compositing,
    ]
    diagnostic = None
    if diagnostics_dir is not None and flags.get("v3_diagnostics"):
        from fusion_engine.controlled.fusion_v3.diagnostics import save_diagnostics

        diagnostic = save_diagnostics(
            diagnostics_dir, scene, candidate, foreground, result, layers, cast, contact
        )
    authorized_record = None
    if diagnostics_dir is not None:
        authorized_path = diagnostics_dir.parent / (diagnostics_dir.name + ".authorized.png")
        save_png(layers.authorized.astype(np.uint8) * 255, authorized_path)
        authorized_record = {"path": str(authorized_path), "sha256": sha256(authorized_path)}
    return result, {
        "pipeline_version": "3.0",
        "renderer": "strict_linear_layers",
        "operations": operations,
        "shadow_plan": plan_records,
        "diagnostics": diagnostic,
        "authorized_mask": authorized_record,
        "render_scope": "observed_opaque_object_and_independent_authorized_interactions",
        "global_case_grading_used": False,
        "object_stats": roi_statistics(result, target),
        "visual_acceptance": "REVIEW_REQUIRED",
        "mask_evidence_level": "M2_estimated_not_M3_reviewed",
        "alpha_uncertainty_used_as_opacity": False,
        "edge_decontamination": "NOT_PERFORMED_NO_VERIFIED_TRIMAP_FG_BG",
        "foreground_restoration_present": bool(layers.front.any()),
        "interaction_effects": {
            "contact_pixels": int(contact.receiver.sum()),
            "contact_status": contact.report["status"],
            "cast_pixels": int(cast.receiver.sum()),
            "cast_status": cast.report["status"],
            "foreground_pixels": int(layers.front.sum()),
            "zero_pixels_do_not_prove_interactions_unnecessary": True,
        },
        "relighting": "NOT_PERFORMED",
        "limitations": [
            "不修复已经生成的错位高光、缺失部件或三维穿模",
            "二值估计边界可能有锯齿/细杆误分；未声称真实亚像素matting",
            "阴影是保守二维估计，UNKNOWN不表示不需要投影",
        ],
    }
