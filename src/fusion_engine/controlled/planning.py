"""Joint placement, support and material planning with evidence-aware readiness."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from fusion_engine.controlled.appearance import material_plan, roi_statistics
from fusion_engine.controlled.framing import request_frame
from fusion_engine.controlled.geometry import distance_to_boundary, obb_vertices, scale_plan
from fusion_engine.controlled.job import ControlledJob, asset_record, evidence_confirmed
from fusion_engine.controlled.masks import MaskBundle, build_masks, issue, read_mask
from fusion_engine.controlled.photometry import photometric_plan
from fusion_engine.controlled.prompting import build_prompt_document
from fusion_engine.errors import ConfigError
from fusion_engine.images import load_rgb

STAGES = [
    ("S00", "需求与模式"),
    ("S01", "资产登记"),
    ("S02", "身份/部件树"),
    ("S03", "场景解析"),
    ("S04", "尺度参照"),
    ("S05", "投影OBB/姿态"),
    ("S06", "支撑接触"),
    ("S07", "局部遮挡"),
    ("S08", "六类Mask"),
    ("S09", "材质旧化计划"),
    ("S10", "光照/成像约束"),
    ("S11", "生成"),
    ("S12", "局部重绘/修复"),
    ("S13", "局部调色/成像匹配"),
    ("S14", "背景恢复"),
    ("S15", "硬门禁/人工验收"),
    ("S16", "导出/追溯"),
]


@dataclass
class Prepared:
    job: ControlledJob
    scene: np.ndarray
    masks: MaskBundle
    plan: dict
    prompt: str


def _support_plan(job: ControlledJob, masks: MaskBundle) -> tuple[dict, list[dict]]:
    constraints = job.data["constraints"]
    shape = masks.arrays["edit"].shape
    height, width = shape
    regions, region_data, issues, result = {}, {}, [], {}
    for region in job.data["scene"]["support_regions"]:
        regions[region["id"]], _ = read_mask(job, region["mask"], shape)
        region_data[region["id"]] = region
    for oid, value in masks.objects.items():
        regions[oid] = value.arrays["obj_full"]
    graph = {
        item["id"]: {contact["support_id"] for contact in item["support_contacts"]}
        for item in job.data["objects"]
    }
    anchors = {
        sid
        for sid, region in region_data.items()
        if region["load_bearing"] == "confirmed" and evidence_confirmed(region["mask"]["evidence"])
    }

    anchored_ids = set(anchors)
    for _ in range(len(graph)):
        old_size = len(anchored_ids)
        anchored_ids.update(oid for oid, supports in graph.items() if supports & anchored_ids)
        if len(anchored_ids) == old_size:
            break

    for item in job.data["objects"]:
        oid, contacts = item["id"], []
        if constraints["require_support"] and oid not in anchored_ids:
            issues.append(
                issue(
                    "UNANCHORED_SUPPORT_GRAPH",
                    "UNKNOWN",
                    "支撑链没有连接到已核验的场景承载区；互相支撑不能自动证明不悬浮",
                    oid,
                )
            )
        if constraints["require_support"] and not item["support_contacts"]:
            issues.append(
                issue("MISSING_SUPPORT", "UNKNOWN", "未提供承载支点，未推断悬浮为接触", oid)
            )
        for contact in item["support_contacts"]:
            support = regions[contact["support_id"]]
            x = min(width - 1, round(contact["point"][0] * width))
            y = min(height - 1, round(contact["point"][1] * height))
            gap = distance_to_boundary(masks.objects[oid].arrays["obj_full"], contact["point"])
            tolerance = job.data["evaluation"]["contact_gap_max_px"]
            distance = cv2.distanceTransform((~support).astype(np.uint8), cv2.DIST_L2, 5)
            valid = gap is not None and gap <= tolerance and distance[y, x] <= tolerance
            contacts.append(
                {
                    **contact,
                    "planned_boundary_gap_px": gap,
                    "support_region_gap_px": float(distance[y, x]),
                    "image_space_contact_check": "PASS" if valid else "FAIL",
                }
            )
            if not valid:
                issues.append(issue("SUPPORT_GAP", "FAIL", "计划支点未接近物体轮廓或承载区", oid))
            support_region = region_data.get(contact["support_id"])
            if support_region:
                load = support_region["load_bearing"]
                if load != "confirmed" or not evidence_confirmed(
                    support_region["mask"]["evidence"]
                ):
                    issues.append(
                        issue(
                            "SUPPORT_CAPACITY",
                            "FAIL" if load == "unsafe" else "UNKNOWN",
                            "承载区的可承重状态没有可靠证据",
                            oid,
                        )
                    )
            if not evidence_confirmed(contact["evidence"]):
                issues.append(issue("UNCONFIRMED_CONTACT", "UNKNOWN", "支撑接触尚未审核", oid))
        if constraints["require_stability_evidence"] and not evidence_confirmed(
            item.get("stability_evidence")
        ):
            issues.append(
                issue("STABILITY_UNKNOWN", "UNKNOWN", "缺少稳定性证据，2D接近不等于3D可承重", oid)
            )
        com_check = {"status": "UNKNOWN", "reason": "未提供可形成平面支撑包络的支点/重心投影"}
        if len(contacts) >= 3 and item.get("center_of_mass_projection"):
            points = np.array([contact["point"] for contact in contacts], np.float32) * [
                width,
                height,
            ]
            hull = cv2.convexHull(points.astype(np.float32))
            if cv2.contourArea(hull) > 0:
                center = tuple(np.array(item["center_of_mass_projection"]) * [width, height])
                inside = cv2.pointPolygonTest(hull, center, False) >= 0
                com_check = {
                    "status": "PASS" if inside else "FAIL",
                    "scope": "仅图像平面必要条件，不是三维力学稳定性证明",
                }
                if not inside:
                    issues.append(
                        issue("COM_OUTSIDE_SUPPORT", "FAIL", "声明的重心投影超出支撑包络", oid)
                    )
        result[oid] = {
            "anchored_to_confirmed_scene_region": oid in anchored_ids,
            "contacts": contacts,
            "com_projection": com_check,
            "stability_evidence": item.get("stability_evidence"),
            "physics_simulation_performed": False,
        }
    return result, issues


def build_prompt(job: ControlledJob, objects: list[dict]) -> str:
    """Render the same formatted text used by the prompt-only CLI and adapters."""
    return build_prompt_document(job.data, objects)["text"]


def prepare(job: ControlledJob) -> Prepared:
    background = asset_record(job, job.data["background"], image=True)
    scene = load_rgb(job.asset(job.data["background"]["path"]))
    height, width = scene.shape[:2]
    for key, observed in (("width", width), ("height", height)):
        if key in job.data["background"] and job.data["background"][key] != observed:
            raise ConfigError("声明背景尺寸与 EXIF 校正后的原始坐标画布不一致")
    masks = build_masks(job, (height, width))
    v3_plan = None
    if job.data.get("feature_flags", {}).get("v3_layer_compositor"):
        from fusion_engine.controlled.fusion_v3.renderer import validate_plan

        v3_plan = validate_plan(job, masks)
    issues, objects = list(masks.diagnostics), []
    for item in job.data["objects"]:
        oid, placement = item["id"], item["placement"]
        references = [asset_record(job, reference, image=True) for reference in item["references"]]
        corners = obb_vertices(placement, width, height)
        if not placement.get("allow_crop", False) and (
            np.any(corners < 0) or np.any(corners > [width, height])
        ):
            issues.append(
                issue("OBB_OUTSIDE_CANVAS", "FAIL", "旋转后的完整 OBB 越出画布，且未允许裁切", oid)
            )
        if placement["status"] != "confirmed" or not evidence_confirmed(placement["evidence"]):
            issues.append(
                issue("PLACEMENT_UNCONFIRMED", "UNKNOWN", "投影摆放与完整尺度目标尚未确认", oid)
            )
        scale = scale_plan(placement["scale"])
        if scale["status"] == "UNKNOWN":
            issues.append(
                issue("SCALE_EVIDENCE_MISSING", "UNKNOWN", "缺少所选尺度等级的必要输入", oid)
            )
        interval = scale.get("pixel_length_range")
        planned_length = max(placement["width"] * width, placement["height"] * height)
        if interval and not interval[0] <= planned_length <= interval[1]:
            issues.append(
                issue(
                    "SCALE_PLAN_CONFLICT",
                    "FAIL",
                    "完整投影 OBB 的最长边不在声明的投影长度区间",
                    oid,
                )
            )
        if item["render_class"] != "opaque_rigid":
            issues.append(
                issue(
                    "SPECIAL_RENDERER_REQUIRED",
                    "UNKNOWN",
                    "透明/软体需要透射或形变适配器，当前尚未接入",
                    oid,
                )
            )
        objects.append(
            {
                "id": oid,
                "dimensions": item["dimensions"],
                "references": references,
                "placement": placement,
                "obb_vertices_pixels": corners.tolist(),
                "scale": scale,
                "material": material_plan(item),
                "mask_sources": masks.objects[oid].sources,
                "visible_pixels": int(masks.objects[oid].arrays["obj_visible"].sum()),
                "full_projection_pixels_in_canvas": int(
                    masks.objects[oid].arrays["obj_full"].sum()
                ),
                "visible_is_not_full_projection": True,
            }
        )
    support, support_issues = _support_plan(job, masks)
    issues += support_issues
    photometry = photometric_plan(job, scene, masks)
    frame = request_frame(job, masks, scene.shape[:2])
    concept = job.data["mode"] == "A"
    blocking = [value for value in issues if value["status"] in ("FAIL", "UNKNOWN")]
    # Concept A can proceed with missing physical evidence, but cannot be accepted as B/C.
    spatial_fail = job.data["constraints"].get("enforce_spatial_plan", False) and any(
        value["status"] == "FAIL" for value in issues
    )
    ready = (not blocking or concept) and not spatial_fail
    plan = {
        "task_id": job.data["task_id"],
        "schema_version": "2.0",
        "pipeline_version": job.data.get("pipeline_version", "2.0"),
        "v3_shadow_authorization": v3_plan,
        "mode": job.data["mode"],
        "readiness": "READY" if ready else "INPUT_REQUIRED",
        "background": background,
        "scene": {
            "configured_hypotheses": job.data["scene"],
            "editable_roi_statistics": roi_statistics(scene, masks.arrays["edit"]),
            "depth_estimation": "not_performed",
            "camera_calibration": "not_inferred",
            "lighting_estimation": "configured_hypothesis_not_measurement",
        },
        "objects": objects,
        "support": support,
        "photometry": photometry,
        "request_frame": frame,
        "occlusion": masks.depth_order,
        "mask_counts": {name: int(mask.sum()) for name, mask in masks.arrays.items()},
        "mask_sources": masks.sources,
        "issues": issues,
        "stages": [{"id": sid, "name": name, "status": "PLANNED"} for sid, name in STAGES],
        "model_internal_steps": "not_exposed",
        "model_calls_performed": 0,
        "limitations": [
            "OBB 不是分割轮廓",
            "单目相对深度不等于米",
            "接触像素接近不是3D稳定性证明",
            "VLM意见不能替代独立几何证据与人工验收",
            "未执行三维重建/真实物理渲染",
        ],
    }
    document = build_prompt_document(job.data, objects, photometry=photometry, frame=frame)
    plan["prompt_document"] = document
    return Prepared(job, scene, masks, plan, document["text"])
