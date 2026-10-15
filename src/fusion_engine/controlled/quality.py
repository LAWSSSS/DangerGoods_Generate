"""Hard gates, independent observed masks and hash-bound human judgements."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from fusion_engine.controlled.geometry import distance_to_boundary, geometry_metrics, mask_iou
from fusion_engine.controlled.job import evidence_confirmed
from fusion_engine.controlled.masks import read_mask
from fusion_engine.controlled.planning import Prepared
from fusion_engine.evaluation import protected_delta
from fusion_engine.images import save_png

REPAIR_STAGES = {
    "geometry": ("S04/S05", "复核尺度参照、完整 OBB、姿态和独立完整轮廓；不要用调色修尺寸"),
    "support": ("S06", "复核承载区、支点和重心；需要时重新放置并更新接触区"),
    "occlusion": ("S07/S08", "修正局部前后关系，重建前景/可见/保护蒙版"),
    "identity": ("S02/S12", "对照多视图关键部件与连接，限定最小修复区域"),
    "material": ("S09", "按基材、涂层和暴露历史修旧化，禁止跨材质套红锈"),
    "lighting": ("S10/S12", "复核主光与接触/投影，材质响应先于局部亮度校正"),
    "style": ("S10/S13", "匹配同深度成像；局部处理并限制校正幅度"),
    "background_lock": ("S08/S14", "在最后一步恢复原前景、叠字及保护像素，重新保存并解码PNG"),
    "object_count": ("S02/S11", "逐个核对物体数量，避免把六视图当成六件"),
    "realism": ("S05/S06/S07/S10", "逐项检查边缘、接触、前景连续、光度和成像，定位具体根因"),
}


def check(status: str, detail: str, **measurements) -> dict:
    return {"status": status, "detail": detail, **measurements}


def evaluate(prepared: Prepared, master: np.ndarray, review: dict | None = None) -> dict:
    job, scene, masks = prepared.job, prepared.scene, prepared.masks
    checks, observed, objects = {}, {}, {}
    mode = job.data["mode"]
    if scene.shape != master.shape:
        return {
            "status": "REJECTED",
            "hard_checks": {"canvas": check("FAIL", "尺寸/坐标画布不一致")},
            "objects": {},
            "repairs": [],
            "annotation_ready": False,
        }
    protect = masks.arrays["protect"]
    delta = protected_delta(scene, master, protect)
    checks["background_lock"] = check(
        "NOT_APPLICABLE"
        if mode == "A"
        else (
            "UNKNOWN"
            if not delta["applicable"]
            else "PASS"
            if delta["max_channel_delta"] == 0
            else "FAIL"
        ),
        "以最终已保存、重新解码的 RGB8 PNG 逐像素检查保护范围",
        measurements=delta,
    )
    if mode != "A":
        for index, issue in enumerate(prepared.plan["issues"]):
            checks[f"input_{index}_{issue['code']}"] = check(issue["status"], issue["detail"])
    supplied = {item["id"]: item for item in (review or {}).get("objects", [])}
    configured = {item["id"]: item for item in job.data["objects"]}
    if set(supplied) - set(configured):
        checks["unexpected_observed_object"] = check("FAIL", "验收报告出现任务外的物体 ID")
    settings = job.data["evaluation"]
    for oid, item in configured.items():
        plan_masks = masks.objects[oid].arrays
        selected = plan_masks["obj_visible"]
        changed = np.any(master != scene, axis=2)
        fraction = float(changed[selected].mean()) if selected.any() else None
        checks[f"change_{oid}"] = check(
            "UNKNOWN"
            if fraction is None
            else "PASS"
            if fraction > 0 and fraction >= settings["min_changed_visible_fraction"]
            else "FAIL",
            "只证明可见区域有改动，不证明物品正确",
            fraction=fraction,
        )
        record = supplied.get(oid, {})
        objects[oid] = {
            "geometry": {"status": "UNKNOWN", "reason": "缺少绑定当前母版的独立完整轮廓"}
        }
        observed[oid] = {}
        for name in ("amodal_mask", "visible_mask"):
            descriptor = record.get(name)
            if (
                descriptor
                and record.get("mask_matches_master_confirmed")
                and evidence_confirmed(descriptor["evidence"])
            ):
                value, _ = read_mask(job, descriptor, scene.shape[:2])
                observed[oid][name] = value
        full, visible = observed[oid].get("amodal_mask"), observed[oid].get("visible_mask")
        checks[f"observed_nonempty_{oid}"] = check(
            "UNKNOWN"
            if full is None or visible is None
            else "PASS"
            if full.any() and visible.any()
            else "FAIL",
            "独立实际完整轮廓和可见轮廓都必须非空；阈值不能让空标注通过",
        )
        if full is not None and full.any() and not item["placement"].get("allow_crop"):
            metrics = geometry_metrics(full, item["placement"])
            objects[oid]["geometry"] = metrics
            valid = (
                metrics.get("status") == "COMPUTED"
                and metrics["center_error"] <= settings["center_error_max"]
                and metrics["scale_log_error"] <= settings["scale_log_error_max"]
                and metrics["rotation_error_deg"] <= settings.get("rotation_error_max_deg", 10)
            )
            checks[f"geometry_{oid}"] = check(
                "PASS" if valid else "FAIL",
                "使用独立完整轮廓拟合 OBB；不使用可见 bbox",
                measurements=metrics,
            )
        else:
            checks[f"geometry_{oid}"] = check(
                "UNKNOWN", "缺少独立非空完整轮廓，或裁切后无法验证完整尺度"
            )
        for name, actual, target in (
            ("amodal", full, plan_masks["obj_full"]),
            ("visible", visible, selected),
        ):
            score = mask_iou(actual, target) if actual is not None else None
            checks[f"mask_{name}_{oid}"] = check(
                "UNKNOWN"
                if score is None
                else "PASS"
                if score >= settings["mask_iou_min"]
                else "FAIL",
                "实际审核轮廓与计划轮廓的 IoU；计划本身不充当实际标注",
                iou=score,
            )
        if full is not None and visible is not None:
            consistent = not np.any(visible & ~full) and not np.any(visible & protect)
            checks[f"mask_consistency_{oid}"] = check(
                "PASS" if consistent else "FAIL",
                "独立可见轮廓必须属于完整轮廓，且不得覆盖保护/原前景像素",
            )
    if job.data["constraints"]["require_support"]:
        support_regions = {
            region["id"]: read_mask(job, region["mask"], scene.shape[:2])[0]
            for region in job.data["scene"]["support_regions"]
        }
        for oid, item in configured.items():
            full = observed[oid].get("amodal_mask")
            gaps = [
                distance_to_boundary(full, contact["point"]) if full is not None else None
                for contact in item["support_contacts"]
            ]
            support_gaps = []
            for contact in item["support_contacts"]:
                sid = contact["support_id"]
                if sid in configured:
                    support = observed[sid].get("amodal_mask")
                    support_gaps.append(
                        distance_to_boundary(support, contact["point"])
                        if support is not None
                        else None
                    )
                else:
                    support = support_regions[sid]
                    x = min(scene.shape[1] - 1, round(contact["point"][0] * scene.shape[1]))
                    y = min(scene.shape[0] - 1, round(contact["point"][1] * scene.shape[0]))
                    distances = cv2.distanceTransform((~support).astype(np.uint8), cv2.DIST_L2, 5)
                    support_gaps.append(float(distances[y, x]))
            known = bool(gaps) and all(gap is not None for gap in gaps + support_gaps)
            checks[f"contact_{oid}"] = check(
                "UNKNOWN"
                if not known
                else "PASS"
                if max(gaps + support_gaps) <= settings["contact_gap_max_px"]
                else "FAIL",
                "测量声明支点到独立实际完整轮廓的2D边界距离；不等于力学证明",
                gaps_px=gaps,
                actual_support_gaps_px=support_gaps,
            )
    visible_masks = [
        value["visible_mask"] for value in observed.values() if "visible_mask" in value
    ]
    if len(visible_masks) > 1:
        checks["visible_exclusivity"] = check(
            "PASS" if np.max(np.sum(visible_masks, axis=0)) <= 1 else "FAIL",
            "不同不透明物体的实际可见标注不能占据同一像素",
        )
    for name in settings["required_visual_checks"]:
        judgement = (review or {}).get("checks", {}).get(name)
        checks[name] = check(
            judgement["status"] if judgement else "UNKNOWN",
            judgement["notes"] if judgement else "等待针对当前母版的显式人工验收；VLM仅供参考",
            source="hash_bound_human_review" if judgement else "not_observed",
        )
    if mode == "C":
        for oid in configured:
            complete = len(observed[oid]) == 2 and all(
                value.any() for value in observed[oid].values()
            )
            checks[f"annotation_{oid}"] = check(
                "PASS" if complete else "UNKNOWN", "C 模式要求两类实际且已审核的标注轮廓"
            )
    states = [item["status"] for item in checks.values()]
    status = (
        "CONCEPT_ONLY"
        if mode == "A"
        else "REJECTED"
        if "FAIL" in states
        else "REVIEW_REQUIRED"
        if "UNKNOWN" in states
        else "ACCEPTED"
    )
    repairs = []
    for name, item in checks.items():
        if item["status"] not in ("FAIL", "UNKNOWN"):
            continue
        category = next((key for key in REPAIR_STAGES if key in name.lower()), "geometry")
        if any(key in name.lower() for key in ("contact", "stability", "com_outside")):
            category = "support"
        elif "overlap" in name.lower() or "visible" in name.lower():
            category = "occlusion"
        stage, action = REPAIR_STAGES[category]
        repairs.append(
            {
                "check": name,
                "status": item["status"],
                "return_to_stage": stage,
                "action": action,
                "evidence_needed": item["detail"],
            }
        )
    return {
        "status": status,
        "hard_checks": checks,
        "objects": objects,
        "repairs": repairs,
        "annotation_ready": status == "ACCEPTED"
        and all(
            len(value) == 2 and all(mask.any() for mask in value.values())
            for value in observed.values()
        ),
        "vlm_can_override_hard_failure": False,
        "weighted_average_can_override_hard_failure": False,
        "method": "deterministic_metrics_plus_hash_bound_human_evidence",
    }


def uncompressed_rle(mask: np.ndarray) -> dict:
    pixels = mask.flatten(order="F").astype(np.uint8)
    padded = np.r_[0, pixels, 0]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    runs, position, value = [], 0, 0
    for index in changes:
        runs.append(int(index - position))
        position, value = int(index), 1 - value
    if position < len(pixels) or not runs:
        runs.append(int(len(pixels) - position))
    elif value == 1:
        runs.append(0)
    return {"size": list(mask.shape), "counts": runs}


def export_annotations(prepared: Prepared, review: dict, destination: Path) -> dict:
    annotations, categories = [], []
    category_ids = {}
    height, width = prepared.scene.shape[:2]
    records = {item["id"]: item for item in review["objects"]}
    for index, item in enumerate(prepared.job.data["objects"], 1):
        oid = item["id"]
        if item["category"] not in category_ids:
            category_id = len(category_ids) + 1
            category_ids[item["category"]] = category_id
            categories.append({"id": category_id, "name": item["category"]})
        actual = {}
        for name in ("amodal_mask", "visible_mask"):
            mask, _ = read_mask(prepared.job, records[oid][name], (height, width))
            actual[name] = mask
            save_png(
                mask.astype(np.uint8) * 255, destination / "observed_masks" / f"{oid}_{name}.png"
            )
        visible = actual["visible_mask"]
        y, x = np.nonzero(visible)
        annotations.append(
            {
                "id": index,
                "image_id": 1,
                "category_id": category_ids[item["category"]],
                "iscrowd": 0,
                "segmentation": uncompressed_rle(visible),
                "area": int(visible.sum()),
                "bbox": [
                    int(x.min()),
                    int(y.min()),
                    int(x.max() - x.min() + 1),
                    int(y.max() - y.min() + 1),
                ],
                "object_instance_id": oid,
                "amodal_segmentation": uncompressed_rle(actual["amodal_mask"]),
                "annotation_source": "independent_hash_bound_human_reviewed_masks",
                "occlusion_ratio": 1.0 - float(visible.sum() / actual["amodal_mask"].sum()),
                "truncation": bool(item["placement"].get("allow_crop", False)),
                "placement_uncertainty": item["placement"]["scale"]["evidence"],
            }
        )
    return {
        "images": [
            {
                "id": 1,
                "file_name": "../outputs/final.png",
                "width": width,
                "height": height,
                "synthetic": True,
                "source_scene_id": prepared.plan["background"]["sha256"],
                "source_object_ids": [item["id"] for item in prepared.job.data["objects"]],
                "review_status": "ACCEPTED",
            }
        ],
        "categories": categories,
        "annotations": annotations,
        "format": "COCO-compatible uncompressed RLE; amodal_segmentation is an extension",
        "reviewed_master_sha256": review["master_sha256"],
        "reviewer": review["reviewer"],
    }
