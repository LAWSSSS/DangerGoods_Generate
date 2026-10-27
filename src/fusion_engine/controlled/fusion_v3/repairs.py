"""Typed, candidate-bound defect proposals; no claimed native geometry lock."""

from __future__ import annotations

import copy
import math

import numpy as np
from jsonschema import Draft202012Validator

from fusion_engine.controlled.job import ControlledJob
from fusion_engine.controlled.masks import MaskBundle
from fusion_engine.errors import AdapterError
from fusion_engine.providers.vlm import REVIEW_SCHEMA

DEFECT_ACTIONS = {
    "identity_missing_part": (
        0,
        "guided_regeneration",
        "仅修复缺失部件及其连接，保持其它物体结构和旧化分布",
    ),
    "wrong_pose_scale": (
        0,
        "guided_regeneration",
        "重新贴合指定承载面与尺度，不以阴影补偿位置错误",
    ),
    "no_contact": (
        0,
        "guided_regeneration",
        "调整刚体实际接触，使实体底面相接；禁止用黑影填空气缝",
    ),
    "occlusion_inversion": (1, "review_required", "先核验原前景分割与交叉顺序，再提出局部重绘"),
    "mask_halo": (1, "review_required", "先复核实体/孔洞分割与边缘覆盖；不扩大羽化"),
    "background_changed": (1, "review_required", "先检查编辑授权和背景恢复，再处理候选"),
    "lighting_specular_mismatch": (
        2,
        "guided_regeneration",
        "按原场景重表现物体受光和粗糙反射，保留几何；不继承棚拍亮带",
    ),
    "color_cast": (2, "review_required", "先检查H0与分材质校色证据；不拿邻域钢材RGB均值填充物体"),
    "cast_shadow_missing": (2, "review_required", "先核实接收面和光向；缺少可靠证据时不画投影"),
    "camera_blur_mismatch": (
        3,
        "review_required",
        "先核对同深度结构的清晰度，不凭单一高频能量增加模糊",
    ),
    "unknown": (9, "review_required", "证据不足，保留原候选等待复核"),
}


def review_schema() -> dict:
    schema = copy.deepcopy(REVIEW_SCHEMA)
    schema["required"].append("defects")
    schema["properties"]["defects"] = {
        "type": "array",
        "maxItems": 16,
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["object_id", "kind", "status", "region", "evidence", "instruction"],
            "properties": {
                "object_id": {"type": "string"},
                "kind": {"enum": list(DEFECT_ACTIONS)},
                "status": {"enum": ["FAIL", "UNKNOWN"]},
                "region": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "evidence": {"type": "string", "minLength": 1},
                "instruction": {"type": "string", "minLength": 1},
            },
        },
    }
    return schema


def validate_defects(result: dict, job: ControlledJob) -> None:
    ids = {v["id"] for v in job.data["objects"]}
    for defect in result.get("defects", []):
        x0, y0, x1, y1 = defect["region"]
        if defect["object_id"] not in ids or x1 <= x0 or y1 <= y0:
            raise AdapterError("V3审阅缺陷的实例绑定或ROI非法；未猜测缺失坐标")


def review_response_schema() -> dict:
    """Validate the core review first; localization entries are checked individually."""
    schema = review_schema()
    schema["properties"]["defects"]["items"] = {}
    return schema


def partition_defects(result: dict, job: ControlledJob) -> tuple[dict, dict]:
    """Reject bad localizations without discarding the independent global review."""
    ids = {item["id"] for item in job.data["objects"]}
    validator = Draft202012Validator(review_schema()["properties"]["defects"]["items"])
    accepted, rejected = [], []
    for index, defect in enumerate(result.get("defects", [])):
        reasons = [
            {
                "field": ".".join(map(str, error.absolute_path)) or "entry",
                "validator": error.validator,
            }
            for error in validator.iter_errors(defect)
        ]
        if not reasons:
            if defect["object_id"] not in ids:
                reasons.append({"field": "object_id", "validator": "task_instance_binding"})
            x0, y0, x1, y1 = defect["region"]
            if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
                reasons.append({"field": "region", "validator": "finite_coordinates"})
            elif x1 <= x0 or y1 <= y0:
                reasons.append({"field": "region", "validator": "positive_rectangle"})
        if reasons:
            rejected.append(
                {
                    "index": index,
                    "status": "UNKNOWN_UNUSABLE_LOCALIZATION",
                    "reasons": reasons,
                    "model_entry": copy.deepcopy(defect),
                }
            )
        else:
            accepted.append(copy.deepcopy(defect))
    normalized = copy.deepcopy(result)
    normalized["defects"] = accepted
    if rejected:
        normalized["uncertainties"].append(
            f"程序发现{len(rejected)}条缺陷定位不符合坐标/实例契约；原条目单独留档，未推测或裁剪坐标。"
        )
        if normalized["recommendation"] == "accept":
            normalized["recommendation"] = "unknown"
    return normalized, {
        "status": "PARTIAL_LOCALIZATION_UNKNOWN" if rejected else "VALID",
        "valid_defects": len(accepted),
        "rejected_defects": rejected,
        "model_recommendation": result["recommendation"],
        "coordinates_clamped_or_ids_guessed": False,
    }


def repair_plan(
    job: ControlledJob,
    masks: MaskBundle,
    geometry: dict,
    processing: dict,
    opinion: dict | None,
    candidate_hash: str,
) -> dict:
    entries = []
    hard = geometry.get("hard_checks", {})
    failed = [name for name, check in hard.items() if check["status"] == "FAIL"]
    unknown = [name for name, check in hard.items() if check["status"] == "UNKNOWN"]
    if failed:
        entries.append(
            {
                "priority": 0,
                "kind": "geometry_contract",
                "action": "guided_regeneration",
                "instruction": "几何检查失败："
                + "、".join(failed)
                + "。重新落位并核对实体接触与完整连接，保持原图视野。",
            }
        )
    elif unknown or processing.get("status") != "APPLIED":
        entries.append(
            {
                "priority": 0,
                "kind": "unknown",
                "action": "review_required",
                "instruction": "分割/配准/处理证据不足，不能用重生掩盖未知；先复核缓存证据。",
            }
        )
    defects = (opinion or {}).get("result", {}).get("defects", [])
    review = (opinion or {}).get("result")
    if not review or review.get("recommendation") == "unknown":
        entries.append(
            {
                "priority": 4,
                "kind": "review_unavailable",
                "action": "review_required",
                "instruction": "最终视觉审阅缺失/UNKNOWN；无可定位证据，不能解释为没有缺陷。",
            }
        )
    elif review.get("recommendation") in ("refine", "reject") and not defects:
        entries.append(
            {
                "priority": 4,
                "kind": "unlocalized_visual_issue",
                "action": "review_required",
                "instruction": "辅助意见要求修复但缺少可定位类型/ROI，保留原始意见等待复核。",
            }
        )
    h, w = masks.arrays["edit"].shape
    for defect in defects:
        priority, action, instruction = DEFECT_ACTIONS[defect["kind"]]
        x0, y0, x1, y1 = defect["region"]
        rect = [
            int(np.floor(x0 * w)),
            int(np.floor(y0 * h)),
            min(w, int(np.ceil(x1 * w))),
            min(h, int(np.ceil(y1 * h))),
        ]
        region = masks.objects[defect["object_id"]].arrays["edit"][
            rect[1] : rect[3], rect[0] : rect[2]
        ]
        if defect["status"] != "FAIL" or not region.any():
            action = "review_required"
        entries.append(
            {
                "priority": priority,
                "kind": defect["kind"],
                "action": action,
                "object_id": defect["object_id"],
                "roi_original_pixels": rect,
                "evidence": defect["evidence"],
                "instruction": instruction + "；具体观察：" + defect["instruction"],
            }
        )
    entries.sort(key=lambda v: v["priority"])
    selected = entries[:2]
    # If the first problem needs evidence, automatic regeneration must not skip it.
    automatic = bool(selected) and selected[0]["action"] == "guided_regeneration"
    return {
        "input_candidate_sha256": candidate_hash,
        "status": "PROPOSED_GUIDED_REGENERATION" if automatic else "REVIEW_REQUIRED",
        "automatic_generation_allowed": automatic,
        "selected_defects": selected,
        "remaining_defects": entries[2:],
        "prompt_feedback": [
            v["instruction"] for v in selected if v["action"] == "guided_regeneration"
        ]
        if automatic
        else [],
        "native_mask_or_geometry_lock": False,
        "edit_scope": "existing_bbox_guidance_and_final_pixel_restore_not_native_inpaint",
        "allowed_property_changes": sorted({v["kind"] for v in selected}),
        "observation_is_ground_truth": False,
    }
