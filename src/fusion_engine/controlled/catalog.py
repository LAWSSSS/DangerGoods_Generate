"""Migrate the existing per-object catalog without upgrading estimates into facts."""

from __future__ import annotations

from pathlib import Path

from fusion_engine.artifacts import sha256
from fusion_engine.controlled.job import read_json
from fusion_engine.controlled.schema import CHECK_NAMES, MATERIALS
from fusion_engine.errors import ConfigError
from fusion_engine.images import image_info


def draft_evidence(notes: str) -> dict:
    return {"source": "proposal", "confirmed": False, "notes": notes, "confidence": 0.2}


def from_catalog(
    workspace: Path,
    catalog_path: Path,
    background: Path,
    ids: list[str],
    reference_overrides: dict[str, Path],
) -> dict:
    catalog = read_json(catalog_path)
    lookup = {item["id"]: item for item in catalog["objects"]}
    if not ids or len(set(ids)) != len(ids) or set(ids) - set(lookup):
        raise ConfigError("物品 ID 为空、重复或不在目录内")
    objects = []
    background_info = image_info(background)
    for index, oid in enumerate(ids):
        item = lookup[oid]
        reference = reference_overrides.get(oid)
        if reference is None:
            root = Path(catalog["reference_root"]).expanduser()
            if not root.is_absolute():
                root = catalog_path.resolve().parent.parent / root
            reference = root / item["reference_filename"]
        if not reference.is_file():
            raise ConfigError(f"参考文件不存在：{reference.name}")
        material = item.get("material", "unknown")
        if material not in MATERIALS:
            material = "unknown"
        # This is an editable image-space proposal, never based on the legacy 2.4m assumption.
        ratio = item.get("size_policy", {}).get("cross_mm", 1) / max(
            item.get("size_policy", {}).get("reference_length_mm", 1), 1
        )
        width = min(0.26, 0.65 / max(len(ids), 1))
        height = min(
            0.3, max(0.025, width * background_info["width"] / background_info["height"] * ratio)
        )
        objects.append(
            {
                "id": oid,
                "category": item["name"],
                "references": [
                    {
                        "path": str(reference.resolve()),
                        "view_id": "atlas",
                        "sha256": sha256(reference),
                        "source_kind": "rusted_atlas" if oid in reference_overrides else "atlas",
                        "orientation_hint": "六视图仅表示同一物体；不作为正交三维测量",
                        "complete": True,
                    }
                ],
                "dimensions": {
                    "status": "estimated",
                    "units": "mm",
                    "values": item.get("dimensions_mm", {}),
                    "ranges": item.get("dimension_ranges_mm", {}),
                    "evidence": {
                        "source": "estimate",
                        "confirmed": False,
                        "notes": item.get("dimension_evidence", {}).get(
                            "note", "既有工作值未经实测"
                        ),
                    },
                },
                "identity": {
                    "level": "instance",
                    "critical_parts": item.get("identity_features", []),
                    "positive": item.get("identity_features", []),
                    "negative": item.get("forbidden_changes", []),
                },
                "material_parts": [
                    {
                        "id": "body",
                        "base_material": material,
                        "evidence": draft_evidence(
                            "从旧目录迁移的材质提示；需补充橡胶/玻璃/塑料等独立部件"
                        ),
                    }
                ],
                "ageing": {
                    "level": 3,
                    "exposure_history": "场景中长期弃置；具体暴露史待确认",
                    "allow_structural_damage": False,
                },
                "placement": {
                    "box_type": "oriented",
                    "coordinate_space": "normalized_original_background",
                    "center": [(index + 1) / (len(ids) + 1), 0.56],
                    "width": round(width, 5),
                    "height": round(height, 5),
                    "rotation_deg": -12,
                    "allow_crop": False,
                    "status": "proposed",
                    "evidence": draft_evidence(
                        "仅初始化可编辑的图像空间 OBB，不代表场景标定或已找到支撑"
                    ),
                    "scale": {
                        "level": 0,
                        "evidence": draft_evidence("未沿用 2.4m 料斗跨度假设；实际尺度参照待补充"),
                    },
                },
                "masks": {},
                "support_contacts": [],
                "render_class": "opaque_rigid",
            }
        )
    return {
        "schema_version": "2.0",
        "task_id": "catalog_" + "_".join(ids),
        "workspace": str(workspace.resolve()),
        "mode": "B",
        "background": {
            "path": str(background.resolve()),
            "sha256": sha256(background),
            "width": background_info["width"],
            "height": background_info["height"],
            "description": "用户提供的原始场景；叠字/前景/承载区需标注",
        },
        "objects": objects,
        "scene": {
            "lighting": "尚未估计主光；参考局部漫射明暗，不增设摄影棚高光",
            "camera_style": "原图局部风格；需选同深度对照区",
            "support_regions": [],
            "occlusion_edges": [],
            "evidence": draft_evidence("待场景分析；未执行深度估计和三维重建"),
        },
        "constraints": {
            "require_support": True,
            "require_foreground_occlusion": True,
            "require_confirmed_masks": True,
            "background_protection": "strict",
            "require_stability_evidence": True,
        },
        "generator": {
            "adapter": "bailian",
            "route": "guided_generation_then_restore",
            "output_registration": "resize_same_aspect",
        },
        "budget": {"max_image_calls": 1, "max_vlm_calls": 1, "max_candidates": 1},
        "harmonization": {
            "mode": "none",
            "blur_sigma_px": 0,
            "noise_std": 0,
            "feather_radius_px": 0,
        },
        "evaluation": {
            "center_error_max": 0.02,
            "scale_log_error_max": 0.25,
            "rotation_error_max_deg": 10,
            "mask_iou_min": 0.8,
            "contact_gap_max_px": 4,
            "min_changed_visible_fraction": 0.02,
            "required_visual_checks": CHECK_NAMES,
            "require_observed_masks": True,
        },
        "output_dir": "output/controlled_fusion",
        "prompt_notes": "使用当前背景适配的旧物，保留参考身份；所有几何/Mask/支撑草案需先落实。",
    }


def ablation_matrix(job: dict) -> dict:
    variants = [
        ("E0", "alpha粘贴基线", "仅使用同画布候选和已审核二值Mask，作为贴图感对照"),
        ("E1", "A与B路线", "概念整图与像素保护方案分别记录；不共享像素保持结论"),
        ("E2", "完整OBB约束", "比较几何误差；不把可见bbox当完整投影"),
        ("E3", "原前景恢复", "比较杆件连续与实际遮挡；关闭时只能是实验候选"),
        ("E4", "接触支撑", "检查接触间隙、支点和局部阴影"),
        ("E5", "按部件旧化", "检查材质类型及旧化定位，不以红色面积为唯一指标"),
        ("E6", "局部与全局调色", "全局方案不满足B/C像素保护；只作为独立实验"),
        ("E7", "多视图与单视图", "比较关键结构和连接；盲测并绑定输出哈希"),
    ]
    return {
        "task_id": job["task_id"],
        "type": "experiment_plan_not_executed",
        "image_calls_performed": 0,
        "controls": [
            "相同背景",
            "同一完整投影/支撑计划",
            "相同候选预算",
            "有种子时记录实际提交值",
            "固定指标定义与盲测程序",
            "改变一项并记录假设；不宣称像素确定性",
        ],
        "experiments": [
            {"id": eid, "name": name, "procedure": procedure, "status": "NOT_EXECUTED"}
            for eid, name, procedure in variants
        ],
    }
