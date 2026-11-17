"""Per-part weathering prompts and strictly local, bounded camera matching."""

from __future__ import annotations

import cv2
import numpy as np

from fusion_engine.controlled.camera_feedback import apply_user_appearance_feedback
from fusion_engine.controlled.grounding import render_contact
from fusion_engine.controlled.job import ControlledJob, evidence_confirmed
from fusion_engine.controlled.masks import MaskBundle, read_mask
from fusion_engine.controlled.photometry import adjust_surfaces, paired_local_correction
from fusion_engine.errors import ConfigError

MATERIAL_RULES = {
    "carbon_steel": ("接缝/划伤处不规则暗红褐氧化、点蚀、灰尘油泥，降低镜面反射", "纯橙均匀覆盖"),
    "painted_steel": (
        "暗淡残漆、擦伤脱漆；只在露钢/涂层破损处生成深褐锈蚀",
        "完好漆面随机印花式红锈",
    ),
    "cast_iron": ("暗褐氧化、局部鳞片锈、油垢和粗糙铸造表面", "均匀鲜橙涂层"),
    "stainless_steel": ("水渍失光、茶色污染、局部点蚀，保留拉丝方向", "无依据整件厚重碳钢红锈"),
    "aluminium": ("失光、灰白氧化、划痕和点蚀", "把红锈当作铝基材腐蚀产物"),
    "galvanized_steel": ("灰白锌氧化、镀层磨损，只有露钢位置可出现红褐锈", "全表面厚红锈"),
    "copper": ("氧化变暗、局部绿色铜锈和指纹", "碳钢红锈"),
    "brass": ("失光变暗、局部绿锈及接缝污垢", "碳钢红锈"),
    "plastic": ("紫外褪色、发黄粉化、划伤、积灰和应力白痕", "金属锈壳"),
    "rubber": ("积灰、失光、细裂和表面发白", "金属红锈、错误金属高光"),
    "wood": ("顺纹风化灰化、裂纹、水渍、纤维磨损", "金属锈斑、乱向木纹"),
    "paper": ("潮斑、压痕、掉色、折角和纤维破口", "钢板高光"),
    "glass": ("灰尘、水垢和划伤，保持透明、厚度反光和背景透射", "整片不透明锈层"),
    "fabric": ("灰尘油污、磨毛、缝线和顺重力褶皱", "金属硬壳轮廓"),
    "ceramic": ("积垢、细釉裂纹和局部缺口", "普通金属锈"),
    "unknown": ("保守失光、积灰和擦痕；基材待确认", "猜测腐蚀产物或夸张结构损坏"),
}


def material_plan(item: dict) -> dict:
    parts = []
    for part in item["material_parts"]:
        positive, negative = MATERIAL_RULES[part["base_material"]]
        parts.append(
            {
                **part,
                "weathering_positive": positive,
                "weathering_negative": negative,
                "geometry_change_allowed": item["ageing"]["allow_structural_damage"]
                and bool(part.get("allowed_damage")),
                "locations": part.get(
                    "weathering_locations", ["依据真实接缝、暴露面、接触区和流路定位"]
                ),
                "rendering_performed": False,
            }
        )
    return {
        "object_id": item["id"],
        "ageing": item["ageing"],
        "parts": parts,
        "level_is_not_rust_area_fraction": True,
        "type": "requested_material_plan_not_observed_fact",
    }


def roi_statistics(image: np.ndarray, mask: np.ndarray) -> dict:
    if not mask.any():
        return {"status": "UNKNOWN", "reason": "ROI 为空"}
    values = image[mask]
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[mask]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    high_frequency = cv2.Laplacian(gray, cv2.CV_64F)[mask]
    return {
        "status": "OBSERVED",
        "pixels": int(mask.sum()),
        "rgb_median": np.median(values, axis=0).tolist(),
        "lab8_median": np.median(lab, axis=0).tolist(),
        "luminance_quantiles": np.quantile(lab[:, 0], [0.1, 0.5, 0.9]).tolist(),
        "laplacian_variance": float(np.var(high_frequency)),
        "limits": "高频能量受内容/材质影响，不单独证明真实感；LAB8 未做 ICC 色彩管理",
    }


def _blend_local(before: np.ndarray, adjusted: np.ndarray, target: np.ndarray) -> np.ndarray:
    if not target.any():
        return before.copy()
    distance = cv2.distanceTransform(target.astype(np.uint8), cv2.DIST_L2, 5)[target]
    alpha = np.minimum(distance / 3, 1.0)[:, None]
    result = before.copy()
    result[target] = np.clip(
        np.rint(before[target] * (1 - alpha) + adjusted[target] * alpha), 0, 255
    ).astype(np.uint8)
    return result


def prepare_foreground(
    scene: np.ndarray,
    candidate: np.ndarray,
    job: ControlledJob,
    masks: MaskBundle,
    estimated_visible: dict[str, np.ndarray] | None = None,
    estimated_parts: dict[str, np.ndarray] | None = None,
    *,
    skip_unreliable_photometry: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    if candidate.shape != scene.shape:
        raise ConfigError("候选与原图尺寸不一致，禁止在后处理阶段隐式缩放")
    if estimated_visible is not None and any(
        np.any(value & ~masks.objects[oid].arrays["edit"])
        for oid, value in estimated_visible.items()
    ):
        raise ConfigError("候选实体越出编辑范围，必须重新规划或重生，不能用编辑Mask裁掉主体")
    settings = job.data["harmonization"]
    target = masks.arrays["obj_visible"] & masks.arrays["edit"] & ~masks.arrays["protect"]
    if estimated_visible is not None:
        observed = np.zeros(scene.shape[:2], bool)
        for oid, value in estimated_visible.items():
            if oid in masks.objects:
                observed |= value & masks.objects[oid].arrays["edit"]
        target = observed & ~masks.arrays["protect"]
    result, operations = candidate.copy(), []
    if settings["mode"] == "paired_local":
        if target.any():
            try:
                result, record = paired_local_correction(scene, result, job, masks, target)
            except ConfigError:
                if not skip_unreliable_photometry:
                    raise
                record = {
                    "operation": "paired_background_local_linear_RGB",
                    "status": "SKIPPED_UNRELIABLE_H0",
                    "reason": "配对样本不足；保留候选，不放宽阈值；阴影漂移校正也将UNKNOWN",
                }
            operations.append(record)
        else:
            operations.append(
                {
                    "operation": "paired_background_local_linear_RGB",
                    "status": "SKIPPED_NO_RELIABLE_TARGET",
                }
            )
    if settings["mode"] == "bounded_lab":
        if not settings.get("comparable_roi_confirmed") or "reference_roi" not in settings:
            raise ConfigError("局部 LAB 匹配需要人工确认的同材质/同照明对照 ROI")
        if not evidence_confirmed(settings["reference_roi"]["evidence"]):
            raise ConfigError("对照 ROI 的证据状态尚未审核，不能只设置确认开关")
        reference, record = read_mask(job, settings["reference_roi"], scene.shape[:2])
        if not reference.any() or not target.any() or np.any(reference & masks.arrays["edit"]):
            raise ConfigError("对照 ROI 必须非空且位于编辑区外，目标区域必须非空")
        lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB).astype(np.float32)
        reference_lab = cv2.cvtColor(scene, cv2.COLOR_RGB2LAB)
        requested = np.median(reference_lab[reference], axis=0) - np.median(lab[target], axis=0)
        bound = np.array(settings.get("max_lab_shift", [12, 5, 5]))
        shift = np.clip(requested, -bound, bound) * settings.get("strength", 0.5)
        lab[target] = np.clip(lab[target] + shift, 0, 255)
        converted = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        result = _blend_local(result, converted, target)
        operations.append(
            {
                "operation": "bounded_local_LAB8",
                "actual_shift": shift.tolist(),
                "requested_shift": requested.tolist(),
                "reference": record,
                "claim": "局部统计校正，不代表重新计算物理光照",
            }
        )
    result, records = adjust_surfaces(
        result, job, masks, target, estimated_visible, estimated_parts
    )
    operations += records
    result, records = apply_user_appearance_feedback(
        result, job, masks, target, estimated_visible
    )
    operations += records
    sigma = settings.get("blur_sigma_px", 0)
    if sigma and target.any():
        for oid, values in masks.objects.items():
            region = (
                (
                    estimated_visible.get(oid, np.zeros(scene.shape[:2], bool))
                    if estimated_visible is not None
                    else values.arrays["obj_visible"]
                )
                & values.arrays["edit"]
                & target
            )
            if not region.any():
                continue
            weights = region.astype(np.float32)
            numerator = cv2.GaussianBlur(
                result.astype(np.float32) * weights[..., None], (0, 0), sigma
            )
            denominator = cv2.GaussianBlur(weights, (0, 0), sigma)
            blurred = numerator / np.maximum(denominator[..., None], 1e-6)
            adjusted = result.copy()
            adjusted[region] = np.clip(np.rint(blurred[region]), 0, 255).astype(np.uint8)
            result = _blend_local(result, adjusted, region)
            operations.append(
                {
                    "operation": "masked_normalized_gaussian",
                    "object_id": oid,
                    "sigma_px": sigma,
                    "edge_fade_px": 3,
                }
            )
    std = settings.get("noise_std", 0)
    if std and target.any():
        if "noise_seed" not in settings:
            raise ConfigError("加噪必须提供显式 noise_seed 以便审计")
        rng = np.random.default_rng(settings["noise_seed"])
        adjusted = result.copy()
        adjusted[target] = np.clip(
            np.rint(result[target] + rng.normal(0, std, result[target].shape)), 0, 255
        ).astype(np.uint8)
        result = _blend_local(result, adjusted, target)
        operations.append(
            {"operation": "local_rgb_gaussian_noise", "std": std, "seed": settings["noise_seed"]}
        )
    return result, target, operations


def harmonize(
    scene: np.ndarray,
    candidate: np.ndarray,
    job: ControlledJob,
    masks: MaskBundle,
    estimated_visible: dict[str, np.ndarray] | None = None,
    estimated_parts: dict[str, np.ndarray] | None = None,
    *,
    registration: dict | None = None,
    diagnostics_dir=None,
) -> tuple[np.ndarray, dict]:
    if job.data.get("feature_flags", {}).get("v3_layer_compositor"):
        from fusion_engine.controlled.fusion_v3.renderer import render

        return render(
            scene,
            candidate,
            job,
            masks,
            estimated_visible,
            estimated_parts,
            registration=registration,
            diagnostics_dir=diagnostics_dir,
        )
    result, target, operations = prepare_foreground(
        scene, candidate, job, masks, estimated_visible, estimated_parts
    )
    settings = job.data["harmonization"]
    if (
        job.data["mode"] in ("B", "C")
        or job.data["constraints"]["background_protection"] == "strict"
    ):
        alpha = masks.arrays["edit"].astype(np.float32)
        if estimated_visible is not None:
            # Contact proposals are not image-copy regions: doing so imported dark rectangles.
            alpha = target.astype(np.float32)
        radius = settings.get("feather_radius_px", 0)
        if radius:
            distance = cv2.distanceTransform(alpha.astype(np.uint8), cv2.DIST_L2, 5)
            alpha = np.minimum(distance / radius, 1.0) * alpha
        alpha[masks.arrays["protect"]] = 0
        result = np.clip(
            np.rint(scene * (1 - alpha[..., None]) + result * alpha[..., None]), 0, 255
        ).astype(np.uint8)
    if estimated_visible is not None:
        result, contact_record = render_contact(scene, result, job, masks, estimated_visible)
        operations.append({"operation": "bounded_receiver_contact_occlusion", **contact_record})
    result[masks.arrays["protect"]] = scene[masks.arrays["protect"]]
    operations.append(
        {
            "operation": "restore_original_protected_pixels_last",
            "pixels": int(masks.arrays["protect"].sum()),
            "original_foreground_restored": True,
        }
    )
    return result, {
        "operations": operations,
        "render_scope": "estimated_instance_visible_union_contact_within_explicit_edit"
        if estimated_visible is not None
        else "planned_edit",
        "colour_edge_fade_px": 3,
        "global_case_grading_used": False,
        "object_stats": roi_statistics(result, target),
    }
