"""Conservative visible-shape diagnostics for rejection; never amodal or annotation truth."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from fusion_engine.controlled.job import ControlledJob
from fusion_engine.controlled.masks import MaskBundle, read_mask
from fusion_engine.controlled.observation import polygon_region


@dataclass
class CandidateGeometry:
    visible: dict[str, np.ndarray]
    report: dict
    witnesses: dict = field(default_factory=dict)


def analyze_candidate(
    scene: np.ndarray,
    candidate: np.ndarray,
    job: ControlledJob,
    masks: MaskBundle,
    observation: dict | None = None,
) -> CandidateGeometry:
    settings = job.data.get("candidate_geometry")
    if not settings:
        return CandidateGeometry({}, {"status": "NOT_REQUESTED", "hard_checks": {}})
    visible, checks, objects, witnesses = {}, {}, {}, {}
    from fusion_engine.controlled.fusion_v3.shape_evidence import candidate_evidence, enabled

    preserve_structure = enabled(job)
    height, width = scene.shape[:2]
    for item in job.data["objects"]:
        oid = item["id"]
        planned = masks.objects[oid].arrays["obj_visible"]
        if not planned.any():
            objects[oid] = {"status": "UNKNOWN", "reason": "缺少非空可见形状目标，未用OBB假装分割"}
            continue
        guide = next(
            (v for v in (observation or {}).get("result", {}).get("objects", []) if v["id"] == oid),
            None,
        )
        guided = settings["method"] == "vlm_seeded_grabcut"
        if guided and guide is None:
            objects[oid] = {
                "status": "UNKNOWN",
                "reason": "没有候选绑定的VLM可见轮廓建议；未用计划Mask代替",
            }
            continue
        semantic = polygon_region(guide, (height, width)) if guided else planned
        evidence = None
        if preserve_structure and guide is not None:
            evidence = candidate_evidence(scene, candidate, guide, seed=settings["seed"])
            witnesses[oid] = evidence
            semantic = polygon_region({"polygon": guide["polygon"], "holes": []}, (height, width))
        if semantic.sum() < 64 or semantic.mean() > 0.3:
            objects[oid] = {"status": "UNKNOWN", "reason": "语义轮廓为空或覆盖异常，未用于处理"}
            continue
        ys, xs = np.where(masks.objects[oid].arrays["edit"] | planned | semantic)
        margin = max(24, round(max(np.ptp(xs), np.ptp(ys)) * 0.12))
        x0, x1 = max(0, int(xs.min()) - margin), min(width, int(xs.max()) + margin + 1)
        y0, y1 = max(0, int(ys.min()) - margin), min(height, int(ys.max()) + margin + 1)
        size = (x1 - x0, y1 - y0)
        ratio = min(1.0, settings.get("max_side", 1000) / max(size))
        small_size = max(2, round(size[0] * ratio)), max(2, round(size[1] * ratio))

        def crop(value: np.ndarray, mask: bool = False) -> np.ndarray:
            return cv2.resize(
                value[y0:y1, x0:x1].astype(np.uint8),
                small_size,
                interpolation=cv2.INTER_NEAREST if mask else cv2.INTER_AREA,
            )

        original, raw = crop(scene), crop(candidate)
        target = crop(semantic, True).astype(bool)
        allowed = crop(masks.objects[oid].arrays["edit"], True).astype(bool)
        foreground = crop(masks.arrays["fg_occ"], True).astype(bool)
        kernel = np.ones((5, 5), np.uint8)
        expanded = cv2.dilate(target.astype(np.uint8), kernel, iterations=3).astype(bool)
        background = ~expanded & ~foreground
        if background.sum() < 64:
            objects[oid] = {"status": "UNKNOWN", "reason": "独立背景样本不足，停止估计"}
            continue
        offset = np.median((raw.astype(np.float32) - original)[background], axis=0)
        difference = np.max(np.abs(raw.astype(np.float32) - original - offset), axis=2)
        noise = float(np.quantile(difference[background], 0.90))
        if not guided and noise > settings.get("max_background_difference", 40):
            objects[oid] = {
                "status": "UNKNOWN",
                "reason": "局部背景被重绘或错位，差分分割不可靠",
                "background_p90": noise,
            }
            continue
        threshold = max(settings.get("difference_floor", 12), noise * 1.2)
        core = cv2.erode(target.astype(np.uint8), kernel).astype(bool)
        base = np.maximum(original.astype(np.float32) + offset, 1)
        raw_float = raw.astype(np.float32)
        brightness_ratio = raw_float.mean(axis=2) / np.maximum(base.mean(axis=2), 1)
        chromatic_distance = np.max(
            np.abs(
                raw_float / np.maximum(raw_float.sum(axis=2, keepdims=True), 1)
                - base / np.maximum(base.sum(axis=2, keepdims=True), 1)
            ),
            axis=2,
        )
        shadow_like = (
            (brightness_ratio > 0.35)
            & (brightness_ratio < 0.92)
            & (chromatic_distance < 0.025)
            & (base.mean(axis=2) > 24)
        )
        sure_foreground = core & (difference > threshold * 1.5) & ~foreground & ~shadow_like
        if guided:
            # Semantic points are independent proposals about THIS candidate, not the planned proxy.
            # Do not treat the whole VLM polygon as a sure foreground segmentation.
            sure_foreground = np.zeros(target.shape, np.uint8)
            for point in guide["foreground_points"]:
                px, py = (np.asarray(point) * [width, height] - [x0, y0]) * ratio
                cv2.circle(sure_foreground, (round(px), round(py)), max(2, round(5 * ratio)), 1, -1)
            sure_foreground = sure_foreground.astype(bool) & expanded & ~foreground
            if evidence is not None:
                sure_foreground |= crop(evidence.foreground, True).astype(bool) & ~foreground
        if sure_foreground.sum() < 16:
            objects[oid] = {
                "status": "UNKNOWN",
                "reason": "可靠前景种子不足；不把计划Mask当生成物体",
                "threshold": threshold,
            }
            continue
        labels = np.full(target.shape, cv2.GC_PR_BGD, np.uint8)
        if guided:
            labels[target & ~foreground] = cv2.GC_PR_FGD
        else:
            labels[(difference > threshold) & expanded & ~foreground] = cv2.GC_PR_FGD
            labels[shadow_like & ~core] = cv2.GC_PR_BGD
        labels[sure_foreground] = cv2.GC_FGD
        negative = ~cv2.dilate(
            (target if guided else allowed).astype(np.uint8), kernel, iterations=2
        ).astype(bool)
        if guided:
            # The VLM polygon is coarse: a tight negative band clipped real object volume.
            # Permit the appearance model to find the boundary inside an enlarged semantic box.
            sy, sx = np.where(target)
            slack = max(10, round(max(np.ptp(sx), np.ptp(sy)) * 0.08))
            negative = np.ones_like(target)
            negative[
                max(0, int(sy.min()) - slack) : min(target.shape[0], int(sy.max()) + slack + 1),
                max(0, int(sx.min()) - slack) : min(target.shape[1], int(sx.max()) + slack + 1),
            ] = False
            if evidence is None:
                for hole in guide.get("holes", []):
                    hole_mask = crop(
                        polygon_region({"polygon": hole, "holes": []}, (height, width)), True
                    )
                    negative |= cv2.erode(hole_mask, kernel).astype(bool)
            else:
                proposed = crop(evidence.suggested_holes, True).astype(bool)
                supported = crop(evidence.background, True).astype(bool)
                labels[proposed] = cv2.GC_PR_BGD
                negative |= supported
                negative &= ~sure_foreground
        labels[negative] = cv2.GC_BGD
        if evidence is not None:
            labels[sure_foreground] = cv2.GC_FGD
        # Original foreground is always in front. Border pixels anchor background mixtures.
        labels[foreground] = cv2.GC_BGD
        labels[[0, -1], :] = cv2.GC_BGD
        labels[:, [0, -1]] = cv2.GC_BGD
        cv2.setRNGSeed(settings["seed"])
        try:
            cv2.grabCut(
                cv2.cvtColor(raw, cv2.COLOR_RGB2BGR),
                labels,
                None,
                np.zeros((1, 65), np.float64),
                np.zeros((1, 65), np.float64),
                settings.get("iterations", 3),
                cv2.GC_INIT_WITH_MASK,
            )
        except cv2.error:
            objects[oid] = {"status": "UNKNOWN", "reason": "GrabCut无法稳定拟合；未回退为计划真值"}
            continue
        estimated = (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)
        count, components = cv2.connectedComponents(estimated.astype(np.uint8))
        # Keep every connected component anchored by explicit high-difference interior seeds.
        valid_ids = np.unique(components[sure_foreground])
        estimated &= np.isin(components, valid_ids[valid_ids != 0])
        selected_ids = valid_ids[valid_ids != 0]
        component_quality = []
        for component_id in selected_ids:
            region = components == component_id
            component_quality.append(
                {
                    "area": int(region.sum()),
                    "seed_density": float(
                        (sure_foreground & region).sum() / max(int(region.sum()), 1)
                    ),
                }
            )
        area_ratio = float(estimated.sum() / max(int(target.sum()), 1))
        major = [
            value
            for value in component_quality
            if value["area"] >= max(4, int(target.sum() * 0.01))
        ]
        weak = any(
            value["seed_density"] < settings.get("min_seed_density", 0.03) for value in major
        )
        shadow_fraction = float((shadow_like & estimated).sum() / max(int(estimated.sum()), 1))
        if (
            (weak and not guided)
            or len(major) > settings.get("max_component_count", 8)
            or not settings.get("min_area_ratio", 0.2)
            <= area_ratio
            <= settings.get("max_area_ratio", 3)
            or (shadow_fraction > 0.5 and not guided)
        ):
            objects[oid] = {
                "status": "UNKNOWN",
                "reason": "估计分量缺少可靠种子、过碎、面积异常或疑似阴影过多；未将该轮廓用于外观处理",
                "component_quality": component_quality,
                "area_ratio": area_ratio,
                "shadow_like_fraction": shadow_fraction,
            }
            checks[f"candidate_segmentation_quality_{oid}"] = {
                "status": "UNKNOWN",
                "detail": objects[oid]["reason"],
            }
            continue
        full = np.zeros((height, width), bool)
        full[y0:y1, x0:x1] = cv2.resize(
            estimated.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        full &= ~masks.arrays["fg_occ"]
        refinement = None
        if job.data.get("feature_flags", {}).get("v3_mask_refinement") and guide is not None:
            from fusion_engine.controlled.fusion_v3.refinement import refine_instance

            full, refinement = refine_instance(
                scene,
                candidate,
                full,
                guide,
                masks.objects[oid].arrays["edit"],
                masks.arrays["fg_occ"],
                seed=settings["seed"],
                evidence=evidence,
            )
            checks[f"candidate_part_connectivity_{oid}"] = {
                "status": "UNKNOWN" if refinement["status"] == "UNKNOWN" else "PASS",
                "detail": "估计轮廓的部件连通性；原候选缺件不能由颜色/膨胀补出",
                "report": refinement,
            }
        if not full.any():
            objects[oid] = {"status": "UNKNOWN", "reason": "估计可见轮廓为空"}
            continue
        visible[oid] = full
        intersection, union = (full & planned).sum(), (full | planned).sum()
        iou = float(intersection / union)
        escape = float((full & ~masks.objects[oid].arrays["edit"]).sum() / full.sum())
        checks[f"candidate_visible_geometry_{oid}"] = {
            "status": "FAIL"
            if iou < settings.get("min_visible_iou", 0.4)
            or escape > settings.get("max_escape_fraction", 0.03)
            else "PASS",
            "detail": "仅检测估计可见形状的偏离/裁切风险；PASS不证明实际完整尺寸或三维无碰撞",
            "visible_iou": iou,
            "escape_fraction": escape,
        }
        spatial = item.get("spatial_constraints", {})
        if "forbidden" in spatial:
            forbidden, _ = read_mask(job, spatial["forbidden"], (height, width))
            pixels = int((full & forbidden).sum())
            checks[f"candidate_forbidden_{oid}"] = {
                "status": "FAIL" if pixels else "PASS",
                "detail": "估计可见物体进入明确二维禁入区",
                "pixels": pixels,
            }
        distances = cv2.distanceTransform((~full).astype(np.uint8), cv2.DIST_L2, 5)
        gaps = []
        for contact in item["support_contacts"]:
            x, y = (
                min(width - 1, round(contact["point"][0] * width)),
                min(height - 1, round(contact["point"][1] * height)),
            )
            if not masks.arrays["fg_occ"][y, x]:
                gaps.append(float(distances[y, x]))
        if gaps:
            checks[f"candidate_contact_{oid}"] = {
                "status": "FAIL"
                if max(gaps) > settings.get("contact_tolerance_px", 12)
                else "PASS",
                "detail": "支点到估计前景的距离；阴影/同色可能误分，仍需人工分辨实体接触；遮挡内支点跳过",
                "gaps_px": gaps,
            }
        objects[oid] = {
            "status": "ESTIMATED",
            "method": "VLM_candidate_points_seeded_GrabCut"
            if guided
            else "seeded_grabcut_with_robust_background_difference",
            "semantic_guidance_is_estimate": guided,
            "negative_seed_scope": "expanded_semantic_box_with_internal_holes"
            if guided
            else "expanded_explicit_edit",
            "threshold_rgb8": threshold,
            "background_p90": noise,
            "seed_pixels_at_work_scale": int(sure_foreground.sum()),
            "work_size": list(small_size),
            "estimated_visible_pixels": int(full.sum()),
            "connected_components": int(count - 1),
            "amodal_inferred": False,
            "component_quality": component_quality,
            "negative_seed_pixels": int(negative.sum()),
            "shadow_like_fraction": shadow_fraction,
            "shadow_heuristic_is_not_semantic_segmentation": True,
            **({"part_refinement": refinement} if refinement else {}),
            **({"structure_evidence": evidence.report} if evidence is not None else {}),
        }
    uncertain = len(visible) != len(job.data["objects"])
    if len(visible) > 1:
        overlap = int((np.sum(list(visible.values()), axis=0) > 1).sum())
        checks["candidate_instance_overlap"] = {
            "status": "FAIL" if overlap else "PASS",
            "detail": "不同不透明实例的估计可见轮廓互斥；重叠时停止分区外观处理",
            "pixels": overlap,
        }
    status = (
        "RISK"
        if any(check["status"] == "FAIL" for check in checks.values())
        else "UNKNOWN"
        if uncertain or any(check["status"] == "UNKNOWN" for check in checks.values())
        else "ESTIMATED_ONLY"
    )
    return CandidateGeometry(
        visible,
        {
            "status": status,
            "hard_checks": checks,
            "objects": objects,
            "reject_on_risk": settings["reject_on_risk"],
            "annotation_eligible": False,
            "physical_collision_verified": False,
            "limitations": "同色、细杆、轮孔、阴影和背景重绘会影响估计；只可辅助拒收，不可自动验收或代替人工分割",
        },
        witnesses,
    )
