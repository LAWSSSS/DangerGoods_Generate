"""Part-local GrabCut refinement of this candidate, without inventing missing geometry."""

from __future__ import annotations

import cv2
import numpy as np

from fusion_engine.controlled.observation import polygon_region


def refine_instance(
    scene: np.ndarray,
    candidate: np.ndarray,
    visible: np.ndarray,
    guide: dict,
    edit: np.ndarray,
    front: np.ndarray,
    *,
    seed: int,
    evidence=None,
) -> tuple[np.ndarray, dict]:
    result = visible.copy()
    reports = []
    h, w = scene.shape[:2]
    # Preserve the negative-space decomposition established by the initial
    # candidate observation/segmentation. Part refinement may not silently turn
    # a background hole into dark opaque object pixels.
    preserved_holes = np.zeros((h, w), bool)
    outer = polygon_region({"polygon": guide["polygon"], "holes": []}, (h, w))
    for hole in guide.get("holes", []):
        negative = polygon_region({"polygon": hole, "holes": []}, (h, w))
        preserved_holes |= negative & outer & ~visible
    if evidence is not None:
        preserved_holes = evidence.background.copy()
    # Separate colour mixtures prevent a large yellow body from dominating a small blue wheel.
    # Parts are semantic proposals; only candidate colours and seeded segmentation supply pixels.
    for part in guide.get("parts", []):
        proposal = polygon_region(
            {"polygon": part["polygon"], "holes": []} if evidence is not None else part,
            (h, w),
        )
        if not proposal.any():
            continue
        yy, xx = np.where(proposal)
        margin = max(8, min(24, round(max(np.ptp(xx), np.ptp(yy)) * 0.12)))
        x0, x1 = max(0, int(xx.min()) - margin), min(w, int(xx.max()) + margin + 1)
        y0, y1 = max(0, int(yy.min()) - margin), min(h, int(yy.max()) + margin + 1)
        roi = np.s_[y0:y1, x0:x1]
        raw = candidate[roi]
        semantic = proposal[roi]
        original_mask = visible[roi] & semantic
        trusted = cv2.erode(original_mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        if trusted.sum() < 12:
            reports.append(
                {"part_id": part["id"], "status": "UNKNOWN", "reason": "部件内部颜色种子不足"}
            )
            continue
        lab = cv2.cvtColor(raw.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
        bg_lab = cv2.cvtColor(scene[roi].astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
        samples = lab[trusted].reshape(-1, 3)
        samples = samples[:: max(1, int(np.ceil(len(samples) / 12000)))].copy()
        cv2.setRNGSeed(seed)
        _, _, centers = cv2.kmeans(
            samples,
            min(3, len(samples)),
            None,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1),
            1,
            cv2.KMEANS_PP_CENTERS,
        )
        distance_to_part = np.sqrt(
            np.min(np.sum((lab[..., None, :] - centers) ** 2, axis=-1), axis=-1)
        )
        background_difference = np.linalg.norm(lab - bg_lab, axis=2)
        # Only strong candidate-supported colours add sure seeds; semantics alone never do.
        near_proposal = (
            cv2.distanceTransform((~semantic).astype(np.uint8), cv2.DIST_L2, 5) <= margin * 0.9
        )
        chromatic_change = np.linalg.norm(lab[..., 1:] - bg_lab[..., 1:], axis=2)
        not_shadow_only = (chromatic_change > 8) | ((lab[..., 0] - bg_lab[..., 0]) > 8)
        seeds = trusted | (
            near_proposal & (distance_to_part < 10) & (background_difference > 18) & not_shadow_only
        )
        seeds &= edit[roi] & ~front[roi]
        if evidence is not None:
            seeds |= (
                evidence.parts.get(part["id"], evidence.foreground)[roi] & edit[roi] & ~front[roi]
            )
        explicit_seeds = np.zeros_like(seeds, np.uint8)
        for point in part.get("foreground_points", []):
            x, y = np.rint(np.asarray(point) * [w, h] - [x0, y0]).astype(int)
            if 1 <= x < seeds.shape[1] - 1 and 1 <= y < seeds.shape[0] - 1:
                cv2.circle(explicit_seeds, (x, y), 1, 1, -1)
        explicit_seeds = explicit_seeds.astype(bool) & edit[roi] & ~front[roi]
        seeds |= explicit_seeds
        seed_hole_conflicts = int((seeds & preserved_holes[roi]).sum())
        seeds &= ~preserved_holes[roi]
        explicit_seeds &= ~preserved_holes[roi]
        expanded = cv2.dilate(
            semantic.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=2
        ).astype(bool)
        labels = np.full(semantic.shape, cv2.GC_PR_BGD, np.uint8)
        labels[expanded] = cv2.GC_PR_FGD
        labels[seeds] = cv2.GC_FGD
        # A hole proposal that disagrees with background colour is uncertain, not hard-negative truth.
        hole_conflicts, unscoped_holes = 0, 0
        for hole in part.get("holes", []):
            negative = polygon_region({"polygon": hole, "holes": []}, (h, w))[roi]
            if negative.any() and (negative & ~semantic).sum() / negative.sum() > 0.2:
                unscoped_holes += 1
                continue
            negative = cv2.erode(negative.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            hole_conflicts += int((negative & (background_difference > 18)).sum())
            if evidence is None:
                labels[negative & (background_difference <= 18)] = cv2.GC_BGD
            else:
                labels[negative & ~seeds] = cv2.GC_PR_BGD
        for point in part.get("background_points", []):
            x, y = np.rint(np.asarray(point) * [w, h] - [x0, y0]).astype(int)
            if 1 <= x < seeds.shape[1] - 1 and 1 <= y < seeds.shape[0] - 1:
                if seeds[y, x]:
                    hole_conflicts += 1
                elif background_difference[y, x] <= 18:
                    labels[y, x] = cv2.GC_BGD
        labels[explicit_seeds] = cv2.GC_FGD
        labels[preserved_holes[roi]] = cv2.GC_BGD
        if evidence is not None:
            labels[seeds] = cv2.GC_FGD
        # Coarse semantic contours are not a hard clipping boundary. Let the
        # part-specific colour model inspect the complete padded native ROI.
        labels[~near_proposal & ~original_mask] = cv2.GC_BGD
        labels[front[roi] | ~edit[roi]] = cv2.GC_BGD
        labels[[0, -1], :] = cv2.GC_BGD
        labels[:, [0, -1]] = cv2.GC_BGD
        try:
            cv2.setRNGSeed(seed)
            cv2.grabCut(
                cv2.cvtColor(raw, cv2.COLOR_RGB2BGR),
                labels,
                None,
                np.zeros((1, 65), np.float64),
                np.zeros((1, 65), np.float64),
                3,
                cv2.GC_INIT_WITH_MASK,
            )
        except cv2.error:
            reports.append(
                {"part_id": part["id"], "status": "UNKNOWN", "reason": "部件分割拟合失败"}
            )
            continue
        selected = (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)
        _, components = cv2.connectedComponents(selected.astype(np.uint8))
        ids = np.unique(components[seeds & selected])
        selected &= np.isin(components, ids[ids > 0])
        area_ratio = float(selected.sum() / max(int(semantic.sum()), 1))
        if not 0.2 <= area_ratio <= 2.5:
            reports.append(
                {
                    "part_id": part["id"],
                    "status": "UNKNOWN",
                    "reason": "部件面积异常，未合入",
                    "area_ratio": area_ratio,
                }
            )
            continue
        added = selected & ~result[roi]
        result[roi] |= selected
        reports.append(
            {
                "part_id": part["id"],
                "status": "ESTIMATED",
                "work_size": [x1 - x0, y1 - y0],
                "added_pixels": int(added.sum()),
                "seed_pixels": int(seeds.sum()),
                "explicit_VLM_seed_pixels": int(explicit_seeds.sum()),
                "out_of_part_holes_ignored": unscoped_holes,
                "preserved_hole_seed_conflicts": seed_hole_conflicts,
                "unconfirmed_hole_colour_conflicts": hole_conflicts,
                "semantic_polygon_copied_as_mask": False,
                "semantic_boundary_used_as_hard_clip": False,
            }
        )
    count, _, stats, _ = cv2.connectedComponentsWithStats(result.astype(np.uint8), connectivity=8)
    major = [
        int(stats[i, cv2.CC_STAT_AREA])
        for i in range(1, count)
        if stats[i, cv2.CC_STAT_AREA] >= max(12, result.sum() * 0.005)
    ]
    connection_unknown = len(major) > 1 and not front.any()
    return result, {
        "method": "native_part_local_candidate_colour_seeded_GrabCut",
        "status": "UNKNOWN" if connection_unknown else "ESTIMATED",
        "parts": reports,
        "major_component_areas": major,
        "connection_status": "UNKNOWN_DISCONNECTED_WITHOUT_CONFIRMED_OCCLUDER"
        if connection_unknown
        else "ESTIMATED_CONNECTED",
        "added_pixels": int((result & ~visible).sum()),
        "reconstructed_geometry": False,
        "gap_bridging_morphology_used": False,
        "preserved_initial_negative_space_pixels": int(preserved_holes.sum()),
        "hole_policy": "paired_background_supported_only_foreground_seeds_take_priority"
        if evidence is not None
        else "legacy_initial_hole_preservation",
        "annotation_eligible": False,
    }
