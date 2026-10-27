"""Candidate-supported seeds and structure witnesses; VLM holes remain proposals."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from fusion_engine.controlled.observation import polygon_region


@dataclass
class ShapeEvidence:
    foreground: np.ndarray
    background: np.ndarray
    suggested_holes: np.ndarray
    parts: dict[str, np.ndarray] = field(default_factory=dict)
    report: dict = field(default_factory=dict)


def enabled(job) -> bool:
    return job.data.get("feature_flags", {}).get("v3_structure_preservation", False)


def candidate_evidence(
    scene: np.ndarray, candidate: np.ndarray, guide: dict, *, seed: int = 42
) -> ShapeEvidence:
    """Use local paired appearance and explicit seeds, never a filled design shape.

    Masks below are estimates. A matching colour is a witness for preservation,
    not proof of identity, depth, or the existence of a hidden connection.
    """
    shape = scene.shape[:2]
    h, w = shape
    outer = polygon_region({"polygon": guide["polygon"], "holes": []}, shape)
    holes = np.zeros(shape, bool)
    for part in [guide, *guide.get("parts", [])]:
        for points in part.get("holes", []):
            holes |= polygon_region({"polygon": points, "holes": []}, shape)
    holes &= outer
    fg, bg = np.zeros(shape, bool), np.zeros(shape, bool)
    answer = ShapeEvidence(fg, bg, holes)
    yy, xx = np.where(outer)
    if not len(xx):
        answer.report = {"status": "UNKNOWN", "reason": "empty_candidate_proposal"}
        return answer
    pad = max(16, round(max(np.ptp(xx), np.ptp(yy)) * 0.08))
    x0, x1 = max(0, int(xx.min()) - pad), min(w, int(xx.max()) + pad + 1)
    y0, y1 = max(0, int(yy.min()) - pad), min(h, int(yy.max()) + pad + 1)
    roi = np.s_[y0:y1, x0:x1]
    raw = candidate[roi]
    ring = ~cv2.dilate(outer[roi].astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    offset = (
        np.median(raw.astype(np.float32)[ring] - scene[roi].astype(np.float32)[ring], axis=0)
        if ring.sum() >= 32
        else np.zeros(3, np.float32)
    )
    reference = np.clip(scene[roi].astype(np.float32) + offset, 0, 255) / 255
    lab = cv2.cvtColor(raw.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB)
    bg_distance = np.linalg.norm(lab - ref_lab, axis=2)
    chroma_change = np.linalg.norm(lab[..., 1:] - ref_lab[..., 1:], axis=2)
    not_shadow_only = (chroma_change > 4) | ((lab[..., 0] - ref_lab[..., 0]) > 6)
    closest_part = np.full(raw.shape[:2], np.inf, np.float32)
    reports = []
    # Parts separate a small blue wheel from the much larger rust-coloured body.
    proposals = guide.get("parts") or [{"id": "object", **guide}]
    for part in proposals:
        scope = polygon_region({"polygon": part["polygon"], "holes": []}, shape)[roi]
        search = cv2.dilate(scope.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        points = part.get("foreground_points") or guide.get("foreground_points", [])
        explicit = np.zeros(raw.shape[:2], np.uint8)
        for point in points:
            x, y = np.rint(np.asarray(point) * [w, h] - [x0, y0]).astype(int)
            if 1 <= x < raw.shape[1] - 1 and 1 <= y < raw.shape[0] - 1 and search[y, x]:
                cv2.circle(explicit, (x, y), 1, 1, -1)
        explicit = explicit.astype(bool) & search & (bg_distance > 8)
        samples = lab[explicit]
        if len(samples) < 4:
            reports.append(
                {"part_id": part["id"], "status": "UNKNOWN", "seed_pixels": len(samples)}
            )
            continue
        cv2.setRNGSeed(seed)
        _, _, centers = cv2.kmeans(
            samples.copy(),
            min(3, len(samples)),
            None,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1),
            1,
            cv2.KMEANS_PP_CENTERS,
        )
        distance = np.sqrt(np.min(np.sum((lab[..., None, :] - centers) ** 2, axis=-1), axis=-1))
        closest_part[search] = np.minimum(closest_part[search], distance[search])
        witness = explicit | (
            search
            & (distance < 12)
            & (bg_distance > 8)
            & (distance + 6 < bg_distance)
            & not_shadow_only
        )
        full = np.zeros(shape, bool)
        full[roi] = witness
        answer.parts[part["id"]] = full
        fg[roi] |= witness
        reports.append(
            {
                "part_id": part["id"],
                "status": "ESTIMATED",
                "seed_pixels": len(samples),
                "witness_pixels": int(witness.sum()),
            }
        )
    # Only pixels agreeing with paired background and disagreeing with learned
    # foreground can become hard negatives. Ambiguous hole interiors stay soft.
    bg[roi] = holes[roi] & (bg_distance <= 6) & (closest_part >= 12) & np.isfinite(closest_part)
    conflicts = int((fg & holes).sum())
    bg &= ~fg
    answer.report = {
        "status": "ESTIMATED" if fg.any() else "UNKNOWN",
        "method": "paired_background_and_part_seeded_candidate_colour_witnesses",
        "foreground_pixels": int(fg.sum()),
        "background_supported_hole_pixels": int(bg.sum()),
        "suggested_hole_pixels": int(holes.sum()),
        "foreground_hole_conflicts": conflicts,
        "ambiguous_hole_pixels": int((holes & ~fg & ~bg).sum()),
        "paired_offset_rgb8": offset.tolist(),
        "parts": reports,
        "vlm_polygons_are_ground_truth": False,
        "geometry_reconstructed": False,
    }
    return answer


def structure_retention(
    scene: np.ndarray,
    candidate: np.ndarray,
    output: np.ndarray,
    visible: dict[str, np.ndarray],
    evidence: dict[str, ShapeEvidence],
    *,
    front: np.ndarray | None = None,
    max_loss_fraction: float = 0.02,
) -> dict:
    records = []
    front = np.zeros(scene.shape[:2], bool) if front is None else front
    distinct = np.max(np.abs(candidate.astype(np.int16) - scene), axis=2) > 8
    restored_pixels = np.all(output == scene, axis=2)
    for oid, value in evidence.items():
        mask = visible.get(oid, np.zeros(scene.shape[:2], bool))
        for part, pixels in value.parts.items():
            witnesses = pixels & ~front
            count = int(witnesses.sum())
            lost_mask = witnesses & ~mask
            # Exact restoration of a distinct candidate pixel proves that the
            # compositor removed its contribution; no colour-distance score is
            # used to invent missing geometry.
            restored = witnesses & distinct & restored_pixels
            lost = int((lost_mask | restored).sum())
            status = (
                "UNKNOWN" if count < 16 else "FAIL" if lost / count > max_loss_fraction else "PASS"
            )
            records.append(
                {
                    "object_id": oid,
                    "part_id": part,
                    "status": status,
                    "witness_pixels": count,
                    "removed_by_mask": int(lost_mask.sum()),
                    "restored_to_background": int(restored.sum()),
                    "lost_fraction": lost / max(count, 1),
                }
            )
    status = (
        "FAIL"
        if any(r["status"] == "FAIL" for r in records)
        else "PASS"
        if records and all(r["status"] == "PASS" for r in records)
        else "UNKNOWN"
    )
    return {
        "status": status,
        "parts": records,
        "max_witness_loss_fraction": max_loss_fraction,
        "scope": "candidate_supported_visible_structure_preservation_not_identity_or_physics",
        "annotation_eligible": False,
    }
