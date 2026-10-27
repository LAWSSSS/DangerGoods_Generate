"""Stage attribution and conservative parent selection, never a naturalness score."""

from __future__ import annotations

from pathlib import Path

from fusion_engine.artifacts import sha256


def diagnose(record: dict) -> dict:
    integrity = record.get("postprocess_integrity", {})
    origins = []
    if integrity.get("status") == "FAIL":
        parts = integrity.get("parts", [])
        if any(p["removed_by_mask"] for p in parts):
            origins.append("segmentation")
        if any(p["restored_to_background"] > p["removed_by_mask"] for p in parts):
            origins.append("compositing")
    observation = (record.get("candidate_observation") or {}).get("result", {})
    failures = []
    for obj in observation.get("objects", []):
        for key in ("identity_status", "support_status", "occlusion_status", "color_status"):
            if obj.get(key) == "FAIL":
                failures.append({"object_id": obj["id"], "check": key})
    if failures:
        origins.append("generation")
    if record["processing"]["status"] != "APPLIED":
        origins.append("scope_or_processing_evidence")
    review = record.get("vlm_review", {}).get("result", {})
    if any(o.get("color_status") == "PASS" for o in observation.get("objects", [])) and any(
        d.get("kind") in ("color_cast", "lighting_specular_mismatch")
        for d in review.get("defects", [])
    ):
        origins.append("photometry_or_review_disagreement")
    return {
        "origins": origins or ["unresolved"],
        "generation_failures": failures,
        "repair_target": "guided_generation"
        if failures
        else "local_reprocess"
        if integrity.get("status") == "FAIL"
        else "review_required",
        "stage_comparison": "candidate_observation_before_processing_vs_saved_output_witnesses",
        "model_opinions_are_ground_truth": False,
    }


def _evidence(record: dict, stage: str) -> dict[str, str]:
    values = {}
    observation = (record.get("candidate_observation") or {}).get("result", {})
    for obj in observation.get("objects", []):
        for key in ("identity_status", "support_status", "occlusion_status", "color_status"):
            values[obj["id"] + "/" + key] = obj.get(key, "UNKNOWN")
    if not values:
        values["candidate_observation"] = "UNKNOWN"
    for key, value in record.get("quality", {}).get("hard_checks", {}).items():
        if key.startswith(("candidate_visible_geometry_", "receiver_contact_")):
            values[key] = value["status"]
    if stage == "master":
        values["postprocess_integrity"] = record.get("postprocess_integrity", {}).get(
            "status", "UNKNOWN"
        )
        for key in ("background_lock", "original_canvas_size", "v3_saved_layers_contract"):
            values[key] = record["quality"]["hard_checks"].get(key, {}).get("status", "UNKNOWN")
    else:
        for key in ("postprocess_integrity", "background_lock", "v3_saved_layers_contract"):
            values[key] = "UNKNOWN"
        values["original_canvas_size"] = (
            record["quality"]["hard_checks"]
            .get("original_canvas_size", {})
            .get("status", "UNKNOWN")
        )
    return values


def select_parent(record: dict, root: Path, previous: dict | None) -> dict:
    processing_ok = (
        record["processing"]["status"] == "APPLIED"
        and record.get("postprocess_integrity", {}).get("status") == "PASS"
        and all(
            record["quality"]["hard_checks"].get(k, {}).get("status") == "PASS"
            for k in ("background_lock", "original_canvas_size", "v3_saved_layers_contract")
        )
    )
    stage = "master" if processing_ok else "aligned_raw"
    path = root / record["master_path" if processing_ok else "aligned_path"]
    current = {
        "candidate_id": record["id"],
        "stage": stage,
        "path": str(path),
        "sha256": sha256(path),
        "comparison_evidence": _evidence(record, stage),
        "is_accepted_output": False,
    }
    if previous is None:
        return {**current, "decision": "INITIAL_UNACCEPTED_REPAIR_REFERENCE"}
    # Only compare shared criteria. A new unknown/failure cannot establish
    # improvement, and a change from FAIL to UNKNOWN isn't a verified repair.
    old, new = previous["comparison_evidence"], current["comparison_evidence"]
    common = set(old) & set(new)
    improved = [k for k in common if new[k] == "PASS" and old[k] != "PASS"]
    regressions = [
        k
        for k in common
        if (old[k] == "PASS" and new[k] != "PASS") or (new[k] == "FAIL" and old[k] != "FAIL")
    ]
    added_failures = [k for k in set(new) - set(old) if new[k] == "FAIL"]
    if improved and not regressions and not added_failures:
        return {
            **current,
            "decision": "IMPROVED_WITHOUT_NEW_KNOWN_REGRESSION",
            "improved_checks": sorted(improved),
        }
    return {
        **previous,
        "decision": "KEEP_PREVIOUS_NO_CONFIRMED_IMPROVEMENT",
        "considered_candidate_id": record["id"],
        "considered_stage": stage,
        "regressions": sorted(regressions + added_failures),
    }
