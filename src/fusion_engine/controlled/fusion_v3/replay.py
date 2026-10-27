"""Zero-network reprocessing of a hash-bound historical candidate and frozen M2 masks."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from PIL import Image

from fusion_engine.artifacts import new_run_directory, sha256, write_json
from fusion_engine.controlled.appearance import harmonize
from fusion_engine.controlled.fusion_v3.diagnostics import surface_stats
from fusion_engine.controlled.grounding import contact_diagnostics
from fusion_engine.controlled.job import ControlledJob, read_json
from fusion_engine.controlled.observation import material_regions
from fusion_engine.controlled.planning import prepare
from fusion_engine.controlled.quality import evaluate
from fusion_engine.errors import ConfigError
from fusion_engine.images import load_rgb, save_png


def _inside(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ConfigError("历史运行引用路径越界或文件不存在")
    return path


def _same_geometry(old: dict, new: dict) -> bool:
    keys = ("placement", "masks", "support_contacts", "spatial_constraints", "render_class")
    a = {v["id"]: {k: v.get(k) for k in keys} for v in old["objects"]}
    b = {v["id"]: {k: v.get(k) for k in keys} for v in new["objects"]}
    return (
        a == b
        and old["scene"] == new["scene"]
        and old.get("candidate_geometry") == new.get("candidate_geometry")
    )


def reprocess(
    job: ControlledJob,
    source_run: Path,
    candidate_id: str,
    *,
    ablations: bool = False,
    refine_masks: bool = False,
) -> dict:
    from fusion_engine.controlled.workflow import _inventory, _verify_inputs, runtime_provenance

    if not job.data.get("feature_flags", {}).get("v3_layer_compositor"):
        raise ConfigError("reprocess 需要显式 V3 图层开关")
    if ablations and not job.data.get("feature_flags", {}).get("v3_diagnostics"):
        raise ConfigError("固定图层消融需要开启v3_diagnostics以保存同一份交互场")
    source = source_run.expanduser().resolve()
    manifest = read_json(source / "manifest.json")
    old_job_path = source / "job.snapshot.json"
    if sha256(old_job_path) != manifest["config_sha256"]:
        raise ConfigError("历史运行配置已变化，不能绑定缓存证据")
    old = read_json(old_job_path)
    if not _same_geometry(old, job.data):
        raise ConfigError("固定轮廓重处理不得改变放置、支撑、Mask或分割设置；请重新运行候选观察")
    _verify_inputs(manifest["input_inventory"])
    record = next((r for r in manifest["candidates"] if r["id"] == candidate_id), None)
    if record is None:
        raise ConfigError("历史候选ID不存在")
    if sha256(_inside(source, record["master_path"])) != record["master_sha256"]:
        raise ConfigError("历史母版哈希已改变，拒绝对照")
    geometry_path = _inside(source, f"analysis/{candidate_id}.geometry.json")
    geometry = read_json(geometry_path)
    aligned = _inside(source, f"analysis/{candidate_id}.aligned.png")
    if sha256(aligned) != geometry.get("aligned_candidate_sha256"):
        raise ConfigError("历史配准候选哈希不匹配")
    # The manifest independently carries the same geometry and candidate binding.
    if geometry != record.get("processing", {}).get("candidate_geometry"):
        raise ConfigError("历史几何报告与manifest不一致")
    prepared = prepare(job)
    if manifest["input_inventory"] and not any(
        r["sha256"] == sha256(job.asset(job.data["background"]["path"]))
        for r in manifest["input_inventory"]
    ):
        raise ConfigError("重处理背景不是历史原图")
    observation = record.get("candidate_observation")
    if observation and (
        observation.get("candidate_sha256") != sha256(aligned)
        or observation.get("background_sha256") != sha256(job.asset(job.data["background"]["path"]))
    ):
        raise ConfigError("历史VLM观察未绑定当前候选与背景")
    pixels = load_rgb(aligned)
    visible, mask_records = {}, []
    for obj in job.data["objects"]:
        oid = obj["id"]
        path = _inside(source, f"analysis/{candidate_id}/{oid}.estimated_visible.png")
        with Image.open(path) as image:
            if image.mode not in ("1", "L") or image.size != tuple(prepared.scene.shape[1::-1]):
                raise ConfigError("历史轮廓类型或尺寸非法，禁止隐式修复")
            values = np.asarray(image.convert("L"))
        if not np.isin(values, [0, 255]).all():
            raise ConfigError("历史轮廓不是二值 Mask")
        visible[oid] = values == 255
        expected = geometry["objects"][oid].get("estimated_visible_pixels")
        if not visible[oid].any() or expected != int(visible[oid].sum()):
            raise ConfigError("历史轮廓像素数与候选报告不一致")
        mask_records.append(
            {
                "object_id": oid,
                "path": str(path),
                "sha256": sha256(path),
                "evidence_level": "M2_frozen_current_file",
                "historical_mask_hash_available": False,
            }
        )
    # Old v2 records have no mask hashes. They can support A/B diagnosis, not acceptance.
    if job.data["mode"] != "A":
        raise ConfigError("此缓存适配器仅用于A模式消融；旧Mask缺少生成当时的独立哈希，不能升级验收")
    frozen_visible = visible
    if refine_masks:
        from fusion_engine.controlled.candidate_geometry import analyze_candidate

        if not job.data.get("feature_flags", {}).get("v3_mask_refinement") or not observation:
            raise ConfigError("部件分割重处理需要开启v3_mask_refinement及候选绑定的缓存观察")
        estimated = analyze_candidate(prepared.scene, pixels, job, prepared.masks, observation)
        if set(estimated.visible) != set(visible):
            raise ConfigError("部件重估未提供完整实例证据；未沿用计划轮廓填充缺失")
        visible, geometry = estimated.visible, estimated.report
    root = new_run_directory(job.output_root, job.data["task_id"])
    inventory = _inventory(job)
    snapshot = copy.deepcopy(job.data)
    snapshot["workspace"] = str(job.workspace)
    write_json(root / "job.snapshot.json", snapshot)
    write_json(root / "analysis/placement.json", prepared.plan)
    (root / "prompts").mkdir(exist_ok=True)
    (root / "prompts/proposed_v3_prompt_NOT_SENT.txt").write_text(prepared.prompt, encoding="utf8")
    source_binding = {
        "source_run": str(source),
        "manifest_sha256_at_import": sha256(source / "manifest.json"),
        "geometry_sha256_at_import": sha256(geometry_path),
        "aligned_sha256": sha256(aligned),
        "masks": mask_records,
        "source_candidate_id": candidate_id,
        "mask_policy": "reestimated_with_part_refinement_on_cached_observation"
        if refine_masks
        else "frozen_historical_M2",
        "actual_source_generation_prompt_changed": False,
        "legacy_limitation": "历史Mask未记录独立哈希；仅校验现存像素数并冻结此次读取哈希；不证明历史文件从未被改动",
    }
    report = {
        "schema_version": "2.0",
        "pipeline_version": "3.0",
        "task_id": job.data["task_id"],
        "run_dir": str(root),
        "status": "PREPARED",
        "synthetic": True,
        "config_sha256": sha256(root / "job.snapshot.json"),
        "source": source_binding,
        "input_inventory": inventory,
        "runtime": runtime_provenance(),
        "execution": {"image_calls": 0, "vlm_calls": 0, "frozen_candidates": 1},
        "annotations": {"status": "NOT_CREATED"},
        "internal_model_steps": "not_exposed",
    }
    write_json(root / "manifest.json", report)
    try:
        parts = (
            material_regions(
                observation,
                visible,
                prepared.scene.shape[:2],
                {
                    p["object_id"]: p["id"]
                    for p in job.data["harmonization"].get("surface_adjustments", [])
                    if p.get("fill_unassigned_visible")
                },
            )
            if observation
            else None
        )
        output, processing = harmonize(
            prepared.scene,
            pixels,
            job,
            prepared.masks,
            visible,
            parts,
            registration=record["processing"]["registration"],
            diagnostics_dir=root / "analysis/layers",
        )
        path = root / "outputs/candidate-001.png"
        save_png(output, path)
        decoded = load_rgb(path)
        quality = evaluate(prepared, decoded)
        grounding = contact_diagnostics(job, prepared.masks, visible)
        quality["hard_checks"].update(grounding["hard_checks"])
        quality["hard_checks"].update(geometry["hard_checks"])
        integrity = None
        if job.data.get("feature_flags", {}).get("v3_structure_preservation"):
            from fusion_engine.controlled.fusion_v3.shape_evidence import (
                candidate_evidence,
                structure_retention,
            )

            witnesses = {
                obj["id"]: candidate_evidence(
                    prepared.scene, pixels, obj, seed=job.data["candidate_geometry"]["seed"]
                )
                for obj in (observation or {}).get("result", {}).get("objects", [])
            }
            integrity = structure_retention(
                prepared.scene,
                pixels,
                decoded,
                visible,
                witnesses,
                front=prepared.masks.arrays["fg_occ"],
                max_loss_fraction=job.data.get("fusion_v3", {}).get(
                    "max_structure_loss_fraction", 0.02
                ),
            )
            quality["hard_checks"]["postprocess_structure_retention"] = integrity
            write_json(root / "analysis/structure.json", integrity)
        from fusion_engine.controlled.fusion_v3.diagnostics import saved_contract

        quality["hard_checks"]["v3_saved_layers_contract"] = saved_contract(
            prepared.scene, path, prepared.masks, processing
        )
        # Cached visual opinions describe an older master. Preserve, do not reuse as new scores.
        quality["status"] = (
            "REJECTED"
            if any(v["status"] == "FAIL" for v in quality["hard_checks"].values())
            else "CONCEPT_ONLY"
        )
        quality["annotation_ready"] = False
        quality["new_visual_review"] = "NOT_PERFORMED_REVIEW_REQUIRED"
        if geometry.get("status") in ("UNKNOWN", "RISK"):
            quality["status"] = "REJECTED"
        if integrity and integrity["status"] != "PASS":
            quality["status"] = "REJECTED"
        report.update(
            {
                "status": quality["status"],
                "processing": processing,
                "quality": quality,
                "candidate_geometry": geometry,
                "source_observation": observation,
                "historical_visual_opinion_NOT_CURRENT_REVIEW": record.get("vlm_review"),
                "output": {
                    "path": str(path),
                    "sha256": sha256(path),
                    "size": list(decoded.shape[1::-1]),
                },
            }
        )
        current_masks = []
        for oid, region in visible.items():
            destination = root / f"analysis/current_masks/{oid}.png"
            save_png(region.astype(np.uint8) * 255, destination)
            current_masks.append(
                {
                    "object_id": oid,
                    "path": str(destination),
                    "sha256": sha256(destination),
                    "pixels": int(region.sum()),
                    "annotation_eligible": False,
                }
            )
        report["current_masks"] = current_masks
        if ablations:
            baseline_name = (
                "A0_source_v3" if manifest.get("pipeline_version") == "3.0" else "A0_historical_v2"
            )
            variants = {baseline_name: load_rgb(_inside(source, record["master_path"]))}
            if refine_masks:
                data = copy.deepcopy(job.data)
                data["feature_flags"]["v3_diagnostics"] = False
                original_parts = material_regions(
                    observation,
                    frozen_visible,
                    prepared.scene.shape[:2],
                    {
                        p["object_id"]: p["id"]
                        for p in job.data["harmonization"].get("surface_adjustments", [])
                        if p.get("fill_unassigned_visible")
                    },
                )
                variants["A4_v3_with_frozen_old_mask"] = harmonize(
                    prepared.scene,
                    pixels,
                    ControlledJob(data, job.config_path, job.workspace),
                    prepared.masks,
                    frozen_visible,
                    original_parts,
                    registration=record["processing"]["registration"],
                )[0]
            for label, no_appearance, no_contact in (
                ("A1_no_appearance", True, False),
                ("A2_no_contact", False, True),
            ):
                data = copy.deepcopy(job.data)
                data["feature_flags"]["v3_diagnostics"] = False
                if no_appearance:
                    data["harmonization"] = {"mode": "none"}
                    # Hold the independently estimated cast field fixed: do not refit H0 with appearance off.
                    from fusion_engine.controlled.fusion_v3.compositor import (
                        Layers,
                        binary_partition,
                        compose_layers,
                    )

                    diagnostic_dir = root / "analysis/layers"
                    arrays = {
                        name: np.asarray(Image.open(diagnostic_dir / f"mask_{name}.png")) == 255
                        for name in ("object", "front", "contact", "cast", "edit", "protect")
                    }
                    core, edge = binary_partition(arrays["object"])
                    layer = Layers(
                        arrays["object"],
                        core,
                        edge,
                        arrays["front"],
                        arrays["contact"],
                        arrays["cast"],
                        arrays["edit"],
                        arrays["protect"],
                    )
                    variants[label] = compose_layers(
                        prepared.scene,
                        pixels,
                        layer,
                        cast_transmission=np.load(
                            diagnostic_dir / "cast_transmission.npy", allow_pickle=False
                        ),
                        contact_transmission=np.load(
                            diagnostic_dir / "contact_transmission.npy", allow_pickle=False
                        ),
                    )[0]
                else:
                    data["contact_rendering"]["enabled"] = not no_contact
                    variants[label] = harmonize(
                        prepared.scene,
                        pixels,
                        ControlledJob(data, job.config_path, job.workspace),
                        prepared.masks,
                        visible,
                        parts,
                        registration=record["processing"]["registration"],
                    )[0]
            patch = prepared.scene.copy()
            patch[prepared.masks.arrays["edit"]] = pixels[prepared.masks.arrays["edit"]]
            variants["A3_local_patch_DIAGNOSTIC_ONLY_NOT_ACCEPTABLE"] = patch
            variants["A5_v3_final"] = decoded
            rows = []
            object_mask = np.logical_or.reduce(list(visible.values()))
            for name, image in variants.items():
                destination = root / f"ablations/{name}.png"
                save_png(image, destination)
                rows.append(
                    {
                        "name": name,
                        "path": str(destination),
                        "sha256": sha256(destination),
                        "surface": surface_stats(image, object_mask),
                        "human_blind_review": "NOT_PERFORMED",
                    }
                )
            report["ablations"] = rows
        _verify_inputs(inventory)
        for item in mask_records:
            if sha256(Path(item["path"])) != item["sha256"]:
                raise ConfigError("固定Mask在重处理期间变化")
        if sha256(aligned) != source_binding["aligned_sha256"]:
            raise ConfigError("固定候选在重处理期间变化")
        if (
            runtime_provenance()["engine_source_sha256"]
            != report["runtime"]["engine_source_sha256"]
        ):
            raise ConfigError("源码在重处理期间变化，请冻结代码后另建运行")
        write_json(root / "manifest.json", report)
        return report
    except Exception as exc:
        report.update({"status": "FAILED", "error_type": type(exc).__name__})
        write_json(root / "manifest.json", report)
        raise
