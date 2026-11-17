"""Audited planning, bounded generation, deterministic processing, review and export."""

from __future__ import annotations

import copy
import hashlib
import platform
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

import numpy as np

from fusion_engine.artifacts import (
    new_run_directory,
    record_event,
    review_lock,
    sha256,
    utc_now,
    write_json,
)
from fusion_engine.controlled import VERSION
from fusion_engine.controlled.adapters import ConfiguredImageAdapter, GeneratedCandidate
from fusion_engine.controlled.appearance import harmonize
from fusion_engine.controlled.candidate_geometry import analyze_candidate
from fusion_engine.controlled.framing import reassemble
from fusion_engine.controlled.grounding import contact_diagnostics
from fusion_engine.controlled.iteration import (
    IterationPolicy,
    PromptJournal,
    feedback_from_record,
    may_stop_accepted,
)
from fusion_engine.controlled.job import ControlledJob, load_job, load_review, read_json
from fusion_engine.controlled.layout import layout_hint
from fusion_engine.controlled.masks import placement_preview, read_mask, save_masks
from fusion_engine.controlled.normalization import normalize_candidate
from fusion_engine.controlled.observation import (
    OBSERVATION_SCHEMA,
    material_regions,
    observation_request,
)
from fusion_engine.controlled.planning import Prepared, prepare
from fusion_engine.controlled.quality import evaluate, export_annotations
from fusion_engine.controlled.registration import register_background
from fusion_engine.controlled.schema import CHECK_NAMES
from fusion_engine.errors import AcceptanceError, AdapterError, ConfigError, FusionError
from fusion_engine.images import load_rgb, save_png
from fusion_engine.providers.configuration import load_profile


def runtime_provenance() -> dict:
    versions = {}
    for name in (
        "numpy",
        "Pillow",
        "opencv-python",
        "opencv-python-headless",
        "jsonschema",
        "httpx",
        "PyYAML",
    ):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not_installed_under_this_distribution_name"
    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(
        [*package_root.rglob("*.py"), *package_root.rglob("*.json"), *package_root.rglob("*.yaml")]
    ):
        digest.update(str(path.relative_to(package_root)).encode())
        digest.update(path.read_bytes())
    return {
        "python": platform.python_version(),
        "dependencies": versions,
        "engine_source_sha256": digest.hexdigest(),
        "color_pipeline": "EXIF_normalized_decoded_RGB8_no_ICC_transform",
    }


def _inventory(job: ControlledJob) -> list[dict]:
    records = {}

    def walk(value):
        if isinstance(value, dict):
            if "path" in value and isinstance(value["path"], str):
                path = job.asset(value["path"])
                if not path.is_file():
                    raise ConfigError(f"输入文件不存在：{path.name}")
                records[str(path)] = {"path": str(path), "sha256": sha256(path)}
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(job.data)
    return list(records.values())


def _verify_inputs(records: list[dict]) -> None:
    for record in records:
        path = Path(record["path"])
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise AcceptanceError("原始输入已变化，需重新规划；旧证据不能继续使用")


def write_plan(prepared: Prepared, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "placement.json", prepared.plan)
    write_json(directory / "mask_paths.json", save_masks(prepared.masks, directory / "masks"))
    (directory / "prompt.txt").write_text(prepared.prompt, encoding="utf-8")
    write_json(directory / "prompt.json", prepared.plan["prompt_document"])
    save_png(
        placement_preview(prepared.scene, prepared.job, prepared.masks),
        directory / "placement_preview.png",
    )
    x0, y0, x1, y1 = prepared.plan["request_frame"]["rect_pixels"]
    save_png(prepared.scene[y0:y1, x0:x1], directory / "request_scene.png")
    (directory / "README.md").write_text(
        "# 规划预览\n\n绿色框为完整投影 OBB，橙色为编辑区，蓝色为原前景；红色为禁入区，黄色圆点为建议支点，青色框为配对光色采样区。\n\n"
        "此图不是融合结果，轮廓不是自动标注真值；请查看 placement.json 的证据和缺失项。\n",
        encoding="utf-8",
    )


def run(
    job: ControlledJob,
    *,
    candidate: Path | None = None,
    generate: bool = False,
    provider_config: Path | None = None,
    candidate_frame: str = "original",
    max_rounds: int | None = None,
) -> dict:
    if (candidate is None) == (not generate):
        raise ConfigError("请选择 --candidate 或 --generate 中的一种执行方式")
    if candidate_frame not in ("original", "request"):
        raise ConfigError("导入画布只能显式选择 original 或 request")
    if max_rounds is not None and not generate:
        raise ConfigError("--rounds只用于--generate；导入/离线处理不会自动发起生图")
    policy = IterationPolicy.from_job(job, max_rounds)
    prepared = prepare(job)
    journal = PromptJournal(prepared.prompt)
    root = new_run_directory(job.output_root, job.data["task_id"])
    snapshot = copy.deepcopy(job.data)
    snapshot["workspace"] = str(job.workspace)
    write_json(root / "job.snapshot.json", snapshot)
    inventory = _inventory(job)
    write_json(root / "analysis/placement.json", prepared.plan)
    write_json(root / "analysis/input_inventory.json", inventory)
    save_masks(prepared.masks, root / "masks")
    if job.data["generator"].get("layout_hint") == "procedural_proxy":
        hint_pixels, hint_record = layout_hint(prepared)
        save_png(hint_pixels, root / "analysis/layout_hint.png")
        write_json(root / "analysis/layout_hint.json", hint_record)
    save_png(
        placement_preview(prepared.scene, job, prepared.masks),
        root / "analysis/placement_preview.png",
    )
    (root / "prompts/joint_prompt.txt").write_text(prepared.prompt, encoding="utf-8")
    write_json(root / "prompts/joint_prompt.json", prepared.plan["prompt_document"])
    ledger = {"image_calls": 0, "vlm_calls": 0, "candidates": 0, "budget": job.data["budget"]}
    manifest = {
        "schema_version": "2.0",
        "engine_version": VERSION,
        "pipeline_version": job.data.get("pipeline_version", "2.0"),
        "feature_flags": job.data.get("feature_flags", {}),
        "task_id": job.data["task_id"],
        "run_dir": str(root),
        "created_at": utc_now(),
        "status": "PREPARED",
        "mode": job.data["mode"],
        "synthetic": True,
        "config_sha256": sha256(root / "job.snapshot.json"),
        "input_inventory": inventory,
        "candidates": [],
        "execution": ledger,
        "iteration": {"rounds": [], "stop_reason": None},
        "internal_model_steps": "not_exposed",
        "runtime": runtime_provenance(),
        "annotations": {"status": "NOT_CREATED"},
        "selection_policy": "hard_constraints_first_no_automatic_promotion",
    }

    def persist():
        write_json(root / "manifest.json", manifest)
        write_json(root / "logs/budget.json", ledger)

    persist()
    record_event(root, "S00-S10", "PREPARED", readiness=prepared.plan["readiness"])
    if prepared.plan["readiness"] != "READY":
        manifest["status"] = "INPUT_REQUIRED"
        manifest["issues"] = prepared.plan["issues"]
        record_event(root, "S08", "INPUT_REQUIRED", model_calls=0)
        persist()
        return manifest
    try:
        iteration = manifest["iteration"]
        iteration.update(policy.plan(job, generate=generate))
        journal.save(root)
        adapter, profile = None, None
        if generate:
            if job.data["generator"]["adapter"] != "bailian":
                raise ConfigError("--generate 只适用于已配置的 bailian 适配器")
            if provider_config is None:
                raise ConfigError("显式传入 --provider-config；不猜测凭据来源")
            profile = load_profile(provider_config)
            adapter = ConfiguredImageAdapter(profile)
            adapter.preflight(prepared)
            journal.configure_envelope(
                adapter.capabilities["max_prompt_characters"],
                iteration["effective_max_rounds"],
                prepared.plan["prompt_document"]["background_image_index"],
                parent_description_budget=64
                if job.data.get("feature_flags", {}).get("v3_causal_iteration")
                else 0,
            )
            iteration["prompt_envelope"] = {
                "max_characters": journal.max_characters,
                "feedback_characters_per_round": journal.patch_budget,
                "original_prompt_characters": len(prepared.prompt),
                "server_truncation_avoided_by_client_guard": True,
            }
            journal.save(root)
            if iteration["enabled_for_this_run"]:
                if profile.image_options["max_images"] != 1:
                    raise ConfigError("逐轮生图/审阅需要max_images=1；未静默改写私有模型配置")
                if (
                    iteration["effective_max_rounds"] > 1
                    and prepared.plan["request_frame"]["kind"] != "original"
                ):
                    raise ConfigError("累积纠正需要完整原图；禁止用局部裁切候选继续迭代")
                if (
                    iteration["effective_max_rounds"] > 1
                    and sum(len(v["references"]) for v in job.data["objects"]) > 7
                ):
                    raise ConfigError(
                        "累积纠正需为原背景和上一候选保留输入位置；当前最多7张物品参考"
                    )
            manifest["provider"] = profile.public_view()
            manifest["capabilities"] = adapter.capabilities
            manifest["legacy_execution_settings_used"] = False
            manifest["budget_policy"] = (
                "任务预算约束实际调用次数，YAML 中 total_num/优化轮数不启动批处理"
            )
            if job.data["budget"]["max_image_calls"] == 0:
                raise ConfigError("图像调用预算为零")
        else:
            if not candidate.is_file():
                raise ConfigError("导入候选不存在")
            manifest["capabilities"] = {
                "network": False,
                "method": "import_external_candidate",
                "model": "not_claimed",
                "native_mask": "not_claimed",
            }
            if provider_config is not None:
                profile = load_profile(provider_config)
                adapter = ConfiguredImageAdapter(profile)
                manifest["provider"] = profile.public_view()
                manifest["capabilities"].update(
                    {
                        "network": True,
                        "image_generation_requested": False,
                        "vlm_model": profile.vlm_model,
                    }
                )
        prompt, attempt, repair_source = prepared.prompt, 0, None
        selected_parent = None
        causal_iteration = job.data.get("feature_flags", {}).get("v3_causal_iteration", False)
        while True:
            _verify_inputs(inventory)
            if generate:
                limit = job.data["budget"]
                batch_size = profile.image_options["max_images"]
                if (
                    attempt >= iteration["effective_max_rounds"]
                    or ledger["image_calls"] >= limit["max_image_calls"]
                    or ledger["candidates"] + batch_size > limit["max_candidates"]
                ):
                    iteration["stop_reason"] = (
                        "ROUND_LIMIT_REACHED"
                        if attempt >= iteration["effective_max_rounds"]
                        else "BUDGET_EXHAUSTED"
                    )
                    break
                attempt += 1
                adapter.validate_prompt(prepared, journal.current)
                request_record = journal.record_request(attempt)
                prompt = journal.current
                journal.save(root)
                ledger["image_calls"] += 1
                persist()  # Count attempted requests before network I/O; no automatic retry after error.
                record_event(root, "S11", "REQUEST_ATTEMPTED", attempt=attempt)
                (root / f"prompts/attempt-{attempt:02d}.txt").write_text(prompt, encoding="utf-8")
                generated = adapter.generate(
                    prepared, prompt, root / f"candidates/attempt-{attempt:02d}", repair_source
                )
            else:
                imported = root / "candidates/imported.png"
                source_hash = sha256(candidate)
                raw_copy = root / "candidates/imported.source"
                shutil.copyfile(candidate, raw_copy)
                normalized, normalization = normalize_candidate(
                    raw_copy,
                    tuple(prepared.plan["request_frame"]["size"])
                    if candidate_frame == "request"
                    else (prepared.scene.shape[1], prepared.scene.shape[0]),
                    job.data["generator"],
                )
                normalization["import_frame"] = candidate_frame
                if candidate_frame == "request":
                    normalized = reassemble(
                        prepared.scene, normalized, prepared.plan["request_frame"]
                    )
                save_png(normalized, imported)
                if source_hash != sha256(candidate):
                    raise AcceptanceError("候选在导入时变化，无法绑定可靠的输入哈希")
                generated = [
                    GeneratedCandidate(
                        imported,
                        {
                            "source": "explicit_import",
                            "source_path": str(candidate.resolve()),
                            "source_sha256": source_hash,
                            "archived_source_path": str(raw_copy.relative_to(root)),
                            "normalization": normalization,
                            "model": "not_claimed",
                        },
                    )
                ]
            for source in generated:
                ledger["candidates"] += 1
                cid = f"candidate-{ledger['candidates']:03d}"
                decoded = load_rgb(source.path)
                decoded, registration = register_background(
                    prepared.scene, decoded, job, prepared.masks
                )
                aligned_path = root / f"analysis/{cid}.aligned.png"
                save_png(decoded, aligned_path)
                write_json(root / f"analysis/{cid}.registration.json", registration)
                observation = None
                if job.data.get("candidate_geometry", {}).get("method") == "vlm_seeded_grabcut":
                    if adapter and ledger["vlm_calls"] < job.data["budget"]["max_vlm_calls"]:
                        observation_prompt, observation_paths = observation_request(
                            prepared, aligned_path
                        )
                        write_json(
                            root / f"analysis/{cid}.observation-request.json",
                            {
                                "prompt": observation_prompt,
                                "schema": OBSERVATION_SCHEMA,
                                "inputs": [
                                    {"path": str(p), "sha256": sha256(p)} for p in observation_paths
                                ],
                            },
                        )
                        ledger["vlm_calls"] += 1
                        persist()
                        try:
                            observation = adapter.observe_candidate(prepared, aligned_path)
                        except AdapterError as exc:
                            observation = {
                                "status": "UNKNOWN",
                                "error_type": type(exc).__name__,
                                "contract_diagnostic": str(exc),
                                "reason": "候选观察未成功；未自动重试或伪造轮廓",
                            }
                        write_json(root / f"analysis/{cid}.observation.json", observation)
                geometry = analyze_candidate(
                    prepared.scene, decoded, job, prepared.masks, observation
                )
                grounding = contact_diagnostics(job, prepared.masks, geometry.visible)
                geometry.report["hard_checks"].update(grounding["hard_checks"])
                if any(v["status"] == "FAIL" for v in grounding["hard_checks"].values()):
                    geometry.report["status"] = "RISK"
                elif any(v["status"] == "UNKNOWN" for v in grounding["hard_checks"].values()):
                    geometry.report["status"] = "UNKNOWN"
                geometry.report["support_diagnostics"] = grounding
                for oid, visible in geometry.visible.items():
                    save_png(
                        visible.astype("uint8") * 255,
                        root / f"analysis/{cid}/{oid}.estimated_visible.png",
                    )
                geometry.report["candidate_source_sha256"] = sha256(source.path)
                geometry.report["aligned_candidate_sha256"] = sha256(aligned_path)
                write_json(root / f"analysis/{cid}.geometry.json", geometry.report)
                outside_scope = any(
                    np.any(value & ~prepared.masks.objects[oid].arrays["edit"])
                    for oid, value in geometry.visible.items()
                )
                if geometry.report.get("reject_on_risk") and (
                    len(geometry.visible) != len(job.data["objects"]) or outside_scope
                ):
                    master = decoded.copy()
                    processing = {
                        "operations": [],
                        "status": "SKIPPED_OUTSIDE_EDIT"
                        if outside_scope
                        else "SKIPPED_GEOMETRY_" + geometry.report["status"],
                        "reason": "候选明显偏离或无法可靠分割时保留原始工作画布供诊断；不裁掉主体来制造融合结果",
                    }
                else:
                    try:
                        master, processing = harmonize(
                            prepared.scene,
                            decoded,
                            job,
                            prepared.masks,
                            geometry.visible if job.data.get("candidate_geometry") else None,
                            material_regions(
                                observation,
                                geometry.visible,
                                prepared.scene.shape[:2],
                                {
                                    p["object_id"]: p["id"]
                                    for p in job.data["harmonization"].get(
                                        "surface_adjustments", []
                                    )
                                    if p.get("fill_unassigned_visible")
                                },
                            )
                            if observation
                            else None,
                            registration=registration,
                            diagnostics_dir=root / f"analysis/{cid}.layers",
                        )
                        processing["status"] = "APPLIED"
                    except ConfigError as exc:
                        master = decoded.copy()
                        processing = {
                            "status": "FAILED_PHOTOMETRY_OR_SCOPE",
                            "operations": [],
                            "reason": "可靠光色样本或处理区域不足，未伪造校正",
                            "contract_diagnostic": str(exc),
                        }
                processing["registration"] = registration
                processing["candidate_geometry"] = geometry.report
                path = root / f"outputs/{cid}.png"
                save_png(master, path)
                _verify_inputs(inventory)
                quality = evaluate(prepared, load_rgb(path))
                if job.data.get("feature_flags", {}).get("v3_layer_compositor"):
                    from fusion_engine.controlled.fusion_v3.diagnostics import saved_contract

                    v3_check = saved_contract(prepared.scene, path, prepared.masks, processing)
                    quality["hard_checks"]["v3_saved_layers_contract"] = v3_check
                    if v3_check["status"] != "PASS":
                        quality["status"] = "REJECTED"
                        quality["annotation_ready"] = False
                quality["hard_checks"]["original_canvas_size"] = {
                    "status": "PASS" if load_rgb(path).shape == prepared.scene.shape else "FAIL",
                    "actual_size": list(master.shape[1::-1]),
                    "required_size": list(prepared.scene.shape[1::-1]),
                    "request_frame": prepared.plan["request_frame"]["kind"],
                    "detail": "最终保存PNG保持原图宽高；完整输入与同长宽比显式归一化均有独立记录",
                }
                if geometry.report.get("reject_on_risk"):
                    quality["hard_checks"].update(geometry.report["hard_checks"])
                    quality["candidate_geometry_status"] = geometry.report["status"]
                    if (
                        geometry.report["status"] in ("RISK", "UNKNOWN")
                        or processing["status"] != "APPLIED"
                    ):
                        quality["status"] = "REJECTED"
                        quality["annotation_ready"] = False
                for observed in (observation or {}).get("result", {}).get("objects", []):
                    if any(
                        observed[k] == "FAIL"
                        for k in ("support_status", "identity_status", "occlusion_status")
                    ):
                        quality["status"] = "REJECTED"
                        quality["annotation_ready"] = False
                integrity = None
                if job.data.get("feature_flags", {}).get("v3_structure_preservation"):
                    from fusion_engine.controlled.fusion_v3.shape_evidence import (
                        structure_retention,
                    )

                    integrity = (
                        structure_retention(
                            prepared.scene,
                            decoded,
                            load_rgb(path),
                            geometry.visible,
                            geometry.witnesses,
                            front=prepared.masks.arrays["fg_occ"],
                            max_loss_fraction=job.data.get("fusion_v3", {}).get(
                                "max_structure_loss_fraction", 0.02
                            ),
                        )
                        if processing["status"] == "APPLIED"
                        else {
                            "status": "UNKNOWN",
                            "parts": [],
                            "reason": "未完成局部处理，未将范围失败伪装成后处理结构通过",
                        }
                    )
                    quality["hard_checks"]["postprocess_structure_retention"] = integrity
                    if integrity["status"] != "PASS":
                        quality["status"] = "REJECTED"
                        quality["annotation_ready"] = False
                    write_json(root / f"analysis/{cid}.structure.json", integrity)
                    for oid, values in geometry.witnesses.items():
                        save_png(
                            values.foreground.astype(np.uint8) * 255,
                            root / f"analysis/{cid}/{oid}.structure_witnesses.png",
                        )
                record = {
                    "id": cid,
                    "raw_path": str(source.path.relative_to(root)),
                    "raw_sha256": sha256(source.path),
                    "master_path": str(path.relative_to(root)),
                    "master_sha256": sha256(path),
                    "aligned_path": str(aligned_path.relative_to(root)),
                    "provenance": source.provenance,
                    "processing": processing,
                    "candidate_observation": observation,
                    "quality": quality,
                    "generation_round": attempt if generate else None,
                    "prompt_accumulation": request_record if generate else None,
                }
                if integrity is not None:
                    record["postprocess_integrity"] = integrity
                if adapter and ledger["vlm_calls"] < job.data["budget"]["max_vlm_calls"]:
                    ledger["vlm_calls"] += 1
                    persist()
                    try:
                        opinion = adapter.review(
                            prepared,
                            path,
                            max_feedback_chars=journal.patch_budget
                            if iteration["enabled_for_this_run"]
                            else None,
                        )
                        record["vlm_review"] = opinion
                    except AdapterError as exc:
                        record["vlm_review"] = {
                            "status": "UNKNOWN",
                            "error_type": type(exc).__name__,
                            "contract_diagnostic": profile.redact(str(exc)),
                            "reason": "辅助审阅失败，未自动重试或伪造分数",
                        }
                if job.data.get("feature_flags", {}).get("v3_repair_router"):
                    from fusion_engine.controlled.fusion_v3.repairs import repair_plan

                    decision = repair_plan(
                        job,
                        prepared.masks,
                        geometry.report,
                        processing,
                        record.get("vlm_review"),
                        record["master_sha256"],
                    )
                    record["repair_plan"] = decision
                    decision["role"] = "diagnostic_suggestion_not_iteration_stop_gate"
                    decision["iteration_controller_enabled"] = iteration["enabled_for_this_run"]
                    # The optional selective plan remains an audit proposal. The
                    # round controller keeps all visual feedback, including colour.
                    quality["visual_review_status"] = "REVIEW_REQUIRED"
                    write_json(root / f"analysis/{cid}.repair.json", decision)
                manifest["candidates"].append(record)
                if causal_iteration:
                    from fusion_engine.controlled.fusion_v3.causal import diagnose, select_parent

                    record["repair_diagnosis"] = diagnose(record)
                    selected_parent = select_parent(record, root, selected_parent)
                    record["selected_parent"] = selected_parent
                    repair_source = Path(selected_parent["path"])
                    parent_record = next(
                        c
                        for c in manifest["candidates"]
                        if c["id"] == selected_parent["candidate_id"]
                    )
                    feedback = feedback_from_record(
                        parent_record, journal.patch_budget, source_stage=selected_parent["stage"]
                    )
                    feedback["selected_parent_description"] = (
                        selected_parent["candidate_id"]
                        + "的"
                        + (
                            "配准原始候选"
                            if selected_parent["stage"] == "aligned_raw"
                            else "结构检查通过的母版"
                        )
                        + "（仅参考，未验收）"
                    )
                    feedback["selected_parent"] = selected_parent
                elif (
                    iteration["enabled_for_this_run"]
                    or job.data["generator"].get("refinement_source") == "previous_candidate"
                ):
                    repair_source = path if processing["status"] == "APPLIED" else aligned_path
                    feedback = feedback_from_record(record, journal.patch_budget)
                else:
                    feedback = feedback_from_record(record, journal.patch_budget)
                iteration["rounds"].append(
                    {
                        "round": attempt if generate else None,
                        "candidate_id": cid,
                        "quality_status": quality["status"],
                        "processing_status": processing["status"],
                        "feedback": feedback,
                        "next_input_path": str(repair_source) if repair_source else None,
                        "next_input_sha256": sha256(repair_source) if repair_source else None,
                    }
                )
                write_json(root / f"analysis/{cid}.quality.json", quality)
                write_json(
                    root / f"analysis/{cid}.review-template.json",
                    review_template(job, record["master_sha256"]),
                )
                record_event(
                    root,
                    "S13-S15",
                    quality["status"],
                    candidate_id=cid,
                    master_sha256=record["master_sha256"],
                )
                persist()
            if not generate:
                iteration["stop_reason"] = "IMPORT_ONLY_NO_IMAGE_GENERATION"
                break
            if not policy.enabled or attempt >= iteration["effective_max_rounds"]:
                iteration["stop_reason"] = "ROUND_LIMIT_REACHED"
                break
            if policy.stop_on_visual_accept and may_stop_accepted(record):
                iteration["stop_reason"] = (
                    "VISUAL_MODEL_AND_NUMERICAL_CHECKS_SATISFIED_NOT_HUMAN_ACCEPTANCE"
                )
                break
            if not feedback["usable_model_feedback"] and not policy.continue_on_unknown:
                iteration["stop_reason"] = "NO_USABLE_MODEL_FEEDBACK"
                break
            journal.append_feedback(
                feedback,
                completed_round=attempt,
                background_index=prepared.plan["prompt_document"]["background_image_index"],
            )
            journal.save(root)
            record_event(
                root,
                "iteration",
                "FEEDBACK_APPENDED",
                completed_round=attempt,
                next_round=attempt + 1,
                characters=len(journal.current),
            )
            persist()
        states = [item["quality"]["status"] for item in manifest["candidates"]]
        manifest["status"] = (
            (
                "REJECTED"
                if states and all(state == "REJECTED" for state in states)
                else "CONCEPT_ONLY"
            )
            if job.data["mode"] == "A"
            else "REVIEW_REQUIRED"
            if "REVIEW_REQUIRED" in states
            else "REJECTED"
        )
        manifest["budget_exhausted"] = bool(
            generate and ledger["image_calls"] >= job.data["budget"]["max_image_calls"]
        )
        iteration["completed_generation_rounds"] = len(journal.requests)
        iteration["round_limit_reached"] = generate and attempt >= iteration["effective_max_rounds"]
        record_event(root, "S16", manifest["status"], final_exported=False)
        persist()
        return manifest
    except Exception as exc:
        manifest["iteration"]["stop_reason"] = "REQUEST_OR_PIPELINE_FAILED"
        manifest["status"] = "FAILED"
        manifest["error"] = {
            "type": type(exc).__name__,
            "details": "未记录可能含凭据/签名URL的原始异常",
        }
        record_event(root, "pipeline", "FAILED", error_type=type(exc).__name__)
        persist()
        if isinstance(exc, FusionError):
            raise type(exc)(f"{exc}；运行记录：{root}") from None
        raise FusionError(f"流程失败（{type(exc).__name__}）；运行记录：{root}") from None


def review_template(job: ControlledJob, master_hash: str) -> dict:
    return {
        "schema_version": "2.0",
        "master_sha256": master_hash,
        "reviewer": "待填写实际审核人",
        "notes": "待填写针对当前母版的实际检查记录；模板未代表已审核",
        "checks": {name: {"status": "UNKNOWN", "notes": "尚未检查"} for name in CHECK_NAMES},
        "objects": [
            {"id": item["id"], "mask_matches_master_confirmed": False}
            for item in job.data["objects"]
        ],
    }


def _within(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise AcceptanceError("运行记录包含越界路径")
    return path


def review_run(directory: Path, candidate_id: str, evidence_file: Path) -> dict:
    root = directory.resolve()
    with review_lock(root):
        manifest = read_json(root / "manifest.json")
        if manifest["status"] in ("ACCEPTED", "FAILED", "INPUT_REQUIRED", "CONCEPT_ONLY"):
            raise AcceptanceError("该运行状态不能验收；已接受结果不可覆盖")
        config = root / "job.snapshot.json"
        if sha256(config) != manifest["config_sha256"]:
            raise AcceptanceError("运行配置已被修改，不能使用旧结果验收")
        if (
            runtime_provenance()["engine_source_sha256"]
            != manifest["runtime"]["engine_source_sha256"]
        ):
            raise AcceptanceError("引擎源码/契约已变化，请重新处理候选后验收")
        job = load_job(config)
        _verify_inputs(manifest["input_inventory"])
        prepared = prepare(job)
        record = next(
            (value for value in manifest["candidates"] if value["id"] == candidate_id), None
        )
        if record is None:
            raise AcceptanceError("候选 ID 不存在")
        path = _within(root, record["master_path"])
        if sha256(path) != record["master_sha256"]:
            raise AcceptanceError("PNG 母版已变化；请重新运行处理与验收")
        evidence = load_review(evidence_file, record["master_sha256"])
        attempt_dir = root / "analysis/reviews" / uuid4().hex
        attempt_dir.mkdir(parents=True, exist_ok=False)
        # Freeze the exact supplied contours; all acceptance/export reads use this copy.
        for item in evidence["objects"]:
            for name in ("amodal_mask", "visible_mask"):
                if name in item:
                    descriptor = item[name]
                    mask, _ = read_mask(job, descriptor, prepared.scene.shape[:2])
                    frozen = attempt_dir / f"{item['id']}_{name}.png"
                    save_png(mask.astype("uint8") * 255, frozen)
                    item[name] = {
                        "path": str(frozen),
                        "sha256": sha256(frozen),
                        "evidence": descriptor["evidence"],
                    }
        write_json(attempt_dir / "human_review.json", evidence)
        quality = evaluate(prepared, load_rgb(path), evidence)
        if job.data.get("feature_flags", {}).get("v3_layer_compositor"):
            from fusion_engine.controlled.fusion_v3.diagnostics import saved_contract

            v3_check = saved_contract(
                prepared.scene, path, prepared.masks, record.get("processing", {})
            )
            quality["hard_checks"]["v3_saved_layers_contract"] = v3_check
            if v3_check["status"] != "PASS":
                quality["status"] = "REJECTED"
                quality["annotation_ready"] = False
        geometry = record.get("processing", {}).get("candidate_geometry", {})
        if geometry.get("reject_on_risk") and geometry.get("status") == "RISK":
            quality["hard_checks"].update(geometry["hard_checks"])
            quality["status"] = "REJECTED"
            quality["annotation_ready"] = False
        write_json(attempt_dir / "quality.json", quality)
        record["quality"] = quality
        record.setdefault("reviews", []).append(
            {
                "path": str(attempt_dir.relative_to(root)),
                "reviewed_master_sha256": record["master_sha256"],
                "reviewer": evidence["reviewer"],
                "status": quality["status"],
                "runtime": runtime_provenance(),
            }
        )
        manifest["status"] = quality["status"]
        if quality["status"] == "ACCEPTED":
            final = root / "outputs/final.png"
            if final.exists():
                raise AcceptanceError("final.png 已存在，禁止覆盖")
            _verify_inputs(manifest["input_inventory"])
            if sha256(path) != record["master_sha256"]:
                raise AcceptanceError("验收时母版已变化")
            if quality["annotation_ready"]:
                annotations = export_annotations(prepared, evidence, root / "annotations")
                write_json(root / "annotations/instances.json", annotations)
                manifest["annotations"] = {
                    "status": "EXPORTED_FROM_OBSERVED_MASKS",
                    "path": "annotations/instances.json",
                }
            shutil.copyfile(path, final)
            manifest["selected_candidate_id"] = candidate_id
            manifest["final"] = {
                "path": "outputs/final.png",
                "sha256": sha256(final),
                "format": "PNG",
            }
        record_event(
            root,
            "S15-S16",
            quality["status"],
            candidate_id=candidate_id,
            review_path=str(attempt_dir.relative_to(root)),
        )
        write_json(root / "manifest.json", manifest)
        return manifest
