"""Audited planning, bounded generation, deterministic processing, review and export."""

from __future__ import annotations

import copy
import hashlib
import platform
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

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
from fusion_engine.controlled.job import ControlledJob, load_job, load_review, read_json
from fusion_engine.controlled.masks import placement_preview, read_mask, save_masks
from fusion_engine.controlled.planning import Prepared, prepare
from fusion_engine.controlled.quality import evaluate, export_annotations
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
    for path in sorted([*package_root.rglob("*.py"), *package_root.rglob("*.json")]):
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
    save_png(
        placement_preview(prepared.scene, prepared.job, prepared.masks),
        directory / "placement_preview.png",
    )
    (directory / "README.md").write_text(
        "# 规划预览\n\n绿色框为完整投影 OBB，橙色为编辑区，蓝色为原前景。\n\n"
        "此图不是融合结果，轮廓不是自动标注真值；请查看 placement.json 的证据和缺失项。\n",
        encoding="utf-8",
    )


def run(
    job: ControlledJob,
    *,
    candidate: Path | None = None,
    generate: bool = False,
    provider_config: Path | None = None,
) -> dict:
    if (candidate is None) == (not generate):
        raise ConfigError("请选择 --candidate 或 --generate 中的一种执行方式")
    prepared = prepare(job)
    root = new_run_directory(job.output_root, job.data["task_id"])
    snapshot = copy.deepcopy(job.data)
    snapshot["workspace"] = str(job.workspace)
    write_json(root / "job.snapshot.json", snapshot)
    inventory = _inventory(job)
    write_json(root / "analysis/placement.json", prepared.plan)
    write_json(root / "analysis/input_inventory.json", inventory)
    save_masks(prepared.masks, root / "masks")
    save_png(
        placement_preview(prepared.scene, job, prepared.masks),
        root / "analysis/placement_preview.png",
    )
    (root / "prompts/joint_prompt.txt").write_text(prepared.prompt, encoding="utf-8")
    ledger = {"image_calls": 0, "vlm_calls": 0, "candidates": 0, "budget": job.data["budget"]}
    manifest = {
        "schema_version": "2.0",
        "engine_version": VERSION,
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
        adapter, profile = None, None
        if generate:
            if job.data["generator"]["adapter"] != "bailian":
                raise ConfigError("--generate 只适用于已配置的 bailian 适配器")
            if provider_config is None:
                raise ConfigError("显式传入 --provider-config；不猜测凭据来源")
            profile = load_profile(provider_config)
            adapter = ConfiguredImageAdapter(profile)
            adapter.preflight(prepared)
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
        prompt, attempt = prepared.prompt, 0
        while True:
            _verify_inputs(inventory)
            if generate:
                limit = job.data["budget"]
                batch_size = profile.image_options["max_images"]
                if (
                    ledger["image_calls"] >= limit["max_image_calls"]
                    or ledger["candidates"] + batch_size > limit["max_candidates"]
                ):
                    break
                attempt += 1
                ledger["image_calls"] += 1
                persist()  # Count attempted requests before network I/O; no automatic retry after error.
                record_event(root, "S11", "REQUEST_ATTEMPTED", attempt=attempt)
                (root / f"prompts/attempt-{attempt:02d}.txt").write_text(prompt, encoding="utf-8")
                generated = adapter.generate(
                    prepared, prompt, root / f"candidates/attempt-{attempt:02d}"
                )
            else:
                imported = root / "candidates/imported.png"
                source_hash = sha256(candidate)
                save_png(load_rgb(candidate), imported)
                if source_hash != sha256(candidate):
                    raise AcceptanceError("候选在导入时变化，无法绑定可靠的输入哈希")
                generated = [
                    GeneratedCandidate(
                        imported,
                        {
                            "source": "explicit_import",
                            "source_path": str(candidate.resolve()),
                            "source_sha256": source_hash,
                            "model": "not_claimed",
                        },
                    )
                ]
            repairs = []
            for source in generated:
                ledger["candidates"] += 1
                cid = f"candidate-{ledger['candidates']:03d}"
                master, processing = harmonize(
                    prepared.scene, load_rgb(source.path), job, prepared.masks
                )
                path = root / f"outputs/{cid}.png"
                save_png(master, path)
                _verify_inputs(inventory)
                quality = evaluate(prepared, load_rgb(path))
                record = {
                    "id": cid,
                    "raw_path": str(source.path.relative_to(root)),
                    "raw_sha256": sha256(source.path),
                    "master_path": str(path.relative_to(root)),
                    "master_sha256": sha256(path),
                    "provenance": source.provenance,
                    "processing": processing,
                    "quality": quality,
                }
                if adapter and ledger["vlm_calls"] < job.data["budget"]["max_vlm_calls"]:
                    ledger["vlm_calls"] += 1
                    persist()
                    try:
                        opinion = adapter.review(prepared, path)
                        record["vlm_review"] = opinion
                        if opinion["result"]["recommendation"] == "refine":
                            repairs.append(opinion["result"]["fix_prompt"])
                    except AdapterError as exc:
                        record["vlm_review"] = {
                            "status": "UNKNOWN",
                            "error_type": type(exc).__name__,
                            "reason": "辅助审阅失败，未自动重试或伪造分数",
                        }
                manifest["candidates"].append(record)
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
            if not generate or not repairs:
                break
            # A repair request is a suggestion. Every attempt starts from the original scene.
            prompt = (
                prepared.prompt + "\nVLM建议的最小修复（不是已执行事实）：\n" + "\n".join(repairs)
            )
        states = [item["quality"]["status"] for item in manifest["candidates"]]
        manifest["status"] = (
            "CONCEPT_ONLY"
            if job.data["mode"] == "A"
            else "REVIEW_REQUIRED"
            if "REVIEW_REQUIRED" in states
            else "REJECTED"
        )
        manifest["budget_exhausted"] = bool(
            generate and ledger["image_calls"] >= job.data["budget"]["max_image_calls"]
        )
        record_event(root, "S16", manifest["status"], final_exported=False)
        persist()
        return manifest
    except Exception as exc:
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
