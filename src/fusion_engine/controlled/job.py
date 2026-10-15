"""Separate read-only assets from bounded, exclusively created run directories."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from fusion_engine.artifacts import sha256
from fusion_engine.config import reject_secrets
from fusion_engine.controlled.schema import CHECK_NAMES, job_schema, review_schema
from fusion_engine.errors import ConfigError


@dataclass(frozen=True)
class ControlledJob:
    data: dict[str, Any]
    config_path: Path
    workspace: Path

    def asset(self, value: str) -> Path:
        path = Path(value).expanduser()
        return (path if path.is_absolute() else self.workspace / path).resolve()

    @property
    def output_root(self) -> Path:
        value = Path(self.data["output_dir"])
        path = (self.workspace / value).resolve()
        # Existing input data, configs and code can never become a run directory.
        allowed = [(self.workspace / folder).resolve() for folder in ("output", "work")]
        if (
            value.is_absolute()
            or not path.is_relative_to(self.workspace)
            or not any(path.is_relative_to(root) for root in allowed)
        ):
            raise ConfigError("output_dir 必须是 workspace 内 output/ 或 work/ 下的相对路径")
        if any(root.is_symlink() for root in (self.workspace / "output", self.workspace / "work")):
            raise ConfigError("输出根目录不能是符号链接")
        return path


def validate_document(data: Any, schema: dict, label: str) -> None:
    reject_secrets(data, label)
    errors = sorted(Draft202012Validator(schema).iter_errors(data), key=lambda e: str(e.path))
    if errors:
        error = errors[0]
        # Never interpolate an offending value (it could be a credential).
        location = ".".join(map(str, error.absolute_path)) or "root"
        raise ConfigError(f"{label} 字段 {location} 不符合契约（{error.validator}）")


def read_json(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ConfigError("无法读取 JSON 文件，未输出文件内容") from exc


def load_job(path: str | Path) -> ControlledJob:
    source = Path(path).expanduser().resolve()
    data = read_json(source)
    validate_document(data, job_schema(), "job")
    workspace = (source.parent / data["workspace"]).resolve()
    if not workspace.is_dir():
        raise ConfigError("workspace 不存在")
    job = ControlledJob(copy.deepcopy(data), source, workspace)
    job.output_root
    ids = [item["id"] for item in data["objects"]]
    support_ids = [item["id"] for item in data["scene"]["support_regions"]]
    if len(set(ids + support_ids)) != len(ids + support_ids):
        raise ConfigError("物体 ID 与支撑区域 ID 必须全局唯一")
    for item in data["objects"]:
        if len({part["id"] for part in item["material_parts"]}) != len(item["material_parts"]):
            raise ConfigError("同一物体的部件 ID 不可重复")
        dimensions = item["dimensions"]
        if dimensions["status"] in ("measured", "catalog_verified"):
            if not evidence_confirmed(dimensions["evidence"], allow_catalog=True):
                raise ConfigError("实测/已核验尺寸必须带实际审核人及证据；估计值请标 estimated")
        for interval in dimensions.get("ranges", {}).values():
            if interval[0] > interval[1]:
                raise ConfigError("物体尺寸区间必须递增")
        interval = item["placement"]["scale"].get("pixel_length_range")
        if interval and interval[0] > interval[1]:
            raise ConfigError("投影尺寸区间必须递增")
        for contact in item["support_contacts"]:
            if contact["support_id"] not in set(support_ids + ids) - {item["id"]}:
                raise ConfigError("接触点引用了不存在的支撑 ID 或物体自身")
    for edge in data["scene"]["occlusion_edges"]:
        if edge["front_id"] not in ids or edge["back_id"] not in ids:
            raise ConfigError("遮挡边的两端必须是本任务的物体 ID")
        if edge["front_id"] == edge["back_id"]:
            raise ConfigError("遮挡边不能指向自身")
    if data["mode"] in ("B", "C"):
        if data["constraints"]["background_protection"] != "strict":
            raise ConfigError("模式 B/C 必须启用严格背景保护")
        if data["generator"]["route"] == "concept":
            raise ConfigError("模式 B/C 不允许静默采用概念生成路线")
        if not set(CHECK_NAMES).issubset(data["evaluation"]["required_visual_checks"]):
            raise ConfigError("B/C 必须保留身份、支撑、遮挡、材质、光照、风格、数量和真实感验收项")
    if data["mode"] in ("B", "C") and not data["evaluation"]["require_observed_masks"]:
        raise ConfigError("当前 B/C 验收实现要求最终图的独立轮廓证据；不能靠计划 Mask 证明实际几何")
    return job


def evidence_confirmed(value: dict | None, *, allow_catalog: bool = False) -> bool:
    sources = {"human", "cad", "calibrated_geometry"} | ({"catalog"} if allow_catalog else set())
    return bool(
        value
        and value.get("confirmed")
        and value.get("reviewer", "").strip()
        and value.get("source") in sources
    )


def asset_record(job: ControlledJob, descriptor: dict, *, image: bool = False) -> dict:
    path = job.asset(descriptor["path"])
    if not path.is_file():
        raise ConfigError(f"输入文件不存在：{path.name}")
    digest = sha256(path)
    if descriptor.get("sha256") and descriptor["sha256"] != digest:
        raise ConfigError(f"输入哈希不匹配：{path.name}")
    result = {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}
    if image:
        from fusion_engine.images import image_info

        result.update(image_info(path))
    return result


def load_review(path: Path, master_hash: str) -> dict:
    data = read_json(path)
    validate_document(data, review_schema(), "review")
    if data["master_sha256"] != master_hash:
        raise ConfigError("验收报告未绑定当前 PNG 母版哈希，必须重新验收")
    if not data["reviewer"].strip() or not data["notes"].strip():
        raise ConfigError("人工验收必须填写实际审核人和检查记录")
    if any(not value["notes"].strip() for value in data["checks"].values()):
        raise ConfigError("每个人工检查项必须填写依据，不能只填空白")
    ids = [item["id"] for item in data["objects"]]
    if len(ids) != len(set(ids)):
        raise ConfigError("验收报告包含重复物体 ID")
    return data
