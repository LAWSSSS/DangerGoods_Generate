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
        spatial = item.get("spatial_constraints", {})
        interval = spatial.get("occluded_fraction_range")
        if interval and interval[0] > interval[1]:
            raise ConfigError("遮挡面积比例区间必须递增")
        interval = item["placement"]["scale"].get("pixel_length_range")
        if interval and interval[0] > interval[1]:
            raise ConfigError("投影尺寸区间必须递增")
        placement = item["placement"]
        if placement.get("priority") == "support_first":
            flexibility = placement.get("flexibility")
            if (
                data["mode"] != "A"
                or not data["constraints"]["require_support"]
                or "edit_envelope" not in spatial
                or not flexibility
            ):
                raise ConfigError("承托优先摆放需A预览、承托检查、明确编辑包络与微调范围")
            low, high = flexibility["scale_range"]
            if low > 1 or high < 1:
                raise ConfigError("摆放尺度微调范围必须递增且包含初始尺度1")
        for contact in item["support_contacts"]:
            if contact["support_id"] not in set(support_ids + ids) - {item["id"]}:
                raise ConfigError("接触点引用了不存在的支撑 ID 或物体自身")
    for edge in data["scene"]["occlusion_edges"]:
        if edge["front_id"] not in ids or edge["back_id"] not in ids:
            raise ConfigError("遮挡边的两端必须是本任务的物体 ID")
        if edge["front_id"] == edge["back_id"]:
            raise ConfigError("遮挡边不能指向自身")
    harmonization = data["harmonization"]
    feedback = harmonization.get("feedback_correction", {})
    if feedback.get("enabled"):
        selected_ids = feedback["object_ids"]
        if len(set(selected_ids)) != len(selected_ids) or set(selected_ids) - set(ids):
            raise ConfigError("外观反馈的物体ID必须存在且不可重复")
        if feedback["detail_sigma_px"] and harmonization.get("blur_sigma_px", 0):
            raise ConfigError("外观反馈与旧清晰度处理不能叠加；保留一个明确入口")
        if harmonization.get("surface_adjustments") and data.get("fusion_v3", {}).get(
            "surface_policy"
        ) != "paired_only":
            raise ConfigError("外观反馈不能与旧固定表面调色重复叠加")
    generator = data["generator"]
    if generator.get("near_opaque_max_outlier_fraction", 0) and (
        generator.get("alpha_policy") != "near_opaque_unassociated_rgb"
        or "near_opaque_outlier_min_alpha" not in generator
        or generator["near_opaque_outlier_min_alpha"] > generator.get("near_opaque_min_alpha", 251)
    ):
        raise ConfigError("近不透明异常点容差需显式声明RGB假设、合理下限和比例")
    if data["generator"].get("layout_hint") == "procedural_proxy" and (
        data.get("prompt_version") not in ("2.3", "3.0")
        or data["generator"].get("context_crop")
        or data["generator"]["route"] != "guided_generation_then_restore"
    ):
        raise ConfigError("二维落位引导要求Prompt2.3/3.0、完整场景和受控恢复路线")
    flags = data.get("feature_flags", {})
    if flags.get("v3_causal_iteration") and not flags.get("v3_structure_preservation"):
        raise ConfigError("因果迭代需要开启结构保留检查，不能把APPLIED当作结构保真")
    if flags.get("v3_layer_compositor"):
        if (
            data.get("pipeline_version") != "3.0"
            or data["constraints"]["background_protection"] != "strict"
            or not data.get("candidate_geometry")
            or not data.get("registration")
            or not data["generator"].get("preserve_full_frame")
            or data["generator"].get("context_crop")
        ):
            raise ConfigError("V3 分层合成需版本3.0、严格保护、候选轮廓、配准和完整画幅")
        if any(item["render_class"] != "opaque_rigid" for item in data["objects"]):
            raise ConfigError("V3 当前仅实现不透明刚体")
    elif any(flags.values()) or data.get("pipeline_version") == "3.0" or data.get("fusion_v3"):
        raise ConfigError("V3 配置需显式开启 v3_layer_compositor；不允许静默忽略开关")
    if data.get("fusion_v3", {}).get("allow_proposed_interactions") and data["mode"] != "A":
        raise ConfigError("建议性阴影交互只允许在A预览中使用")
    shadow = data.get("fusion_v3", {}).get("shadow", {})
    if shadow.get("min_darkening", 0.02) > shadow.get("max_darkening", 0.22):
        raise ConfigError("V3 最小阴影衰减不能大于最大衰减")
    for key in ("paired_anchors", "surface_adjustments"):
        values = harmonization.get(key, [])
        if len({value["id"] for value in values}) != len(values):
            raise ConfigError("光色控制区域 ID 不得重复")
    if harmonization.get("allow_proposed_anchors") and data["mode"] != "A":
        raise ConfigError("建议性光色区域仅用于A模式预览；B/C需独立审核")
    if (
        data["constraints"]["background_protection"] == "strict"
        and data["generator"]["route"] == "concept"
    ):
        raise ConfigError("严格背景保护需显式使用受控恢复路线")
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
