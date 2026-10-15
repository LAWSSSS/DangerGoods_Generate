"""Read the user's YAML directly; only an allowlisted public view enters audit records."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from fusion_engine.errors import ConfigError

WAN_PATH = "/api/v1/services/aigc/multimodal-generation/generation"


@dataclass(frozen=True)
class Credential:
    value: str = field(repr=False)

    def __repr__(self) -> str:
        return "Credential([REDACTED])"


@dataclass(frozen=True)
class ProviderProfile:
    config_path: Path
    image_model: str
    image_protocol: str
    image_endpoint: str
    image_credential: Credential = field(repr=False)
    vlm_model: str
    vlm_endpoint: str
    vlm_credential: Credential = field(repr=False)
    image_options: dict[str, Any]
    execution_options: dict[str, Any]
    legacy_options: dict[str, Any]
    model_choices: dict[str, str]

    def public_view(self) -> dict[str, Any]:
        return {
            "image_model": self.image_model,
            "image_protocol": self.image_protocol,
            "image_endpoint": self.image_endpoint,
            "vlm_model": self.vlm_model,
            "vlm_endpoint": self.vlm_endpoint,
            "credentials_present": {
                "image": bool(self.image_credential.value),
                "vlm": bool(self.vlm_credential.value),
            },
            "image_options": self.image_options,
            "execution_options": self.execution_options,
            "legacy_options": self.legacy_options,
            "model_choices": self.model_choices,
        }

    def public_fingerprint(self) -> str:
        encoded = json.dumps(self.public_view(), ensure_ascii=False, sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    def redact(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self.redact(child) for key, child in value.items()}
        if isinstance(value, list):
            return [self.redact(child) for child in value]
        if isinstance(value, str):
            for credential in (self.image_credential, self.vlm_credential):
                if credential.value:
                    value = value.replace(credential.value, "[REDACTED]")
            return re.sub(r"sk-[A-Za-z0-9_-]{8,}", "[REDACTED]", value)
        return value


def _text(raw: dict, key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"提供者配置字段 {key} 必须是非空文本")
    return value.strip()


def _number(raw: dict, key: str, default: float, low: float, high: float) -> float:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigError(f"提供者配置字段 {key} 必须为有限数值")
    if not low <= value <= high:
        raise ConfigError(f"提供者配置字段 {key} 超出支持范围")
    return value


def _integer(raw: dict, key: str, default: int, low: int, high: int) -> int:
    value = _number(raw, key, default, low, high)
    if not isinstance(value, int):
        raise ConfigError(f"提供者配置字段 {key} 必须为整数")
    return value


def _boolean(raw: dict, key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"提供者配置字段 {key} 必须为布尔值")
    return value


def validate_endpoint(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ConfigError("API 地址必须是 HTTPS，且不包含嵌入凭据或查询参数")
    return value.rstrip("/")


def _model(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,119}", value) or value.startswith("sk-"):
        raise ConfigError("模型名无效")
    return value


def load_profile(path: str | Path) -> ProviderProfile:
    source = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        # Parser exceptions can include raw YAML lines containing credentials.
        raise ConfigError("无法读取提供者 YAML；未输出文件内容") from exc
    if not isinstance(raw, dict):
        raise ConfigError("提供者 YAML 必须是键值映射")
    if raw.get("route") not in (None, ""):
        raise ConfigError("当前按 model/vlm_model 独立绑定；非空 route 尚未实现")
    image_model, vlm_model = _model(_text(raw, "model")), _model(_text(raw, "vlm_model"))
    if image_model.startswith("wan"):
        base = validate_endpoint(_text(raw, "photo_generate_api_url_ali"))
        parsed = urlsplit(base)
        image_endpoint = (
            base
            if parsed.path.endswith(WAN_PATH)
            else f"{parsed.scheme}://{parsed.netloc}{WAN_PATH}"
        )
        image_protocol, image_key = "ali_wan", _text(raw, "api_key_ali")
    elif image_model.startswith("doubao-seedream"):
        image_endpoint = validate_endpoint(_text(raw, "photo_generate_api_url_huoshan"))
        if not image_endpoint.endswith("/images/generations"):
            image_endpoint += "/images/generations"
        image_protocol, image_key = "huoshan_seedream", _text(raw, "api_key_huoshan")
    else:
        raise ConfigError("仅支持配置中的 Wan 或 Seedream 图像模型族")
    if vlm_model.startswith("qwen"):
        vlm_base = validate_endpoint(_text(raw, "vlm_api_ali_url_"))
        vlm_key = _text(raw, "api_key_ali")
    elif vlm_model.startswith("doubao-seed"):
        vlm_base = validate_endpoint(_text(raw, "vlm_api_huoshan_url"))
        vlm_key = _text(raw, "api_key_huoshan")
    else:
        raise ConfigError("仅支持配置中的 Qwen 或 Doubao VLM 模型族")
    vlm_endpoint = (
        vlm_base if vlm_base.endswith("/chat/completions") else vlm_base + "/chat/completions"
    )
    size = raw.get("size", "2K")
    if size not in ("1K", "2K", "4K"):
        raise ConfigError("当前 provider size 支持 1K/2K/4K 档位")
    output_format = raw.get("output_format", "png")
    if output_format != "png":
        raise ConfigError("当前模型接入使用 PNG 母版，output_format 需为 png")
    seed = raw.get("seed")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31
    ):
        raise ConfigError("seed 必须为 null 或非负 32 位整数")
    choices = {}
    for key in (
        "model_ali_high",
        "model_alimax_low",
        "model_huoshan_high",
        "model_huoshan_low",
        "vlm_model_ali_high",
        "vlm_model_ali_low",
        "vlm_model_huoshan",
    ):
        if key in raw:
            choices[key] = _model(_text(raw, key))
    legacy_keys = (
        "object_resize_scale",
        "object_canvas_scale",
        "object_opacity",
        "object_anchor",
        "size_tolerance",
        "backgrounds_dir",
        "objects_dir",
        "prompt_templates_path",
        "negative_prompt_path",
        "output_images_dir",
        "output_final_images_dir",
        "output_metadata_dir",
        "output_failed_dir",
        "output_bbox_dir",
        "output_analysis_dir",
    )
    legacy = {key: raw[key] for key in legacy_keys if key in raw}
    scale = _number(raw, "object_resize_scale", 0.022, 0.000001, 1)
    profile = ProviderProfile(
        source,
        image_model,
        image_protocol,
        image_endpoint,
        Credential(image_key),
        vlm_model,
        vlm_endpoint,
        Credential(vlm_key),
        {
            "size": size,
            "output_format": output_format,
            "watermark": _boolean(raw, "watermark", False),
            "max_images": _integer(raw, "max_images", 1, 1, 8),
            "seed": seed,
        },
        {
            "timeout_seconds": _number(raw, "timeout_seconds", 300, 1, 3600),
            "sleep_seconds": _number(raw, "sleep_seconds", 1, 0, 60),
            "enable_vlm_optimization": _boolean(raw, "enable_vlm_optimization", True),
            "max_optimization_rounds": _integer(raw, "max_optimization_rounds", 3, 0, 20),
            "total_num": _integer(raw, "total_num", 100, 1, 1000000),
            "object_resize_scale": scale,
        },
        legacy,
        choices,
    )
    # Even if a free-text legacy field accidentally repeats a key, its public view must not.
    object.__setattr__(profile, "legacy_options", profile.redact(legacy))
    return profile
