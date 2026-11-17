"""Structured VLM reports are opinions with evidence, never physical measurements."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from fusion_engine.errors import AdapterError
from fusion_engine.providers.configuration import ProviderProfile
from fusion_engine.providers.image_api import image_data_url
from fusion_engine.providers.transport import ProviderTransport

SCENE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["observations", "hypotheses", "uncertainties"],
    "properties": {
        "observations": {"type": "array", "items": {"type": "string"}},
        "hypotheses": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
}
REVIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["recommendation", "issues", "fix_prompt", "uncertainties"],
    "properties": {
        "recommendation": {"enum": ["accept", "refine", "reject", "unknown"]},
        "issues": {"type": "array", "items": {"type": "string"}},
        "fix_prompt": {"type": "string"},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
}


class VLMClient:
    def __init__(self, profile: ProviderProfile, transport: ProviderTransport | None = None):
        self.profile = profile
        self.transport = transport or ProviderTransport()

    def ask(
        self,
        prompt: str,
        paths: list[Path],
        schema: dict[str, Any],
        *,
        thinking_mode: str | None = None,
        max_output_tokens: int = 2048,
        max_image_pixels: int = 1_048_576,
        discard_unknown_fields: bool = False,
        response_validation_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or not 128 <= max_output_tokens <= 16384
        ):
            raise AdapterError("辅助审阅输出上限必须是128–16384之间的整数")
        if not isinstance(max_image_pixels, int) or not 262144 <= max_image_pixels <= 4194304:
            raise AdapterError("辅助审阅输入像素上限必须为262144–4194304")
        seed2 = bool(re.match(r"^doubao-seed-2[.-]", self.profile.vlm_model))
        if thinking_mode is not None and (
            not seed2 or thinking_mode not in ("enabled", "disabled")
        ):
            raise AdapterError("显式thinking模式目前仅接入Doubao Seed 2系列的enabled/disabled")
        content = [
            {
                "type": "text",
                "text": prompt
                + "\n只输出符合以下 JSON Schema 的对象：\n"
                + json.dumps(schema, ensure_ascii=False),
            }
        ]
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": image_data_url(path, max_pixels=max_image_pixels)[0]},
            }
            for path in paths
        )
        payload = {
            "model": self.profile.vlm_model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.2,
            "stream": False,
        }
        limit_field = "max_completion_tokens" if seed2 else "max_tokens"
        payload[limit_field] = max_output_tokens
        if thinking_mode is not None:
            payload["thinking"] = {"type": thinking_mode}
        response = self.transport.post_json(
            self.profile.vlm_endpoint,
            payload,
            self.profile.vlm_credential,
            self.profile.execution_options["timeout_seconds"],
        )
        try:
            finish_reason = response["choices"][0].get("finish_reason")
            if finish_reason == "length":
                raise AdapterError("辅助审阅达到输出长度上限；未将截断JSON作为有效审阅")
            text = response["choices"][0]["message"]["content"]
            if isinstance(text, list):
                text = "".join(item.get("text", "") for item in text if isinstance(item, dict))
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
            result = json.loads(text)
        except (KeyError, IndexError, TypeError, AttributeError, ValueError) as exc:
            raise AdapterError("VLM 没有返回结构化 JSON，未输出原始内容") from exc
        discarded = []

        def known_fields(value, contract, location="root"):
            if isinstance(value, dict) and contract.get("type") == "object":
                properties = contract.get("properties", {})
                kept = {}
                for key, child in value.items():
                    if key in properties:
                        kept[key] = known_fields(child, properties[key], location + "." + key)
                    elif contract.get("additionalProperties") is not False:
                        kept[key] = child
                    else:
                        discarded.append(location + "." + key)
                return kept
            if isinstance(value, list) and contract.get("type") == "array":
                return [
                    known_fields(child, contract.get("items", {}), location + f".{index}")
                    for index, child in enumerate(value)
                ]
            return value

        if discard_unknown_fields:
            result = known_fields(result, schema)
        # Some callers validate independent entries separately to preserve usable
        # review text when a single localization is malformed. The requested
        # schema above stays strict; this does not repair or guess model output.
        validation_schema = response_validation_schema or schema
        errors = sorted(
            Draft202012Validator(validation_schema).iter_errors(result),
            key=lambda e: str(e.path),
        )
        if errors:
            location = ".".join(map(str, errors[0].absolute_path)) or "root"
            missing = (
                [key for key in errors[0].validator_value if key not in errors[0].instance]
                if errors[0].validator == "required"
                else []
            )
            raise AdapterError(
                f"VLM JSON 字段 {location} 不符合契约（{errors[0].validator}，缺失{missing}）；未输出原始响应"
            )
        return self.profile.redact(
            {
                "result": result,
                "provider_request_id": response.get("id") or response.get("request_id"),
                "model_requested": self.profile.vlm_model,
                "model_returned": response.get("model"),
                "request_options": {
                    key: value for key, value in payload.items() if key not in ("messages", "model")
                },
                "finish_reason": finish_reason,
                "max_input_image_pixels": max_image_pixels,
                "discarded_unknown_fields": discarded,
            }
        )
