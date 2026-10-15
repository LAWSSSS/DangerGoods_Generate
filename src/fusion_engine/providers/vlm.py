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

    def ask(self, prompt: str, paths: list[Path], schema: dict[str, Any]) -> dict[str, Any]:
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
                "image_url": {"url": image_data_url(path, max_pixels=1_048_576)[0]},
            }
            for path in paths
        )
        payload = {
            "model": self.profile.vlm_model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.2,
            "max_tokens": 2048,
            "stream": False,
        }
        response = self.transport.post_json(
            self.profile.vlm_endpoint,
            payload,
            self.profile.vlm_credential,
            self.profile.execution_options["timeout_seconds"],
        )
        try:
            text = response["choices"][0]["message"]["content"]
            if isinstance(text, list):
                text = "".join(item.get("text", "") for item in text if isinstance(item, dict))
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
            result = json.loads(text)
        except (KeyError, IndexError, TypeError, AttributeError, ValueError) as exc:
            raise AdapterError("VLM 没有返回结构化 JSON，未输出原始内容") from exc
        if list(Draft202012Validator(schema).iter_errors(result)):
            raise AdapterError("VLM JSON 不符合场景/评估契约")
        return self.profile.redact(
            {
                "result": result,
                "provider_request_id": response.get("id") or response.get("request_id"),
                "model_requested": self.profile.vlm_model,
                "model_returned": response.get("model"),
                "request_options": {"temperature": 0.2, "max_tokens": 2048, "stream": False},
            }
        )
