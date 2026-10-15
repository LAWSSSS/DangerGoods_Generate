"""Reject secrets and nonfinite numbers in public configuration/audit records."""

import math
import re
from typing import Any

from fusion_engine.errors import ConfigError


def reject_secrets(value: Any, location: str = "config") -> None:
    """Configs and request snapshots are public metadata, never credential stores."""
    forbidden = {
        "api_key",
        "apikey",
        "authorization",
        "access_token",
        "token",
        "password",
        "secret",
    }
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in forbidden or re.match(
                r"^(api_?key|password|secret|access_token)(_|$)", key.lower()
            ):
                raise ConfigError(f"{location}.{key} 不得保存密钥；请使用外部私有提供者配置")
            reject_secrets(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_secrets(child, f"{location}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ConfigError(f"{location} 必须是有限数值")
    elif isinstance(value, str) and re.search(r"sk-[A-Za-z0-9_-]{12,}", value):
        raise ConfigError("公共配置/记录疑似含有明文凭据，请改用外部私有配置")
