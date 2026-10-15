"""TLS-verified HTTP; credentials only enter an API request's Authorization header."""

import re
from typing import Any
from urllib.parse import urlsplit

import httpx

from fusion_engine.errors import AdapterError
from fusion_engine.providers.configuration import Credential, validate_endpoint


class ProviderTransport:
    def __init__(self, client: httpx.Client | None = None):
        self.client = client

    def post_json(
        self, endpoint: str, payload: dict[str, Any], credential: Credential, timeout: float
    ) -> dict[str, Any]:
        validate_endpoint(endpoint)
        try:
            if self.client is not None:
                response = self.client.post(
                    endpoint,
                    json=payload,
                    headers={"Authorization": f"Bearer {credential.value}"},
                    timeout=timeout,
                    follow_redirects=False,
                )
            else:
                with httpx.Client(timeout=timeout, follow_redirects=False) as client:
                    response = client.post(
                        endpoint,
                        json=payload,
                        headers={"Authorization": f"Bearer {credential.value}"},
                    )
        except httpx.HTTPError as exc:
            raise AdapterError(f"API 网络失败（{type(exc).__name__}），未自动重试") from exc
        if response.status_code >= 300:
            # Never print response.text, request headers, or credential-bearing exceptions.
            code = None
            try:
                error = response.json().get("error", {})
                raw_code = error.get("code") if isinstance(error, dict) else None
                if isinstance(raw_code, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", raw_code):
                    if credential.value not in raw_code and not raw_code.startswith("sk-"):
                        code = raw_code
            except (ValueError, AttributeError):
                pass
            raise AdapterError(
                f"API 返回 HTTP {response.status_code}" + (f"（{code}）" if code else "")
            )
        try:
            result = response.json()
        except ValueError as exc:
            raise AdapterError("API 返回的内容不是 JSON") from exc
        if not isinstance(result, dict):
            raise AdapterError("API 返回的 JSON 不是对象")
        return result

    def download_image(self, url: str, timeout: float) -> bytes:
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise AdapterError("服务返回的图片地址不是有效 HTTPS URL")
        try:
            if self.client is not None:
                response = self.client.get(url, timeout=timeout, follow_redirects=True)
            else:
                with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                    response = client.get(url)
            response.raise_for_status()
            if len(response.content) > 40 * 1024 * 1024:
                raise AdapterError("返回图片超出 40 MiB 限制")
            return response.content
        except httpx.HTTPError as exc:
            raise AdapterError(f"图片下载失败（{type(exc).__name__}），未输出签名 URL") from exc
