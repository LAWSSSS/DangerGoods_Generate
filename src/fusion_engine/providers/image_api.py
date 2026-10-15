"""Wan native multimodal requests and Seedream flat requests, preserving configured IDs."""

from __future__ import annotations

import base64
import io
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from fusion_engine.errors import AdapterError
from fusion_engine.providers.transport import ProviderTransport


def image_data_url(
    path: Path, max_pixels: int = 4_194_304, min_side: int = 0
) -> tuple[str, dict[str, Any]]:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        original_size = image.size
        if "A" in image.getbands() or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            canvas = Image.new("RGBA", rgba.size, "white")
            canvas.alpha_composite(rgba)
            image = canvas.convert("RGB")
        else:
            image = image.convert("RGB")
        if image.width * image.height > max_pixels:
            ratio = math.sqrt(max_pixels / (image.width * image.height))
            image = image.resize(
                (max(1, int(image.width * ratio)), max(1, int(image.height * ratio))),
                Image.Resampling.LANCZOS,
            )
        if min_side and min(image.size) < min_side:
            ratio = min_side / min(image.size)
            if math.ceil(image.width * ratio) * math.ceil(image.height * ratio) > max_pixels:
                raise AdapterError(
                    "输入过于狭长，达到最短边要求会超出像素上限；请提供已审核的参考裁切"
                )
            image = image.resize(
                (math.ceil(image.width * ratio), math.ceil(image.height * ratio)),
                Image.Resampling.LANCZOS,
            )
        if min_side and (max(image.size) > 8000 or max(image.size) / min(image.size) > 8):
            raise AdapterError("当前 Wan 输入适配只接受尺寸和长宽比符合官方规范的参考图")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(), {
        "path": str(path),
        "original_size": list(original_size),
        "sent_size": list(image.size),
        "encoding": "PNG_RGB8",
        "max_input_pixels": max_pixels,
    }


def extract_image_items(response: dict[str, Any], protocol: str) -> list[dict[str, str]]:
    items = []
    if protocol == "ali_wan":
        for choice in response.get("output", {}).get("choices", []):
            for content in choice.get("message", {}).get("content", []):
                if isinstance(content, dict) and isinstance(content.get("image"), str):
                    items.append({"image": content["image"]})
    else:
        for item in response.get("data", []):
            if isinstance(item, dict):
                if isinstance(item.get("url"), str):
                    items.append({"image": item["url"]})
                elif isinstance(item.get("b64_json"), str):
                    items.append({"base64": item["b64_json"]})
    if not items:
        raise AdapterError("图像 API 未返回有效图片，未保存原始响应内容")
    return items


def decode_output(item: dict[str, str], transport: ProviderTransport, timeout: float) -> bytes:
    try:
        if "base64" in item:
            return base64.b64decode(item["base64"], validate=True)
        value = item["image"]
        if value.startswith("data:image/"):
            return base64.b64decode(value.split(",", 1)[1], validate=True)
        return transport.download_image(value, timeout)
    except (ValueError, IndexError) as exc:
        raise AdapterError("服务图片编码无效") from exc
