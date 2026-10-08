#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异物六视图生成：wan2.7-image-pro，原图参考，一张 2K 3:2 六宫格。
默认每次只生成下一张未完成的，批准后再跑同一命令。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import ssl
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
import urllib3
import yaml
from PIL import Image

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
requests.packages.urllib3.disable_warnings()
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["CURL_CA_BUNDLE"] = ""

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MODEL_NAME = "wan2.7-image"
OUTPUT_SIZE = "2508*1672"  # 3:2, under 2K total-pixel cap (2048*2048)
PREFERRED_FIRST_STEM = "灭火器"
WAN_PROMPT_MAX = 5000
VIEW_ORDER = ("front", "back", "left", "right", "top", "bottom")
VIEW_LABELS = {
    "front": "前",
    "back": "后",
    "left": "左",
    "right": "右",
    "top": "上",
    "bottom": "下",
}

CONFIRMED_PROMPTS = {
    "灭火器": (
        "参考图是 GLORIA 6kg ABC 手提式灭火器的左前方 3/4 照片。把它当成焊死刚体，"
        "生成同一物体的正交六视图，一张图 2 行 3 列。浅灰分隔。格子左下角只标：前、后、左、右、上、下。"
        "禁止英文。纯白背景，正交无透视。\n\n"
        "【焊死方位，六个格子必须同一套】\n"
        "前=GLORIA标签面。后=标签背面。左=软管侧。右=软管对面。\n"
        "握把方向（物体空间）：从阀头指向后方（标签反面）。红压把在上略上扬，黑握把在下略平。"
        "不是朝左，不是朝右。参考图朝右上只是透视。\n"
        "喷口位置：永远在底座左侧卡扣。喷口方向：开口永远朝下（指向地面）。\n"
        "筒径约150mm，总高约500–550mm，外包络约270–300×160–165mm。\n\n"
        "【前】标签正对。\n"
        "握把方向：从阀头指向后方=背离镜头伸进深处。只画阀头铰链短端面。"
        "禁止朝画面左/右画出长压把，禁止朝镜头伸。\n"
        "喷口方向：喷口在画面左下、底座左缘，开口朝下。软管从阀头向左再向下接到喷口。"
        "禁止喷口在画面右。\n\n"
        "【后】无正面大标签。\n"
        "握把方向：从阀头指向后方=正对镜头朝观察者伸出，红上黑下，尖端对着镜头。"
        "禁止朝画面左或右撇。\n"
        "喷口方向：喷口在画面右下（物体左侧，后视左右对调），开口朝下。软管沿画面右缘下行。"
        "禁止喷口在画面左。\n\n"
        "【左】画面右=前方，画面左=后方。软管贴近侧筒壁中央下垂。\n"
        "握把方向：从阀头指向后方=整根朝画面左伸出，略上扬，红上黑下。"
        "禁止朝画面右，禁止缩成朝镜头短头。\n"
        "喷口方向：喷口在近处画面左下，开口朝下。禁止喷口在画面右。\n\n"
        "【右】画面左=前方，画面右=后方。禁止画成前视图。\n"
        "握把方向：从阀头指向后方=整根朝画面右伸出，略上扬，红上黑下。禁止朝画面左。\n"
        "喷口方向：喷口在远端被筒体挡住，开口仍朝下。禁止喷口或软管出现在画面左近处。\n\n"
        "【上】俯视。画面下=前，画面上=后，画面左=左。\n"
        "握把方向：从阀头指向后方=指向画面正上方。禁止指向左、右、下。\n"
        "喷口方向：喷口在画面左下（左前底座），开口朝下，俯视见卡扣圆截面。软管在左。"
        "禁止喷口在右或上。\n\n"
        "【下】仰视黑底座底面。画面下=前，画面上=后，画面左=左。\n"
        "握把方向：从阀头指向后方=指向画面正上方，只在圆外上方露出把尖。禁止出现在左、右、下。\n"
        "喷口方向：喷口在画面左侧卡扣，开口朝下=本画面朝向镜头。禁止喷口在右。"
    ),
}

CHINESE_CLASS_KEYS = [
    "分离过滤器",
    "密封胶罐",
    "自喷漆罐",
    "杀虫剂罐",
    "液压缸",
    "千斤顶",
    "轴承座",
    "集油器",
    "灭火器",
    "储罐",
    "油缸",
    "气缸",
    "气瓶",
    "球阀",
    "减震器",
]


class SSLAdapter(requests.adapters.HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


def get_session_with_ssl_bypass() -> requests.Session:
    session = requests.Session()
    adapter = SSLAdapter()
    session.mount("https://", adapter)
    session.verify = False
    retry_strategy = urllib3.Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"],
    )
    retry_adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", retry_adapter)
    session.mount("https://", retry_adapter)
    return session


def load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    return yaml.safe_load(text) or {}


def origin_of(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"无法解析地址: {url}")
    return f"{parsed.scheme}://{parsed.netloc}"


def load_api_config(base_dir: Path) -> Dict[str, Any]:
    raw = load_yaml_config(base_dir / "config_bailian.yaml")
    api_key = raw.get("api_key_ali") or ""
    image_base = raw.get("photo_generate_api_url_ali") or ""
    if not api_key:
        raise ValueError("config_bailian.yaml 未配置 api_key_ali")
    if not image_base:
        raise ValueError("config_bailian.yaml 未配置 photo_generate_api_url_ali")
    timeout_seconds = int(raw.get("timeout_seconds", 300))
    return {
        "api_key": api_key,
        "api_url": origin_of(image_base) + "/api/v1/services/aigc/multimodal-generation/generation",
        "timeout_seconds": timeout_seconds,
        "watermark": bool(raw.get("watermark", False)),
    }


def match_object_class(stem: str, prompt_templates: Dict[str, List[str]]) -> str:
    chinese_keys = [k for k in CHINESE_CLASS_KEYS if k in prompt_templates]
    chinese_keys.sort(key=len, reverse=True)
    for cls in chinese_keys:
        if cls in stem:
            return cls
    for cls in sorted(prompt_templates.keys(), key=len, reverse=True):
        if cls in stem:
            return cls
        variants = (cls, cls.replace("_", ""), cls.replace("_", "-"))
        stem_l = stem.lower()
        for variant in variants:
            if variant.lower() in stem_l:
                return cls
    return stem


def list_object_images(objects_dir: Path) -> List[Path]:
    if not objects_dir.exists():
        raise FileNotFoundError(f"找不到异物目录: {objects_dir}")
    images = [
        p for p in objects_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(images, key=lambda p: p.name)


def load_size_description(base_dir: Path, stem: str) -> str:
    path = base_dir / "尺寸描述" / f"{stem}_尺寸描述.md"
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    if len(text) > 1800:
        return text[:1800] + "\n…"
    return text


def encode_reference_image(path: Path) -> str:
    img = Image.open(path)
    w, h = img.size
    total_pixels = w * h
    max_pixels = 4194304
    if total_pixels > max_pixels:
        ratio = (max_pixels / total_pixels) ** 0.5
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        new_w += new_w % 2
        new_h += new_h % 2
        print(f"  [压缩] 参考图从 {w}x{h} 压到 {new_w}x{new_h}")
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=92)
    b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_data}"


def landmark_block(object_class: str) -> str:
    return (
        "\n【本物体零件方位，必须遵守】\n"
        "先根据参考图锁定：哪一侧有喷口/管嘴/软管，哪一侧有握把/手柄/扳手，标签朝哪面。"
        "这些位置在六个格子里必须是同一套刚体关系，只随相机方位改变可见面。"
        "握把若指向标签反面（后方），前视后视只画端面，左视右视俯视才画全长。\n"
    )


def approved_prompt_paths(output_root: Path, object_class: str, stem: str) -> List[Path]:
    return [
        output_root / stem / "六视图prompt.json",
        output_root / object_class / "六视图prompt.json",
    ]


KEY_LANDMARK_WORDS = ("喷口", "喷嘴", "软管", "握把", "压把", "接头", "固定座")


def format_landmark_dirs(raw: Any) -> str:
    if isinstance(raw, dict):
        items = [(str(k), str(v)) for k, v in raw.items() if v]
    elif isinstance(raw, list):
        items = [
            (str(item.get("name") or ""), str(item.get("screen_dir") or ""))
            for item in raw
            if isinstance(item, dict)
        ]
    else:
        return ""
    filtered = [
        (name, direction)
        for name, direction in items
        if name and direction and any(word in name for word in KEY_LANDMARK_WORDS)
    ]
    if not filtered:
        filtered = [(name, direction) for name, direction in items if name and direction]
    return " ".join(f"{name}画面方向：{direction}" for name, direction in filtered)


def compact_approved_prompt(data: Dict[str, Any]) -> str:
    body_frame = data.get("body_frame") or {}
    views = data.get("views") or {}
    name = data.get("object_name") or ""
    camera = str(data.get("reference_camera") or "").strip()
    frame_key = {
        "front": "front",
        "back": "back",
        "left": "left",
        "right": "right",
        "top": "up",
        "bottom": "down",
    }
    lines = [
        f"参考图是{name}的实拍。参考相机：{camera}" if camera else f"参考图是{name}的实拍。",
        "把它当成焊死刚体，生成同一物体的正交六视图，一张图 2 行 3 列。浅灰分隔。",
        "格子左下角只标一个汉字：前、后、左、右、上、下。禁止英文。纯白背景，正交无透视。",
        "【焊死方位】",
    ]
    for key in VIEW_ORDER:
        text = str(body_frame.get(key) or body_frame.get(frame_key[key]) or "").strip()
        if text:
            lines.append(f"{VIEW_LABELS[key]}：{text}")
    lines.append("【六个视图，必须遵守各自喷口方向和握把方向】")
    for key in VIEW_ORDER:
        view = views.get(key) or {}
        prompt = str(view.get("prompt") or "").strip()
        extra = format_landmark_dirs(view.get("landmark_screen_dirs"))
        forbidden = str(view.get("forbidden") or "").strip()
        block = " ".join(part for part in (prompt, extra, forbidden) if part)
        if block:
            lines.append(f"【{VIEW_LABELS[key]}】{block}")
    return "\n".join(lines)


def load_approved_prompt(output_root: Path, object_class: str, stem: str) -> Optional[str]:
    for path in approved_prompt_paths(output_root, object_class, stem):
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        stitched = str(data.get("stitched_prompt") or "").strip()
        prompt = stitched if 0 < len(stitched) <= WAN_PROMPT_MAX else compact_approved_prompt(data)
        if len(prompt) > WAN_PROMPT_MAX:
            raise RuntimeError(f"已确认prompt超过{WAN_PROMPT_MAX}字: {path}")
        if prompt:
            print(f"  使用已确认prompt: {path} ({len(prompt)}字)")
            return prompt
    return None


def build_prompt(
    object_class: str,
    stem: str,
    size_text: str,
    output_root: Optional[Path] = None,
) -> str:
    if output_root is not None:
        approved = load_approved_prompt(output_root, object_class, stem)
        if approved:
            return approved
    confirmed = CONFIRMED_PROMPTS.get(stem) or CONFIRMED_PROMPTS.get(object_class)
    if confirmed:
        return confirmed
    size_block = ""
    if size_text:
        size_block = f"\n【真实尺寸，必须保持长宽高比例】\n{size_text}\n"
    return (
        f"参考图是一张真实的{object_class}照片（文件名：{stem}）。"
        f"请根据这张参考图，生成同一个{object_class}的正交产品六视图，全部放在一张图片里。\n\n"
        "【布局，必须严格遵守】\n"
        "整张图为横向 2 行 × 3 列均匀六宫格，浅灰细线分隔：\n"
        "第一行从左到右：前视图、后视图、左视图\n"
        "第二行从左到右：右视图、俯视图、仰视图\n"
        "每个格子左下角只用一个汉字标注：前、后、左、右、上、下。禁止英文。\n\n"
        "【刚体旋转，最重要】\n"
        "物体本身不变形、不重装。喷口、软管、管嘴、握把、手柄、阀门、标签、支架"
        "全部焊死/装死在物体上的固定位置，不会从左侧跳到右侧，也不会每格重新发明一套零件布局。\n"
        "六个格子是同一件静止物体，只有相机在绕它做正交观察（等价于转盘每次转 90°）：\n"
        "前 = 与参考图同一朝向（把参考图的透视压成正视）；"
        "后 = 绕竖直轴转 180°；"
        "左 = 从物体自身左侧正视；"
        "右 = 从物体自身右侧正视；"
        "上 = 正上方俯视；"
        "下 = 正下方仰视。\n"
        "相邻格子必须能对上：例如前视里在左边的喷口/软管，转到后视时必须出现在右边；"
        "握把指向也要跟着转，禁止六个格子都把握把画成同一方向。\n"
        f"{landmark_block(object_class)}"
        "【外观】\n"
        f"六个格子必须是同一个{object_class}，颜色、材质、配件、标签文字、磨损必须与参考图一致。\n"
        "必须严格保持原始长宽高比例和轮廓，禁止拉伸、扭曲、增删零件。\n"
        f"{size_block}\n"
        "【拍摄要求】\n"
        "纯白无缝背景，正交产品摄影，焦距很长、几乎无透视。\n"
        "每个格子里物体完整、居中、大小一致。\n"
        "禁止场景、人手、废钢堆、英文乱码、水印、额外道具。"
    )


def extract_image_urls(result: Dict[str, Any]) -> List[str]:
    urls: List[str] = []
    output = result.get("output", {})
    for choice in output.get("choices", []):
        message = choice.get("message", {})
        for item in message.get("content", []):
            if isinstance(item, dict) and item.get("image"):
                urls.append(item["image"])
    if not urls:
        debug_path = Path("debug_response.json")
        debug_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        raise ValueError(f"无法从响应中提取图片 URL，已写入 {debug_path}")
    return urls


def call_image_api(payload: Dict[str, Any], api_url: str, api_key: str, timeout_seconds: int) -> Dict[str, Any]:
    print(f"  [API] 模型: {payload.get('model')}")
    print(f"  [API] 端点: {api_url}")
    print(f"  [API] size: {payload.get('parameters', {}).get('size')}")
    session = get_session_with_ssl_bypass()
    response = session.post(
        api_url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout_seconds,
    )
    if response.status_code != 200:
        print(f"  [API Error] 状态码: {response.status_code}")
        print(f"  [API Error] 响应: {response.text[:2000]}")
        response.raise_for_status()
    result = response.json()
    print(f"  [API] 响应键: {list(result.keys())}")
    return result


def download_image(url: str) -> bytes:
    session = get_session_with_ssl_bypass()
    response = session.get(url, timeout=60)
    response.raise_for_status()
    return response.content


def output_paths(output_root: Path, object_class: str, stem: str) -> Tuple[Path, Path]:
    out_dir = output_root / object_class
    png_path = out_dir / f"{stem}_6view.png"
    json_path = out_dir / f"{stem}_6view.json"
    return png_path, json_path


def is_done(output_root: Path, object_class: str, stem: str) -> bool:
    png_path, _ = output_paths(output_root, object_class, stem)
    return png_path.exists()


def build_job_list(
    objects_dir: Path,
    output_root: Path,
    prompt_templates: Dict[str, List[str]],
) -> List[Dict[str, str]]:
    jobs: List[Dict[str, str]] = []
    for path in list_object_images(objects_dir):
        stem = path.stem
        object_class = match_object_class(stem, prompt_templates)
        jobs.append({
            "path": str(path),
            "stem": stem,
            "class": object_class,
            "done": is_done(output_root, object_class, stem),
        })
    return jobs


def pick_job(
    jobs: List[Dict[str, str]],
    stem_filter: Optional[str],
    prefer_first_stem: str = PREFERRED_FIRST_STEM,
    force: bool = False,
) -> Dict[str, str]:
    pending = [j for j in jobs if not j["done"]]
    if stem_filter:
        exact = [j for j in jobs if j["stem"] == stem_filter]
        matches = exact or [j for j in jobs if stem_filter in j["stem"]]
        if not matches:
            raise FileNotFoundError(f"找不到 stem={stem_filter} 的异物图")
        chosen = matches[0]
        if chosen["done"] and not force:
            raise FileExistsError(f"已存在成品: {chosen['stem']}。重出来用 --force")
        return chosen
    if force and prefer_first_stem:
        preferred_all = [j for j in jobs if j["stem"] == prefer_first_stem]
        if preferred_all:
            return preferred_all[0]
    if not pending:
        raise RuntimeError("全部异物都已生成六视图")
    preferred = [j for j in pending if j["stem"] == prefer_first_stem]
    if preferred:
        return preferred[0]
    return pending[0]


def generate_one(
    job: Dict[str, str],
    base_dir: Path,
    output_root: Path,
    api_cfg: Dict[str, Any],
) -> Path:
    object_path = Path(job["path"])
    stem = job["stem"]
    object_class = job["class"]
    png_path, json_path = output_paths(output_root, object_class, stem)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    size_text = load_size_description(base_dir, stem)
    prompt = build_prompt(object_class, stem, size_text, output_root=output_root)
    data_url = encode_reference_image(object_path)

    payload = {
        "model": MODEL_NAME,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image": data_url},
                        {"text": prompt},
                    ],
                }
            ]
        },
        "parameters": {
            "size": OUTPUT_SIZE,
            "n": 1,
            "watermark": api_cfg["watermark"],
        },
    }

    print(f"\n===== 生成六视图 =====")
    print(f"  源图: {object_path}")
    print(f"  类别: {object_class}")
    print(f"  输出: {png_path}")
    print(f"  尺寸描述: {'有' if size_text else '无'}")

    result = call_image_api(
        payload,
        api_cfg["api_url"],
        api_cfg["api_key"],
        api_cfg["timeout_seconds"],
    )
    urls = extract_image_urls(result)
    image_bytes = download_image(urls[0])
    png_path.write_bytes(image_bytes)

    metadata = {
        "stem": stem,
        "object_class": object_class,
        "source_image": str(object_path),
        "output_image": str(png_path),
        "model": MODEL_NAME,
        "size": OUTPUT_SIZE,
        "prompt": prompt,
        "size_description_used": bool(size_text),
        "image_url": urls[0],
        "usage": result.get("usage"),
        "request_id": result.get("request_id"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  已保存: {png_path}")
    print(f"  元数据: {json_path}")
    return png_path


def main() -> None:
    parser = argparse.ArgumentParser(description="为 input/objects 生成 wan2.7-image 六视图")
    parser.add_argument("--once", action="store_true", default=True, help="只生成下一张（默认）")
    parser.add_argument("--all", action="store_true", help="连续生成全部未完成（需明确指定）")
    parser.add_argument("--stem", type=str, default=None, help="指定异物文件名（不含后缀），例如 灭火器")
    parser.add_argument("--force", action="store_true", help="覆盖已有成品后重出")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    objects_dir = base_dir / "input" / "objects"
    output_root = base_dir / "output" / "six_views"
    templates_path = base_dir / "prompts" / "prompt_templates.json"
    prompt_templates = json.loads(templates_path.read_text(encoding="utf-8")) if templates_path.exists() else {}

    api_cfg = load_api_config(base_dir)
    jobs = build_job_list(objects_dir, output_root, prompt_templates)
    pending = [j for j in jobs if not j["done"]]
    print(f"异物总数: {len(jobs)}，已完成: {len(jobs) - len(pending)}，待生成: {len(pending)}")

    if args.all and not args.stem:
        targets = jobs if args.force else pending
    else:
        targets = [pick_job(jobs, args.stem, force=args.force)]

    generated: List[Path] = []
    for job in targets:
        generated.append(generate_one(job, base_dir, output_root, api_cfg))
        if not args.all:
            remaining = len(pending) - 1
            print(f"\n已生成 1 张。剩余 {max(remaining, 0)} 张。批准后再运行: python generate_six_views.py --once")
            break

    for path in generated:
        print(f"成品: {path}")


if __name__ == "__main__":
    main()
