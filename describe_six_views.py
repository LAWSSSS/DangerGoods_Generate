#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片处理专家：导入一张图，用 doubao-seed-2.1-pro 写出超详细六视图 prompt。
每个视角分开描述，避免把 3/4 透视误当成正交方位。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import ssl
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests
import urllib3
import yaml

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
requests.packages.urllib3.disable_warnings()
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["CURL_CA_BUNDLE"] = ""

VLM_MODEL = "doubao-seed-2.1-pro"
VIEW_KEYS = ("front", "back", "left", "right", "top", "bottom")
VIEW_LABELS = {
    "front": "前",
    "back": "后",
    "left": "左",
    "right": "右",
    "top": "上",
    "bottom": "下",
}

SYSTEM_PROMPT = """你是图片处理专家，专做工业产品和异物的正交六视图拆解。
你只根据眼前这张照片做三维刚体还原，不要编造照片里没有的零件。

核心原则：
1. 参考图几乎一定是带透视的实拍（常见左前/右前 3/4），不是正交前视图。照片里某个零件「看起来朝右」，往往只是透视，不等于它焊在右侧。
2. 先建立物体坐标系：指定哪一面是前（通常是主标签/操作面），再定义后、左、右、上、下。零件相对这个坐标系焊死，六个相机只换观察方向。
3. 列出所有突出件：握把、压把、手柄、喷口、软管、管嘴、阀门、阀头、喷嘴、支架、卡扣、法兰、活塞杆、铭牌等。每件写清：装在哪里、物体空间指向哪里。
4. 再分别写六个正交视角。每个视角必须单独、详细，包含该视角画面里每个突出件的方向（画面左/右/上/下/朝镜头/背离镜头），以及被挡住的件。
5. 每个视角写「禁止」：最容易和哪一格画混，禁止把透视照片的左右直接抄进前视图。
6. 只输出一个 JSON 对象，不要 markdown，不要解释。"""

USER_PROMPT_TEMPLATE = """分析这张物体照片，生成正交六视图生图提示词。每个视角分别详细写，避免方位出错。

{size_block}

请输出 JSON，字段如下：
{{
  "object_name": "物体中文名",
  "reference_camera": "这张参考图实际相机方位，例如左前方约30度3/4，不是正视",
  "body_frame": {{
    "front": "哪一面算前，根据什么判定",
    "back": "后面是什么",
    "left": "左面有什么标志件",
    "right": "右面有什么标志件",
    "up": "上面有什么",
    "down": "下面有什么"
  }},
  "landmarks": [
    {{
      "name": "零件名，如握把或喷口",
      "mounted_on": "安装位置",
      "object_direction": "在物体坐标系里指向哪（前/后/左/右/上/下或其组合）",
      "notes": "颜色、形状、和别的件的相对关系"
    }}
  ],
  "views": {{
    "front": {{
      "camera": "相机在哪，看向哪",
      "visible": "看见哪些面和文字",
      "hidden": "看不见什么",
      "landmark_screen_dirs": [
        {{"name": "零件名", "screen_dir": "本画面方向：左/右/上/下/朝镜头/背离镜头，写具体"}}
      ],
      "forbidden": "本格最容易画错成什么，明确禁止",
      "prompt": "可直接给生图模型的前视图详细中文提示，必须包含每个突出件的画面方向"
    }},
    "back": {{"camera": "", "visible": "", "hidden": "", "landmark_screen_dirs": [], "forbidden": "", "prompt": ""}},
    "left": {{"camera": "", "visible": "", "hidden": "", "landmark_screen_dirs": [], "forbidden": "", "prompt": ""}},
    "right": {{"camera": "", "visible": "", "hidden": "", "landmark_screen_dirs": [], "forbidden": "", "prompt": ""}},
    "top": {{"camera": "", "visible": "", "hidden": "", "landmark_screen_dirs": [], "forbidden": "", "prompt": ""}},
    "bottom": {{"camera": "", "visible": "", "hidden": "", "landmark_screen_dirs": [], "forbidden": "", "prompt": ""}}
  }},
  "layout_note": "生图时建议 2行3列：前 后 左 / 右 上 下。俯视仰视约定：画面下=物体前，画面上=物体后，画面左=物体左。"
}}

views 里六个键 front/back/left/right/top/bottom 一个都不能缺。
每个 views.*.prompt 要能单独拿去画这一格，至少 6 句，必须出现该格所有突出件的画面方向。
左视图约定：画面右=物体前方，画面左=物体后方。
右视图约定：画面左=物体前方，画面右=物体后方。
俯视仰视约定：画面下=物体前方，画面上=物体后方，画面左=物体左侧。"""


class SSLAdapter(requests.adapters.HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


def get_session() -> requests.Session:
    session = requests.Session()
    adapter = SSLAdapter()
    session.mount("https://", adapter)
    session.verify = False
    return session


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    return yaml.safe_load(text) or {} if text else {}


def chat_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    suffix = "/chat/completions"
    return base if base.endswith(suffix) else base + suffix


def load_vlm_config(base_dir: Path) -> Dict[str, Any]:
    raw = load_yaml(base_dir / "config_bailian.yaml")
    api_key = raw.get("api_key_huoshan") or ""
    base = raw.get("vlm_api_huoshan_url") or ""
    if not api_key:
        raise ValueError("config_bailian.yaml 未配置 api_key_huoshan")
    if not base:
        raise ValueError("config_bailian.yaml 未配置 vlm_api_huoshan_url")
    return {
        "api_key": api_key,
        "api_url": chat_endpoint(base),
        "model": VLM_MODEL,
        "timeout_seconds": int(raw.get("timeout_seconds", 300)),
    }


def encode_image(path: Path) -> str:
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    mime = mime_map.get(path.suffix.lower(), "image/jpeg")
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def load_size_text(base_dir: Path, stem: str) -> str:
    path = base_dir / "尺寸描述" / f"{stem}_尺寸描述.md"
    if not path.exists():
        return "没有额外尺寸文档，只根据照片判断比例。"
    text = path.read_text(encoding="utf-8").strip()
    if len(text) > 1600:
        text = text[:1600] + "\n…"
    return "以下是该物体已有尺寸描述，写 prompt 时保持长宽高比：\n" + text


def extract_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if "```json" in cleaned:
        start = cleaned.find("```json") + 7
        end = cleaned.find("```", start)
        cleaned = cleaned[start:end].strip()
    elif "```" in cleaned:
        start = cleaned.find("```") + 3
        end = cleaned.find("```", start)
        cleaned = cleaned[start:end].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def validate_result(data: Dict[str, Any]) -> None:
    views = data.get("views")
    if not isinstance(views, dict):
        raise ValueError("VLM 未返回 views 对象")
    missing = [k for k in VIEW_KEYS if k not in views]
    if missing:
        raise ValueError(f"缺少视角: {missing}")
    for key in VIEW_KEYS:
        view = views[key]
        if not isinstance(view, dict):
            raise ValueError(f"{key} 不是对象")
        prompt = (view.get("prompt") or "").strip()
        if len(prompt) < 40:
            raise ValueError(f"{key} 的 prompt 过短")
        dirs = view.get("landmark_screen_dirs") or []
        if not dirs:
            raise ValueError(f"{key} 缺少 landmark_screen_dirs")


def stitch_prompt(data: Dict[str, Any]) -> str:
    name = data.get("object_name") or "物体"
    frame = data.get("body_frame") or {}
    landmarks = data.get("landmarks") or []
    layout = data.get("layout_note") or "2行3列：前 后 左 / 右 上 下。"
    lines: List[str] = [
        f"参考图是{name}的实拍。参考相机：{data.get('reference_camera', '')}。",
        "把它当成焊死刚体，生成同一物体的正交六视图，一张图 2 行 3 列。浅灰分隔。",
        "格子左下角只标一个汉字：前、后、左、右、上、下。禁止英文。纯白背景，正交无透视。",
        "",
        "【焊死方位】",
        f"前：{frame.get('front', '')}",
        f"后：{frame.get('back', '')}",
        f"左：{frame.get('left', '')}",
        f"右：{frame.get('right', '')}",
        f"上：{frame.get('up', '')}",
        f"下：{frame.get('down', '')}",
        "",
        "【突出件，六个格子同一套】",
    ]
    for item in landmarks:
        lines.append(
            f"- {item.get('name', '')}：装在{item.get('mounted_on', '')}，"
            f"物体方向{item.get('object_direction', '')}。{item.get('notes', '')}"
        )
    lines += ["", f"【布局】{layout}", ""]
    views = data.get("views") or {}
    for key in VIEW_KEYS:
        view = views.get(key) or {}
        label = VIEW_LABELS[key]
        lines.append(f"【{label}】{view.get('camera', '')}")
        lines.append(view.get("prompt") or "")
        dirs = view.get("landmark_screen_dirs") or []
        if dirs:
            bits = [f"{d.get('name', '')}→{d.get('screen_dir', '')}" for d in dirs]
            lines.append("本画面方向：" + "；".join(bits))
        if view.get("forbidden"):
            lines.append(f"禁止：{view.get('forbidden')}")
        lines.append("")
    return "\n".join(lines).strip()


def to_markdown(data: Dict[str, Any], image_path: Path) -> str:
    frame = data.get("body_frame") or {}
    landmarks = data.get("landmarks") or []
    views = data.get("views") or {}
    chunks = [
        f"# {data.get('object_name', image_path.stem)} 六视图 Prompt",
        "",
        f"依据：`{image_path}`",
        f"参考相机：{data.get('reference_camera', '')}",
        f"模型：{VLM_MODEL}",
        "",
        "## 刚体坐标系",
        "",
        f"- 前：{frame.get('front', '')}",
        f"- 后：{frame.get('back', '')}",
        f"- 左：{frame.get('left', '')}",
        f"- 右：{frame.get('right', '')}",
        f"- 上：{frame.get('up', '')}",
        f"- 下：{frame.get('down', '')}",
        "",
        "## 突出件",
        "",
    ]
    for item in landmarks:
        chunks.append(
            f"- **{item.get('name', '')}**：装在{item.get('mounted_on', '')}，"
            f"物体方向 {item.get('object_direction', '')}。{item.get('notes', '')}"
        )
    chunks += ["", "---", ""]
    for key in VIEW_KEYS:
        view = views.get(key) or {}
        label = VIEW_LABELS[key]
        chunks += [
            f"## 视角：{label}",
            "",
            f"- 相机：{view.get('camera', '')}",
            f"- 可见：{view.get('visible', '')}",
            f"- 不可见：{view.get('hidden', '')}",
            f"- 禁止：{view.get('forbidden', '')}",
            "",
            "本画面突出件方向：",
            "",
        ]
        for item in view.get("landmark_screen_dirs") or []:
            chunks.append(f"- {item.get('name', '')}：{item.get('screen_dir', '')}")
        chunks += ["", "生图提示：", "", view.get("prompt", ""), "", "---", ""]
    chunks += ["## 拼接后的完整生图 Prompt", "", "```", stitch_prompt(data), "```", ""]
    return "\n".join(chunks)


def call_vlm(image_path: Path, user_prompt: str, cfg: Dict[str, Any]) -> str:
    data_url = encode_image(image_path)
    full_text = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
    payload = {
        "model": cfg["model"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": full_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "max_tokens": 8192,
        "temperature": 0.2,
        "thinking": {"type": "disabled"},
    }
    print(f"  [VLM] 模型: {cfg['model']}", flush=True)
    print(f"  [VLM] 端点: {cfg['api_url']}", flush=True)
    session = get_session()
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }
    response = session.post(
        cfg["api_url"],
        headers=headers,
        json=payload,
        timeout=(30, cfg["timeout_seconds"]),
    )
    if response.status_code == 400 and "thinking" in (response.text or "").lower():
        payload.pop("thinking", None)
        print("  [VLM] 接口不接受 thinking，已去掉后重试", flush=True)
        response = session.post(
            cfg["api_url"],
            headers=headers,
            json=payload,
            timeout=(30, cfg["timeout_seconds"]),
        )
    if response.status_code != 200:
        print(f"  [VLM Error] 状态码: {response.status_code}")
        print(f"  [VLM Error] 响应: {response.text[:2000]}")
        response.raise_for_status()
    result = response.json()
    try:
        message = result["choices"][0]["message"]
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        if not str(content).strip():
            raise ValueError("VLM content 为空")
        return str(content)
    except (KeyError, IndexError, ValueError) as exc:
        debug = Path("debug_vlm_response.json")
        debug.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        raise ValueError(f"VLM 返回格式异常，已写入 {debug}") from exc


def describe(image_path: Path, base_dir: Path) -> Path:
    stem = image_path.stem
    out_dir = base_dir / "output" / "six_views" / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_vlm_config(base_dir)
    size_block = load_size_text(base_dir, stem)
    user_prompt = USER_PROMPT_TEMPLATE.format(size_block=size_block)
    print("\n===== 六视图描述 =====", flush=True)
    print(f"  源图: {image_path}", flush=True)
    raw_text = call_vlm(image_path, user_prompt, cfg)
    raw_path = out_dir / "六视图prompt.raw.txt"
    raw_path.write_text(raw_text, encoding="utf-8")
    data = extract_json(raw_text)
    validate_result(data)
    data["source_image"] = str(image_path)
    data["model"] = VLM_MODEL
    data["stitched_prompt"] = stitch_prompt(data)
    data["created_at"] = datetime.now(timezone.utc).isoformat()
    json_path = out_dir / "六视图prompt.json"
    md_path = out_dir / "六视图prompt.md"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(data, image_path), encoding="utf-8")
    print(f"  已保存: {md_path}")
    print(f"  JSON: {json_path}")
    return md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="用 doubao-seed-2.1-pro 为导入图片生成详细六视图 prompt")
    parser.add_argument("--image", required=True, help="物体图片路径")
    args = parser.parse_args()
    base_dir = Path(__file__).resolve().parent
    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = (base_dir / image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"找不到图片: {image_path}")
    describe(image_path, base_dir)


if __name__ == "__main__":
    main()
