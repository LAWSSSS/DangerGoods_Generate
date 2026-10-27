#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""豆包系列。生图 doubao-seedream-5.0-pro，VLM doubao-seed-2.1-pro。算法和提示词只在本文件修改。"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import random
import re
import sys
import time
import ssl
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
import yaml
from PIL import Image, ImageDraw, ImageFilter
import urllib3

THIS_SERIES = "doubao"
OTHER_SCRIPT = "generate_aliyun.py"
DEFAULT_IMAGE = "doubao-seedream-5.0-pro"
DEFAULT_VLM = "doubao-seed-2.1-pro"
ALLOWED_IMAGES = ("doubao-seedream-5.0-pro",)
ALLOWED_VLMS = ("doubao-seed-2.1-pro",)
CHOSEN_IMAGE = DEFAULT_IMAGE
CHOSEN_VLM = DEFAULT_VLM


def series_of_image_model(name: str) -> Optional[str]:
    if name.startswith("doubao-seedream"):
        return "doubao"
    if name.startswith("wan") or name.startswith("qwen"):
        return "qianwen"
    return None

def series_of_vlm(name: str) -> Optional[str]:
    if name.startswith("doubao-seed"):
        return "doubao"
    if name.startswith("qwen"):
        return "qianwen"
    return None

def handoff_to_series(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """同一套参数。生图模型决定系列；没写生图模型时才用 VLM 判断。"""
    image_name = args.model or str(load_app_raw().get("model") or "")
    series = series_of_image_model(image_name) if image_name else None
    if series is None:
        vlm_name = args.vlm_model or str(load_app_raw().get("vlm_model") or "")
        series = series_of_vlm(vlm_name) if vlm_name else None
    if series is None:
        parser.error("无法从生图模型或 VLM 判断是豆包还是千问")
    if args.model and series_of_image_model(args.model) != series:
        parser.error("生图模型与所选系列不一致")
    if series != THIS_SERIES:
        target = Path(__file__).resolve().parent / OTHER_SCRIPT
        os.execv(sys.executable, [sys.executable, str(target), *sys.argv[1:]])
    if args.vlm_model and series_of_vlm(args.vlm_model) != THIS_SERIES:
        parser.error("VLM 与生图模型不是同一系列")
    try:
        image_model, vlm_model = models_for_this_series(args.model or image_name, args.vlm_model)
    except ValueError as exc:
        parser.error(str(exc))
    global CHOSEN_IMAGE, CHOSEN_VLM
    CHOSEN_IMAGE, CHOSEN_VLM = image_model, vlm_model

# ============================================================
# 提示词。改措辞只改这一段，下面的接口和采样不要跟着改。
# 外观五项、逐件生图、类别生图、模板、尺寸句、位置句。
# ============================================================

# ---------- 外观五项 ----------

def geometry_identity(profile: dict[str, Any]) -> list[str]:
    colors = r"青绿色|深绿色|深蓝色|紫粉色|银白色|红色|绿色|蓝色|黑色|银色|白色|灰色|黄色|紫色|红黄|红黑|黑蓝"
    return [
        re.sub(colors, "", feature).strip()
        for feature in profile["identity_features"]
        if not any(token in feature for token in ("字样", "标签文字", "商标"))
    ]

def appearance_neutral_name(profile: dict[str, Any]) -> str:
    return re.sub(
        r"^(?:青绿色|深绿色|深蓝色|紫粉色|红色|绿色|蓝色|黑色|银色|白色|灰色|黄色|紫色|红黄|红黑|黑蓝)\s*",
        "",
        profile["name"],
    )

WEATHERING = {
    "painted_steel": "明确呈现与当前废钢堆相称的报废钢件：可见钢主体主要为深褐锈、灰褐尘土和旧油泥，原漆仅作约10–20%的暗淡碎片残留；此为合成外观目标，不是实测腐蚀面积。裸钢有不规则锈层、点蚀和接缝积锈，大片剥漆，不保留完整红/蓝/绿大色块或亮漆高光。不能只加几个锈点。锈纹沿材质、磕碰和水路分布，不均匀喷成橙色。塑料旋钮/标牌不生铁锈。",
    "painted_castiron": "报废铸铁件的可见外壳明显氧化、漆膜剥落，深褐锈层与铸造粗糙度相互叠加，凹槽积尘油泥，底脚、孔边及螺栓有锈蚀；原色只留下暗淡残漆。受力面可局部擦亮但不能形成干净的新品反光，禁止仅在角落添几处轻微磨痕。",
    "stainless_steel": "呈现长期弃置的不锈钢旧件：镜面高光被灰尘、油膜、水垢和不均匀灰褐污层压低，表面有密集细擦伤、暗斑，焊缝/钢制紧固件可有局部锈迹或铁污染。主体仍为不锈钢，不把全部表面变成碳钢红锈；禁止均匀厚红锈。绝不保留干净银白新品和摄影棚灯条反射。塑料手柄只污旧。",
    "cast_steel": "报废铸钢阀门主体和法兰明显呈深褐/红褐氧化、油泥与积尘，螺栓孔和接缝积锈；银白新金属面改为失去光泽的旧钢材，蓝手轮褪色、剥漆，保留暗淡蓝色身份线索。保留厚法兰、曲面及真实接缝，不套用同一锈贴图，不仅给螺栓点几处轻锈。",
    "aerosol_can": "报废薄壁气雾罐应明显污旧：印刷褪色磨花、标签残损，表面附着灰褐粉尘和油污，钢制罐在掉漆、划伤、卷边处有清楚的不规则氧化/锈斑。保留弱化的主色与结构线索，禁止亮丽完整广告罐和可读新品印刷。塑料罩盖/喷头只积灰黄化擦伤、不生铁锈，薄壁结构不能被画成厚铸铁。",
    "fire_extinguisher": "报废灭火器筒体明显褪色、大片擦伤掉漆，裸钢有清楚可辨的褐色锈蚀、接缝流痕和灰尘油泥；红/绿原漆仅保留暗淡旧色块，标签污损模糊，不能保留亮丽洁净的新筒体。金属握把和连接件可锈蚀，橡胶软管只沾灰油并自然弯曲，不生铁锈，不新增喷射烟雾。",
    "painted_steel_chrome": "报废液压钢缸外壳明显剥漆、锈蚀、油泥和积尘，钢耳环/油口凹处有深褐锈与污垢，原漆暗淡残留。镀铬杆有旧油膜、细擦痕和局部暗斑，只保留弱金属反光，不整根涂成橙锈；禁止干净亮漆缸筒、亮银镜面活塞杆和新品黄标。保持缸筒厚度及机械连接。",
    "mixed_shock": "报废减震器的钢弹簧和钢端件有明显掉漆、褐锈和油灰，弹簧缝积尘；铝件/镀铬杆有灰暗污膜、氧化擦伤，只有微弱反射。原有彩色残漆降低饱和度；橡胶护套和软管积灰油污、轻微老化但不生铁锈。不能保留崭新的彩色弹簧、亮黑新品外壳；保持圈距和连接。",
    "air_shock": "废旧空气避震以明显灰褐积尘、油泥、涂层擦伤、铝表面灰白氧化表现旧化，钢耳件和螺栓有锈斑。黑色气室呈灰暗哑光而非新品亮黑；调节钮仅保留暗淡颜色线索，禁止厚铁锈覆盖铝气室，也不增加螺旋弹簧。",
    "aluminium_steel": "报废气缸铝壳明显灰暗、灰白氧化、密集擦伤并覆盖不均匀油泥积尘；钢拉杆、螺栓、安装端有清楚锈斑。活塞杆有旧油膜和弱反光，禁止干净的银亮铝型材和鲜明新品贴标，禁止把铝壳和杆统一涂成红锈。保持铝/钢/镀铬部件的材质差异。",
    "plastic_pp": "废弃白色PP罐明显积灰、灰褐油污、黄化和擦伤，盖纹及接缝积污，反射变为旧塑料的暗哑漫反射，不能只点几处轻污后仍像白色新品。没有红褐铁锈，不变成金属罐，不无故烧熔或大面积开裂；污损色与附近废料相称。",
}

APPEARANCE_KEYS = ("color_temperature", "tone", "saturation", "pixel", "sharpness")

APPEARANCE_MATCHES = (
    ("color_temperature_match", "色温修正", "describe_color_temperature", "color_temperature"),
    ("tone_match", "色调修正", "describe_tone", "tone"),
    ("saturation_match", "饱和度修正", "describe_saturation", "saturation"),
    ("pixel_match", "像素修正", "describe_pixel", "pixel"),
    ("sharpness_match", "清晰度修正", "describe_sharpness", "sharpness"),
)

def _observed(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    return text if len(text) >= 2 else fallback

def describe_color_temperature(color_temperature: str = "") -> str:
    """色温只写白平衡和冷暖，不写色相。"""
    observed = _observed(color_temperature, "与相邻废钢相同的冷暖和白平衡")
    return (
        f"观察：{observed}。"
        "高光、中间调、阴影共用这一套白平衡；"
        "金属高光不另加暖黄或橙色，阴影不另加青蓝或冷紫；"
        "异物与废钢交界处冷暖连续，不能出现一圈独立色温，异物本身不能单独偏暖或偏冷。"
    )

def describe_tone(tone: str = "") -> str:
    """色调只写色相和明度。饱和度是单独一项。"""
    observed = _observed(tone, "铁锈红、深灰、暗褐，明度不高于周围废钢")
    return (
        f"观察：{observed}。"
        "只看色相和明度，跟相邻废钢，不跳纯色，不加更亮的边。"
        "不要在这句里写饱和度，也不要写冷暖。"
    )

def describe_saturation(saturation: str = "") -> str:
    """饱和度必须发灰，并且明显低于相邻废钢。"""
    observed = _observed(saturation, "发灰、蒙尘、褪色，明显低于周围废钢")
    return (
        f"观察：{observed}。"
        "饱和度必须再压低一档，明显低于相邻废钢，看起来发灰、发浊、蒙尘、褪色；"
        "宁可几乎看不出原来的颜色，也不要鲜艳；"
        "禁止鲜红、橘红、亮黄、翠绿、纯蓝、荧光色和高饱和色块；"
        "商品原色只留一点色相，其余混进灰尘和铁锈灰。"
    )

def describe_pixel(pixel: str = "") -> str:
    """像素只写颗粒、压缩和局部对比，不写清不清楚。"""
    observed = _observed(pixel, "颗粒粗、压缩块明显、局部对比低")
    return (
        f"观察：{observed}。"
        "异物的像素细节必须低于监控原图：采样更粗，颗粒更大，压缩块更明显，局部对比更低；"
        "细线、锈点和刻字不能比相邻废钢分得更开；"
        "和原图像素相同仍然偏高，要再粗一档。"
        "禁止高清贴图、干净边缘和无噪点表面。"
    )

def describe_sharpness(sharpness: str = "") -> str:
    """清晰度只写失焦和拖影。不把整张监控图再柔化。"""
    observed = _observed(sharpness, "没有拖影时保持轻微失焦和发软边缘")
    return (
        f"观察：{observed}。"
        "如果废钢边缘沿同一方向被拉糊，那是拍摄时相机晃动，异物必须带同样方向、同样长度的运动模糊；"
        "没有拖影时，也要保持同样的失焦和发软边缘。"
        "异物不能更锐，不能在发糊的画面里突然清晰。"
        "只改异物，不要把整张监控图再柔化、降分辨率或二次渲染发糊。"
    )

def appearance_from_context(context: dict[str, Any] | None) -> dict[str, str]:
    source = context or {}
    return {key: str(source.get(key) or "").strip() for key in APPEARANCE_KEYS}

def appearance_prompt_block(appearance: dict[str, str] | None = None) -> str:
    observed = appearance or {}
    return "\n".join([
        f"【色温】{describe_color_temperature(observed.get('color_temperature', ''))}",
        f"【色调】{describe_tone(observed.get('tone', ''))}",
        f"【饱和度】{describe_saturation(observed.get('saturation', ''))}",
        f"【像素】{describe_pixel(observed.get('pixel', ''))}",
        f"【清晰度】{describe_sharpness(observed.get('sharpness', ''))}",
    ])

def normalize_appearance_match(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"bad", "poor", "差", "不匹配", "不一致", "失败"}:
        return "bad"
    if text in {"good", "很好", "匹配", "一致"}:
        return "good"
    return "ok"

def appearance_is_bad(evaluation: dict[str, Any]) -> bool:
    return any(
        normalize_appearance_match(evaluation.get(match_key)) == "bad"
        for match_key, _label, _method, _obs in APPEARANCE_MATCHES
    )

def appearance_correction_text(
    evaluation: dict[str, Any], appearance: dict[str, str] | None
) -> str:
    """只把判为 bad 的那一项写成对应的生图要求，不改写禁令用词。"""
    observed = appearance or {}
    writers = {
        "describe_color_temperature": describe_color_temperature,
        "describe_tone": describe_tone,
        "describe_saturation": describe_saturation,
        "describe_pixel": describe_pixel,
        "describe_sharpness": describe_sharpness,
    }
    parts: list[str] = []
    for match_key, label, method, obs_key in APPEARANCE_MATCHES:
        if normalize_appearance_match(evaluation.get(match_key)) != "bad":
            continue
        parts.append(f"【{label}】{writers[method](observed.get(obs_key, ''))}")
    if not parts:
        return ""
    return (
        "\n".join(parts)
        + "\n把每一项修正写成对应的画面要求：色温不写色相，色调不写冷暖也不写饱和度，"
        "饱和度必须发灰且明显低于废钢，像素不写清不清楚，清晰度不写噪点。"
        "不要要求柔化整张图，也不要替换禁止鲜红、青蓝、橙色这类原词。"
    )

# ---------- 场景、评审、逐件生图 ----------

def scene_context_prompt(context: dict[str, Any] | None) -> str:
    labels = {
        "lighting": "当前光照与方向",
        "contrast": "当前明暗和反射强度",
        "capture_style": "当前成像风格与噪声",
        "surface_condition": "附近废料的旧化状态",
        "support_and_occlusion": "支撑面和前景遮挡",
    }
    context = context or {}
    parts = [
        f"{label}：{str(context[key])[:260]}"
        for key, label in labels.items()
        if isinstance(context.get(key), str) and context[key].strip()
    ]
    if not parts:
        return "【背景适配】直接观察本张原监控图，按目标附近钢件的明暗、色温、旧化、反射、噪声和锐度重画物品；无法确定光源时不要添加摄影棚高光或新投影方向。"
    return "【背景适配，来源为VLM观察而非物理测量】" + "；".join(parts)

def review_requirements() -> str:
    return (
        "分别给weathering_score、lighting_score、occlusion_score、style_score、"
        "background_preservation_score、identity_score（0–100）。"
        "旧化检查可见材质是否明确报废且与附近废料相称；铁/碳钢须有清楚可辨的锈蚀，"
        "塑料、铝、不锈钢主要检查积污、氧化、擦伤和弱反射，不能要求全部铁锈。"
        "若仍是新漆、亮银干净新品、摄影棚高光或只有装饰性几个锈点，new_appearance_detected=true且weathering_score不得高于60。"
        "检查物品是否有真实承重支点、至少部分轮廓被原有前景钢件截断、贴合接触阴影；"
        "完整展品摆在堆顶、悬浮、边缘光晕、轮廓全露或独立黑描边判pasted_appearance_detected=true且occlusion_score不得高于60。"
        "照明检查局部亮度、高光方向/软硬、阴影和邻近反射；风格检查对比是否相称，不能用柔化全图掩盖贴图感。"
        "色温、色调、饱和度、像素、清晰度分开打：color_temperature_match、tone_match、saturation_match、pixel_match、sharpness_match，取值 bad|ok|good。"
        "色温只看白平衡和冷暖，不看色相。色调只看色相和明度，不看饱和度，不看冷暖。"
        "饱和度必须发灰、发浊、蒙尘、褪色，并且明显低于周围废钢；鲜红、橘红、亮黄、翠绿、纯蓝、荧光，或和废钢一样鲜，saturation_match 为 bad。"
        "像素只看颗粒、压缩块和局部对比，不看清不清楚。清晰度只看失焦、边缘软硬和拖影；异物比废钢更锐为 bad。不要建议柔化整张图。"
        "tone_fix 只改色相和明度，不要把饱和度写进 tone_fix。"
        "背景保持需对照原图检查目标区域外结构和文字；身份检查该件的主体和连接，不以生造铭文代替结构。"
        "另给object_present（布尔）、new_appearance_detected（布尔）、pasted_appearance_detected（布尔）、"
        "material_violation（布尔）、weathering_fix、lighting_fix、occlusion_fix、style_fix、background_fix、identity_fix。"
        "全部六项必须高于85、存在目标且三个缺陷标志均为false才达到模型复核标准。"
        "分数必须根据看见的图像给出，不能因提示词写了要求就给高分；模型复核通过仍需人工看图。"
    )

def region_prompt(region: list[float] | None) -> str:
    if not region:
        return ""
    x1, y1, x2, y2 = region
    return (
        f"【局部重建区域】归一化坐标x={x1:.3f}–{x2:.3f}、y={y1:.3f}–{y2:.3f}，"
        f"目标位于区域中心附近（{(x1 + x2) / 2:.1%}，{(y1 + y2) / 2:.1%}）。"
        "此框包含物品、接触阴影和少量已有前景钢件，不是物品的外接框；物品可见长度仍按本轮设定比例。"
        "允许在框内重建接触处和遮挡边缘，使原有钢条/钢板自然盖住物品部分轮廓；保持框外场景、时间戳和机位字样。"
    )

def size_spec(
    profile: dict[str, Any], target_ratio: float | None = None
) -> dict[str, Any]:
    policy, brief = profile["size_policy"], profile_brief(profile)
    low, high = policy["visible_ratio_range"]
    ratio = (
        (low + high) / 2 if target_ratio is None else min(high, max(low, target_ratio))
    )
    span_cm = policy["assumed_scene_width_mm"] / 10
    length, cross = brief["height_cm"], brief["diameter_cm"]
    dims = "，".join(
        f"{name} {value:g} mm" for name, value in profile["dimensions_mm"].items()
    )
    parts = "，".join(
        f"{name} {value:g} mm" for name, value in profile["parts_mm"].items()
    )
    text = (
        f"【本轮设定尺寸】{profile['name']}（{profile['id']}）。合成工作尺寸：{dims}。"
        f"结构细节：{parts}。尺寸证据状态：{profile['dimension_evidence']['status']}；"
        "工作值不是对此实例的实测值。"
        f"本轮主体可见最长边目标约占画面宽度 {ratio:.2%}，允许 {low:.2%}–{high:.2%}。"
        f"此范围以背景水平有效宽度暂按 {span_cm:g} cm、局部投影范围的假设推导；背景尚未标定。"
        "遮挡使可见区域变短，不得据此缩小完整主体或更改结构比例。"
        f"{profile['exposure_hint']}。"
    )
    return {
        "object_class": profile["category"],
        "profile_id": profile["id"],
        "physical_name": profile["name"],
        "target_ratio": ratio,
        "target_cm": ratio * span_cm,
        "diameter_cm": cross,
        "diameter_to_height": cross / length,
        "cross_name": brief["cross_name"],
        "scale_min": low,
        "scale_max": high,
        "size_brief": text,
        "real_height_cm": length,
        "real_diameter_cm": cross,
        "parts": brief["parts"],
        "expose": brief["expose"],
        "forbid": brief["forbid"],
        "dimension_evidence": profile["dimension_evidence"],
        "cm_fields_semantics": "configured_working_dimensions_not_instance_measurements",
        "background_calibration": "assumed_not_measured",
        "assumed_scene_width_cm": span_cm,
    }

def render_prompt(
    profile: dict[str, Any],
    target_ratio: float | None = None,
    position_hint: str | None = None,
    orientation_hint: str | None = None,
    feedback: str = "",
    scene_context: dict[str, Any] | None = None,
    edit_region_xyxy: list[float] | None = None,
    reference_kind: str = "six_view_atlas",
) -> str:
    # Avoid recursively nesting old full prompts when the optimizer returns prior constraints.
    fixed_prefixes = (
        "只新增一件",
        "第1张图是",
        "【本轮设定尺寸】",
        "身份和几何必须保留：",
        "材质和旧化：",
        "姿态和支撑：",
        "遮挡：",
        "匹配原场景的",
        "只在原监控图内",
        "禁止：",
        "当前指定位置：",
        "当前朝向建议：",
        "【最高优先级",
        "【背景适配",
        "【局部重建区域】",
        "【接触与遮挡】",
        "【外观不合格条件】",
    )
    feedback_lines = []
    for line in feedback.splitlines():
        line = line.strip()
        if line.startswith(fixed_prefixes):
            continue
        if line.startswith("本轮局部修正建议（服从上述身份、材质和尺寸设定）："):
            line = line.split("：", 1)[1]
        if line and line not in feedback_lines:
            feedback_lines.append(line)
    feedback = "\n".join(feedback_lines)
    lines = [
        "【最高优先级：背景中的报废旧件】输出当前监控场景中长期弃置的这一件物品。几何身份参考当前物品参考图，旧化、光照、色调和成像外观以第1张场景为准。必须重新生成已明显旧化的材质，不能保留新品漆面、干净金属面、摄影棚高光、鲜艳标签或展品摆放状态。名称中的原始颜色只帮助辨认，不得据此恢复饱和新品颜色。",
        (
            f"只新增一件 {appearance_neutral_name(profile)}，后图是用户提供的生锈六视图，同一件物品的六个角度不是六个实例。保留既有锈蚀和剥漆特征，按原场景重新匹配光线、色调、锐度和噪声；禁止恢复新品外观。"
            if reference_kind == "weathered_six_view_atlas"
            else f"只新增一件 {appearance_neutral_name(profile)}，后图为已重新生成的废旧状态身份参考；几何来自该件六视图，完整参考不是最终场景中的完整展示要求。"
            if reference_kind == "weathered_single_view"
            else f"只新增一件 {profile['name']}，使用当前选定六视图辨认同一物品；六个格子不是六个实例。"
        ),
        "第1张图是原始监控场景，后面的图只作身份/结构参考。",
        scene_context_prompt(scene_context),
        appearance_prompt_block(appearance_from_context(scene_context)),
        size_spec(profile, target_ratio)["size_brief"],
        "身份和几何必须保留：" + "；".join(geometry_identity(profile)) + "。",
        "材质和旧化：" + profile["weathering_prompt"],
        "姿态和支撑：" + profile["pose_hint"] + "；" + profile["support_hint"] + "。",
        "遮挡：" + profile["exposure_hint"] + "。",
        "【接触与遮挡】先选真实钢板/型钢支点，把主体大部分置于废钢缝隙深处；至少一块原有宽钢板/型钢遮住主体中段并截断两侧外轮廓，不以一条细杆轻轻跨过表面代替真正埋入。露出一段可识别旧材质及安装端之一，保持完整几何尺度。接触阴影只在真实支点附近贴合产生，不能完整摆在堆顶或出现整圈描边。允许小幅重建接触处和前景遮挡边缘。",
        "接触阴影贴合真实支点，投影服从原场景光源。上面五项分开执行：色温只对齐白平衡，色调只对齐色相和明度，饱和度必须低于废钢，像素只对齐颗粒和压缩，清晰度只对齐失焦和拖影。",
        "只在原监控图内重新生成物品，不抠图/贴图/复制摄影棚底色或宫格线；保持时间戳、机位文字、料斗边框、相机视角和非目标废料结构。",
        "禁止："
        + "；".join(profile["forbidden_changes"])
        + "；悬浮、穿模、重复实例、额外无关部件、整图柔化、错误光照。",
        "【外观不合格条件】钢铁表面仍像新品、只有零星装饰锈点、残漆过亮、孤立摄影棚高光、独立色温、过锐边缘、白边光晕、完整展品轮廓无前景遮挡，均不符合要求。标签只留污旧色块，不要求清晰可读。",
    ]
    if position_hint and not edit_region_xyxy:
        lines.append(
            "当前指定位置："
            + position_hint
            + "；选择可承重的接触面并避开时间戳/机位字样。"
        )
    if orientation_hint:
        lines.append(
            "当前朝向建议："
            + orientation_hint
            + "；若与该物品支撑条件冲突，优先保持可稳定支撑的姿态。"
        )
    if feedback.strip():
        lines.append(
            "本轮局部修正建议（服从上述身份、材质和尺寸设定）：" + feedback[:1200]
        )
    if edit_region_xyxy:
        lines.append(region_prompt(edit_region_xyxy))
    return "\n".join(lines)

def evaluator_prompt(profile: dict[str, Any]) -> str:
    return (
        f"你评估的对象是 {profile['name']}（{profile['id']}），而非该大类的其他型号。"
        "只输出有效JSON。visible_ratio是主体可见最长边/画面宽度，可以依据图像估计；"
        "背景尚未标定，换算厘米仅是有条件的估计，不得宣称从单图测得真实长度。"
        "按该物品的工作尺寸、相对结构比例、姿态、投影和遮挡检查。"
        "不要求可见片段长于完整横截面；被前景遮住的圆弧/边缘可以更小，但不能把整个对象缩成玩具。"
        "同大类不同尺寸不得混用，材料保持自身物理外观。模型意见不等于人工验收。"
        "禁止建议柔化整图；sharpness与尺寸判断分开。"
        "返回visible_ratio、size_judgment（too_large|acceptable|too_small）、"
        "tone_score、size_score、overall_score（0到100）、tone_fix、size_fix、overall_fix、"
        "color_temperature_match、tone_match、saturation_match、pixel_match、sharpness_match（bad|ok|good）、"
        "sharpness（sharp|ok|soft）、confidence（0到1）、reason、improvement_suggestion、"
        "quality_score（0到1）、fusion_quality（very bad|bad|ok|good|very good）。"
        "不确定时降低confidence并在reason说明，不把假设写成测量。"
        + review_requirements()
    )

def evaluation_user_prompt(
    profile: dict[str, Any], spec: dict[str, Any], history: str
) -> str:
    return (
        f"只评估当前 {profile['name']}（{profile['id']}）的生成图。\n"
        + spec["size_brief"]
        + "\n"
        + f"当前目标主体可见最长边/画面宽度={spec['target_ratio']:.2%}，"
        + f"允许{spec['scale_min']:.2%}–{spec['scale_max']:.2%}。\n"
        + "轮次历史只用于理解调整过程，当前判断以本轮设定和当前输出为准。\n"
        + history
        + "\n"
        + evaluator_prompt(profile)
    )

# ---------- 类别生图与单次角色提示 ----------

def build_prompt(
    object_class: str,
    prompt: str,
    negative_prompt: str,
    object_path: Path = None,
    background_path: Path = None,
    position_hint: str = None,
    orientation_hint: str = None,
    cfg: Optional[AppConfig] = None,
    target_ratio: Optional[float] = None,
) -> str:
    profile = getattr(cfg, "active_object_profile", None) if cfg is not None else None
    if profile and profile["category"] == object_class:
        return render_per_item_prompt(
            profile, target_ratio=target_ratio, position_hint=position_hint,
            orientation_hint=orientation_hint, feedback=prompt or "",
            scene_context=getattr(cfg, "active_scene_context", None),
            edit_region_xyxy=getattr(cfg, "active_edit_region", None),
            reference_kind=getattr(cfg, "active_reference_kind", "six_view_atlas"),
        )
    if "【本轮设定尺寸】" in (prompt or ""):
        size_info = "尺寸只按提示词里的【本轮设定尺寸】绘制：结构用真实厘米，可见最长边用本轮厘米，不要把筒径再缩小，不要参考更早轮次"
    elif background_path:
        try:
            size_info = class_size_clause(cfg, object_class, target_ratio)
        except Exception:
            size_info = f"{object_class}必须按真实物体大小出现，不要画成玩具或占据整幅画面"
    else:
        size_info = f"{object_class}必须按真实物体大小出现，不要画成玩具或占据整幅画面"
    
    enhancer = PromptEnhancer()
    weathering = random.choice(enhancer.get_weathering_descriptions(object_class))
    scenario = random.choice(enhancer.get_scenario_descriptions())
    
    fusion_enhance = (
        "【精确尺寸控制】\n"
        f"1. {size_info}\n"
        "2. 筒径、握把、喷管、活塞杆、耳环、法兰使用真实厘米，不要按可见边同比缩成模型\n"
        "3. 可见最长边对应该件在240厘米车斗视野中的真实尺度；没放进去时按同一真实尺寸放入\n"
        "\n【形状严格保持】\n"
        f"4. 必须严格保持{object_class}原本的形状、结构、比例和轮廓特征\n"
        "5. 不允许任何形式的变形、拉伸、扭曲或比例改变\n"
        "6. 圆形必须保持圆形，圆柱形保持圆柱形，方形保持方形\n"
        "\n【嵌入与遮挡】\n"
        "7. 异物必须被废钢挤压包裹，从废钢缝隙中露出能辨认的一段，像被自然埋在里面\n"
        "8. 异物的边缘必须被废钢遮挡，有明显的遮挡和交错，大部分被挡住，但仍须露出能辨认的一小段\n"
        "9. 异物表面的锈迹、油污、灰尘纹理必须与周围废钢完全一致\n"
        "\n【色彩与光影融合】\n"
        "10. 下面五项分开写，禁止混成一句颜色或一句清晰度：\n"
        f"{appearance_prompt_block(appearance_from_context(getattr(cfg, 'active_scene_context', None)))}\n"
        "11. 异物的光影必须与背景图片的光照方向完全一致\n"
        "12. 异物的材质感必须与废钢一致，不能是亮面或光滑的\n"
        "\n【风格与质感】\n"
        "13. 禁止生成干净、崭新、反光强烈的物体\n"
        "14. 废钢的锈蚀纹理必须覆盖在异物表面\n"
        f"15. {weathering}\n"
        f"16. 整体场景：{scenario}\n"
        "17. 废钢场景、时间戳和摄像头文字保持原样，但必须在指定位置新增异物，输出不能与背景图相同\n"
        "18. 整张监控图禁止柔化、涂抹、降分辨率或二次渲染发糊。异物自己的色温、色调、饱和度、像素和清晰度按上面五项贴齐相邻废钢，不能比废钢更锐、更干净、更鲜艳\n"
        "19. 每一轮都只在原始监控背景上重画，不要把任何已生成图当作画布\n"
        "\n【多视图只作参考】\n"
        f"20. 第1张图是废钢监控原图，只作为画布；第2张及之后是{object_class}的多视角参考\n"
        "21. 参考图只用来认外形、零件和直径/高度，不要照抄参考图的鲜艳原色，颜色必须改成发灰、褪色、蒙尘；不要把参考图里的物体直接插入场景\n"
        "22. 禁止抠图、贴图、叠加、原样拷贝；禁止留下白底、摄影棚、宫格线或前/后/左/右/上/下标签\n"
        "23. 必须按本轮真实厘米在废钢堆里重新画出被埋住的同类异物，光影锈蚀与车斗一致\n"
    )
    if position_hint:
        fusion_enhance += f"\n【出现位置】异物必须出现在{position_hint}。\n"
    if orientation_hint:
        fusion_enhance += f"\n【横放朝向】{orientation_hint}。\n"

    position_head = f"【强制位置】把异物放在{position_hint}。\n\n" if position_hint else ""
    orientation_head = f"【强制朝向】{orientation_hint}。\n\n" if orientation_hint else ""
    full_prompt = (
        f"{position_head}"
        f"{orientation_head}"
        f"{prompt}\n\n"
        f"{fusion_enhance}\n"
        f"目标异物类别：{object_class}。\n"
        "不要改变废钢场景、时间戳和摄像头点位文字，不要改变视角，"
        "不要生成卡通风格，不要生成干净崭新或悬浮的异物，不要偏离该类别的真实尺寸。\n"
        f"【必须放入】必须把{object_class}放进第1张背景图的指定位置。"
        "可以小、大部分被遮挡，但必须露出能辨认的一小段。"
        "禁止省略异物，禁止输出一张与背景图相同的图。"
        f"多视图里的{object_class}只是外形样本，禁止把它直接插入第1张场景。"
    )
    return full_prompt

def generation_role_prompt(prompt: str, view_count: int = 0, references_weathered: bool = False, reference_kind: str = "six_view_atlas") -> str:
    """单次调用的文字：只把原始背景当画布；多视图只认外形，禁止当贴图。"""
    lines = [
        "第1张图是最初的废钢监控原图，只作为画布。必须在这张图上新增异物。",
        "废钢场景、时间戳和摄像头文字保持原样，但输出不能与这张背景图相同。",
        "色温：异物的高光、中间调、阴影与相邻废钢同一白平衡，高光不另加暖黄，阴影不另加青蓝，交界不能有一圈独立色温。",
        "色调：只看色相和明度，跟相邻废钢，不跳纯色，不加更亮的边。不要在这句里写饱和度。",
        "饱和度：必须发灰、发浊、蒙尘、褪色，并且明显低于周围废钢。宁可几乎看不出原来的颜色，也不要鲜艳。禁止鲜红、橘红、亮黄、翠绿、纯蓝、荧光色和高饱和色块。商品原色只留一点色相，其余混进灰尘和铁锈灰。",
        "像素：异物细节必须低于监控原图，颗粒更粗，压缩块更明显，局部对比更低；和原图一样细仍然偏高。",
        "清晰度：先沿用背景的拍摄晃动。废钢边缘若沿同一方向拉出拖影，异物必须同样拉糊；没有拖影时保持同样失焦。异物不能比废钢更锐。禁止把整张监控图再柔化。",
        "每一轮都只在这张原始背景上重画，不要使用任何已生成图当作画布。",
    ]
    if view_count > 0:
        lines.extend(
            [
                (
                    f"后面 {view_count} 张是用户提供的生锈六视图，参考几何、零件及既有锈蚀；六格为同一件的六个视角，输出只增加一件。按第1张背景重建光照和接触遮挡，保留旧化，不复制白底或格线。"
                    if reference_kind == "weathered_six_view_atlas" else
                    f"后面 {view_count} 张是已生成的废旧状态物品身份参考，只参考几何、零件和旧化材质；必须在第1张原背景内重新画出同一件。"
                    if references_weathered else
                    f"后面 {view_count} 张只用来认形状、零件和比例；图中是新品参考，输出必须重新生成已明显旧化的报废状态。不要照抄参考图的鲜艳原色。"
                ),
                "多视图只用来认形状、零件和比例，不要照抄参考图的鲜艳原色。不得继承参考图的鲜亮漆面、干净金属面、摄影棚高光、独立白平衡、锐度和标签清晰度；当前监控背景决定外观。",
                "禁止把参考图里的物体直接抠图、贴图、插入、叠加或原样拷贝进第1张场景。",
                "禁止保留白底、摄影棚、宫格分割线，以及前、后、左、右、上、下这类视角标签。",
                "必须按本轮设定尺寸，在废钢堆里重新画出一件被埋住的同类异物，光影和锈蚀跟车斗一致。",
            ]
        )
    lines.append(prompt)
    return "\n".join(lines)

# ============ Prompt增强工具 ============

class PromptEnhancer:
    @staticmethod
    def get_position_descriptions(gaps: str) -> List[str]:
        base_positions = [
            "废钢堆叠的缝隙深处,看不出人工放置的痕迹",
            "两块钢板之间的夹缝中，看不出人工放置的痕迹",
            "齿轮与型钢的交接处，看不出人工放置的痕迹",
            "废钢堆的底部空隙，看不出人工放置的痕迹",
            "大型废钢件的下方阴影处，看不出人工放置的痕迹",
            "杂乱废钢的中间层，看不出人工放置的痕迹",
            "废钢堆的角落凹陷处，看不出人工放置的痕迹",
            "管状废钢与板材的夹角中，看不出人工放置的痕迹",
        ]
        if gaps and len(gaps) > 5:
            custom_positions = [
                f"{gaps}的深处,看不出人工放置的痕迹",
                f"{gaps}的隐蔽角落,看不出人工放置的痕迹",
                f"{gaps}的夹层中,看不出人工放置的痕迹",
                f"{gaps}的下方阴影处,看不出人工放置的痕迹",
            ]
            return custom_positions - ["看不出人工放置的痕迹"] + base_positions
        return base_positions
    
    @staticmethod
    def get_occlusion_descriptions() -> List[str]:
        return [
            "被废钢层层覆盖，仅露出微小边缘",
            "被上方的废钢件压住，大部分被遮挡",
            "嵌在废钢缝隙中，周围被废钢包围",
            "被周围的废钢挤压，仍露出能辨认的一段",
            "藏在废钢堆深处，被多层废钢遮挡",
            "被一块较大的废钢板遮挡了大半",
            "塞在废钢件的夹缝里，只露出一角",
            "被废钢堆掩埋，仅表面微露",
        ]
    
    @staticmethod
    def get_weathering_descriptions(object_class: str) -> List[str]:
        base = [
            "表面覆盖着厚厚的铁锈和灰尘，整个物体没有一个干净的地方",
            "带有明显的磨损痕迹和划痕，整个物体没有一个干净的地方",
            "布满油污和泥垢，显得陈旧，整个物体没有一个干净的地方",
            "锈迹斑斑，颜色暗淡，整个物体没有一个干净的地方",
            "表面有凹坑和腐蚀痕迹，整个物体没有一个干净的地方",
            "沾染了废钢的锈色和污渍，整个物体没有一个干净的地方",
            "部分油漆剥落，露出锈蚀的金属，整个物体没有一个干净的地方",
            "表面粗糙，有明显的老化迹象，整个物体没有一个干净的地方",
        ]
        specific = {
            "hydraulic_cylinder": [
                "缸体表面有油渍和磨损痕迹，活塞杆有轻微锈蚀",
                "表面有磕碰凹坑，油泥包裹",
                "锈迹与油污混合，呈现工业废件的典型质感",
            ],
            "lpg_tank": [
                "罐体表面有大面积锈蚀，阀门处有陈旧破损",
                "底部有凹陷，表面覆盖着泥土和锈迹",
                "油漆大部分脱落，露出锈蚀的金属表面",
            ],
            "axle": [
                "轴表面有磨损痕迹和锈斑，端部有磕碰",
                "布满油污和铁锈，显得非常陈旧",
                "表面有严重的腐蚀和磨损痕迹",
            ],
            "wire_rope_drum": [
                "卷筒表面有锈迹，缠绕着断裂的钢丝",
                "法兰盘锈蚀严重，表面有油污",
                "钢丝绳残留物和锈迹混合，呈现废置状态",
            ],
            "shell": [
                "弹体表面有铁锈和泥土，尾部有磨损",
                "锈蚀严重，表面有划痕和凹坑",
                "沾满泥土和铁锈，颜色暗淡",
            ],
            "fire_extinguisher": [
                "标签模糊不清，完全辨别不了字迹，表面有划痕和掉漆，锈迹斑斑",
                "标签模糊不清，完全辨别不了字迹，瓶体有磕碰痕迹",
                "标签模糊不清，完全辨别不了字迹，表面布满灰尘和污渍，显得十分陈旧",
            ],
            "pesticide_can":[
                "标签模糊不清，完全辨别不了字迹，表面布满灰尘、划痕和污渍，有明显的掉漆，显得十分陈旧；喷口形状正常，但也有老化痕迹，失去光泽",
            ],
            "spray_paint_can":[
                "标签模糊不清，完全辨别不了字迹，表面布满灰尘、划痕和污渍，有明显的掉漆，显得十分陈旧；喷口形状正常，但也有老化痕迹，失去光泽",
            ],
        }
        if object_class in specific:
            return specific[object_class] + base
        return base
    
    @staticmethod
    def get_lighting_descriptions(lighting: str) -> List[str]:
        if lighting and len(lighting) > 3:
            return [
                f"光照{lighting}，与周围环境一致",
                f"受{lighting}影响，阴影自然",
                f"光影与{lighting}匹配",
            ]
        return [
            "光照自然，与周围废钢的光影一致",
            "光线从上方斜射，阴影落在废钢堆上",
            "漫反射光线，无明显定向光源",
            "顶部照明，废钢堆形成丰富阴影",
        ]
    
    @staticmethod
    def get_tone_descriptions(tone: str) -> List[str]:
        if tone and len(tone) > 3:
            return [
                f"色调{tone}，与废钢环境融合",
                f"色彩{tone}，呈现工业废料的质感",
                f"整体偏向{tone}，和谐统一",
            ]
        return [
            "色调偏暗，呈现工业废料的沉稳质感",
            "以铁锈红、深灰、暗褐色为主",
            "色彩饱和度低，呈现陈旧感",
        ]
    
    @staticmethod
    def get_scenario_descriptions() -> List[str]:
        return [
            "废钢堆积的工业场景",
            "废料堆场的典型环境",
            "工业废钢回收场的一角",
            "重型废钢的杂乱堆叠区域",
        ]
    
    @staticmethod
    def get_embedding_verbs() -> List[str]:
        return [
            "嵌入", "塞入", "放置", "卡在", "夹在", 
            "隐藏在", "埋在", "藏在", "陷在", "挤在"
        ]

# ---------- 类别模板与反向提示词 ----------

def ensure_file(path: Path, default_text: str) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(default_text, encoding="utf-8")

def load_prompt_templates(path: Path) -> Dict[str, List[str]]:
    DEFAULT_PROMPT_TEMPLATES = {
        "fire_extinguisher": [
            "在废钢堆的缝隙中嵌入一个真实大小的红色灭火器，被废钢遮挡只露一小部分，表面有灰尘划痕锈迹。",
            "在废钢堆深处放入一个真实大小的灭火器，被遮挡不显眼，保持灭火器的典型形状，表面旧化带锈。",
            "将一个小型灭火器塞入废钢件的夹缝中，瓶体大部分被遮挡，表面有褪色和锈迹。",
        ],
        "gas_cylinder": [
            "在废钢背景中嵌入一个真实大小的气瓶，被废钢遮挡只露一小部分，保持气瓶的圆柱形和两端特征，表面有锈迹磨损。",
            "在废钢堆深处放入一个真实大小的气瓶，被遮挡不显眼，保持气瓶的原本形态，表面旧化带锈。",
            "将一个小气瓶卡在废钢堆的角落里，被周围的废钢件遮挡，表面有腐蚀痕迹。",
        ],
        "shell": [
            "在废钢背景中嵌入一个真实大小的炮弹，被废钢遮挡只露一小部分，保持炮弹的流线型弹头特征，表面有铁锈泥土。",
            "在废钢堆深处放入一个真实大小的炮弹，被遮挡不显眼，保持炮弹的典型外观，表面旧化带锈。",
            "将一枚小炮弹塞入废钢缝隙中，弹体被废钢覆盖，只露出尾部，表面布满锈迹。",
        ],
        "lpg_tank": [
            "在废钢背景中嵌入一个真实大小的液化气罐，被废钢遮挡只露一小部分，保持液化气罐的罐体形状和阀门结构，表面有凹坑锈迹。",
            "在废钢堆深处放入一个真实大小的液化气罐，被遮挡不显眼，保持液化气罐的形态，表面旧化带锈。",
            "将一个小液化气罐横放在废钢堆中，大部分被废钢覆盖，表面有严重锈蚀。",
        ],
        "accumulator": [
            "在废钢背景中嵌入一个真实大小的蓄能器，被废钢遮挡只露一小部分，保持蓄能器的圆柱形和端部结构，表面有油泥灰尘。",
            "在废钢堆深处放入一个真实大小的蓄能器，被遮挡不显眼，保持蓄能器的原本形态，表面旧化带锈。",
            "将一个小蓄能器嵌入废钢堆的中间层，被周围的废钢件包裹，表面有油污。",
        ],
        "hydraulic_cylinder": [
            "在废钢背景中嵌入一个真实大小的液压缸，被废钢遮挡只露一小部分，保持液压缸的圆柱形和活塞杆特征，表面有油泥划痕。",
            "在废钢堆深处放入一个真实大小的液压缸，被遮挡不显眼，保持液压缸的典型形状，表面旧化带锈。",
            "将一个报废的小液压缸塞入废钢缝隙中，缸体被废钢覆盖，活塞杆露出微小部分，表面有油渍。",
        ],
        "shock_absorber": [
            "在废钢背景中嵌入一个真实大小的减震器，被废钢遮挡只露一小部分，保持减震器的圆柱形和连接结构，表面有锈蚀油渍。",
            "在废钢堆深处放入一个真实大小的减震器，被遮挡不显眼，保持减震器的形态，表面旧化带锈。",
            "将一个小减震器卡在废钢件的夹缝中，大部分被遮挡，表面有磨损和锈迹。",
        ],
        "sealed_container": [
            "在废钢背景中嵌入一个真实大小的密闭箱体，被废钢遮挡只露一小部分，保持箱体的方形结构和棱边特征，表面有泥土锈迹。",
            "在废钢堆深处放入一个真实大小的密闭箱体，被遮挡不显眼，保持箱体的原本形状，表面旧化带锈。",
        ],
        "tire": [
            "在废钢背景中嵌入一个真实大小的轮胎，被废钢遮挡只露一小部分，保持轮胎的圆形和胎纹特征，表面有油污磨损。",
            "在废钢堆深处放入一个真实大小的轮胎，被遮挡不显眼，保持轮胎的典型形态，表面旧化带泥。",
            "将一个小轮胎嵌入废钢堆中，被周围的废钢挤压，只露出部分胎面，表面有磨损痕迹。",
        ],
        "roller": [
            "在废钢背景中嵌入一个真实大小的滚筒，被废钢遮挡只露一小部分，保持滚筒的圆柱形和表面特征，表面有铁锈油泥。",
            "在废钢堆深处放入一个真实大小的滚筒，被遮挡不显眼，保持滚筒的原本形状，表面旧化带锈。",
        ],
        "wire_rope_drum": [
            "在废钢背景中嵌入一个真实大小的钢丝绳卷筒，被废钢遮挡只露一小部分，保持卷筒的圆柱形和法兰结构，表面有锈蚀。",
            "在废钢堆深处放入一个真实大小的钢丝绳卷筒，被遮挡不显眼，保持卷筒的形态，表面旧化带锈。",
            "将一个小钢丝绳卷筒塞入废钢堆的缝隙中，被废钢覆盖，表面缠绕着断裂的钢丝，锈迹斑斑。",
        ],
        "axle": [
            "在废钢背景中嵌入一个真实大小的汽车后桥，被废钢遮挡只露一小部分，保持后桥的轴形和端部结构，表面有油污锈迹。",
            "在废钢堆深处放入一个真实大小的汽车后桥，被遮挡不显眼，保持后桥的典型形状，表面旧化带锈。",
            "将一个小型后桥轴嵌入废钢堆中，被废钢件压住，只露出轴端，表面有油污和锈蚀。",
        ],
        "reducer": [
            "在废钢背景中嵌入一个真实大小的减速机，被废钢遮挡只露一小部分，保持减速机的箱体形状和连接轴，表面有灰尘油污。",
            "在废钢堆深处放入一个真实大小的减速机，被遮挡不显眼，保持减速机的形态，表面旧化带锈。",
        ],
        "track_roller": [
            "在废钢背景中嵌入一个真实大小的支重轮，被废钢遮挡只露一小部分，保持支重轮的圆形和轮缘特征，表面有铁锈泥土。",
            "在废钢堆深处放入一个真实大小的支重轮，被遮挡不显眼，保持支重轮的原本形状，表面旧化带锈。",
        ],
        "drive_sprocket": [
            "在废钢背景中嵌入一个真实大小的驱动轮，被废钢遮挡只露一小部分，保持驱动轮的圆形和齿形特征，表面有锈迹磨损。",
            "在废钢堆深处放入一个真实大小的驱动轮，被遮挡不显眼，保持驱动轮的典型形态，表面旧化带锈。",
        ],
        "wheel_end": [
            "在废钢背景中嵌入一个真实大小的轮边总成，被废钢遮挡只露一小部分，保持轮边总成的盘形和结构特征，表面有油泥灰尘。",
            "在废钢堆深处放入一个真实大小的轮边总成，被遮挡不显眼，保持轮边总成的形态，表面旧化带锈。",
        ],
        "guard_block": [
            "在废钢背景中嵌入一个真实大小的防护墩，被废钢遮挡只露一小部分，保持防护墩的方形和棱角特征，表面有破损泥土。",
            "在废钢堆深处放入一个真实大小的防护墩，被遮挡不显眼，保持防护墩的原本形状，表面旧化带泥。",
        ],
    }
    ensure_file(path, json.dumps(DEFAULT_PROMPT_TEMPLATES, ensure_ascii=False, indent=2))
    return json.loads(path.read_text(encoding="utf-8"))

def load_negative_prompt(path: Path) -> str:
    DEFAULT_NEGATIVE_PROMPT = (
        "不要改变原始背景，不要生成卡通风格，不要生成干净崭新的异物，"
        "不要让异物悬浮，不要把异物画成玩具或缩得比真实厘米更小，不要画得比真实参照长度更大，"
        "不要改变异物的基本形状和结构特征，必须保持物体原本的形态，"
        "禁止生成反光强烈、亮面、光滑的物体，纹理必须与废钢一致，"
        "异物必须被废钢遮挡大部分，但仍露出能辨认的一段。"
        "禁止对异物进行任何形式的变形、拉伸、扭曲或比例改变。"
        "不要生成看起来像是人工放置的物体，应该看起来像是自然存在于废钢堆中。"
        "不要放在很显眼的地方，不要让异物成为画面的视觉焦点，必须与背景完美融合。"
        "不要柔化整张图，不要降低整张图的分辨率，不要二次渲染导致发糊。"
        "饱和度必须发灰、发浊、褪色，并且明显低于周围废钢。禁止鲜红、橘红、亮黄、翠绿、纯蓝、荧光色。"
        "异物不能比废钢更锐，也不能比监控原图的颗粒更细。"
    )
    ensure_file(path, DEFAULT_NEGATIVE_PROMPT)
    return path.read_text(encoding="utf-8").strip()

# ---------- 尺寸句、位置句、评审建议清洗 ----------

def round_size_spec(
    cfg: Optional[AppConfig],
    object_class: Optional[str],
    target_ratio: Optional[float] = None,
) -> Dict[str, Any]:
    """返回当前物品工作尺寸和本轮可见占比，兼容旧类别配置。"""
    profile = getattr(cfg, "active_object_profile", None) if cfg is not None else None
    if profile and profile["category"] == object_class:
        return per_item_size_spec(profile, target_ratio)
    name = object_class or "异物"
    low, high = object_scale_range(cfg, object_class) if cfg is not None else (0.06, 0.06)
    ratio = target_ratio if target_ratio is not None else (low + high) / 2
    ratio = min(max(float(ratio), low * 0.85), high * 1.05)
    target_cm = ratio * HOPPER_FOV_M * 100.0
    fov_cm = HOPPER_FOV_M * 100.0
    brief = lookup_size_brief(cfg, object_class)
    if brief is None:
        text = (
            f"【本轮设定尺寸】{name}按真实物体绘制，不要画成玩具。"
            f"本轮埋入后可见最长边约{target_cm:.0f}厘米，约占画面宽度的{ratio:.1%}。"
            f"车斗水平视野约{fov_cm:.0f}厘米。"
        )
        return {
            "object_class": name,
            "target_ratio": ratio,
            "target_cm": target_cm,
            "diameter_cm": None,
            "diameter_to_height": None,
            "cross_name": "筒径",
            "scale_min": low,
            "scale_max": high,
            "size_brief": text,
            "real_height_cm": None,
        }
    real_h = brief["height_cm"]
    real_d = brief["diameter_cm"]
    d_to_h = real_d / real_h
    part_bits = [f"{part['name']}{part['cm']:.0f}厘米" for part in brief["parts"]]
    parts_text = ("，" + "，".join(part_bits)) if part_bits else ""
    expose = brief["expose"] or "可以埋入废钢，只露出一段"
    forbid = brief["forbid"] or "不要画得比真实件更大，也不要缩成玩具"
    text = (
        f"【本轮设定尺寸】{name}按真实件尺寸绘制，不要把筒径和结构缩成模型。"
        f"参照长度{real_h:.0f}厘米，{brief['cross_name']}{real_d:.0f}厘米，直径/高度={d_to_h:.2f}"
        f"{parts_text}。"
        f"车斗水平视野约{fov_cm:.0f}厘米。"
        f"本轮埋入后可见最长边约{target_cm:.0f}厘米，约占画面宽度的{ratio:.1%}。"
        f"筒径和其余结构保持上面的真实厘米，不要按可见边再缩小。"
        f"{expose}。{forbid}。可见最长边不要明显超过参照长度{real_h:.0f}厘米。"
    )
    return {
        "object_class": name,
        "target_ratio": ratio,
        "target_cm": target_cm,
        "diameter_cm": real_d,
        "diameter_to_height": d_to_h,
        "cross_name": brief["cross_name"],
        "scale_min": low,
        "scale_max": high,
        "size_brief": text,
        "real_height_cm": real_h,
        "real_diameter_cm": real_d,
        "parts": [{"name": part["name"], "cm": part["cm"]} for part in brief["parts"]],
        "expose": expose,
        "forbid": forbid,
    }

def class_size_clause(
    cfg: Optional[AppConfig],
    object_class: Optional[str] = None,
    target_ratio: Optional[float] = None,
) -> str:
    return round_size_spec(cfg, object_class, target_ratio)["size_brief"]

_HISTORY_WORDS = ("上一轮", "测得", "历程", "返回尺寸", "前几轮", "前两轮", "第1轮", "第2轮", "第3轮")

def keep_only_current_size(text: str, spec: Dict[str, Any]) -> str:
    """生图提示词只留本轮设定。删掉往轮设定和返回尺寸。"""
    current_pct = f"{spec['target_ratio']:.1%}"
    pieces = re.split(r"(?<=[。！？\n])", text or "")
    kept = []
    for piece in pieces:
        if any(word in piece for word in _HISTORY_WORDS):
            continue
        percents = re.findall(r"\d+(?:\.\d+)?%", piece)
        if any(item != current_pct for item in percents):
            continue
        kept.append(piece)
    body = "".join(kept).strip()
    brief = spec["size_brief"]
    if brief not in body:
        body = f"{body}\n{brief}".strip() if body else brief
    return body

def format_size_history(size_rounds: List[Dict[str, Any]]) -> str:
    """只给 VLM。每一轮的设定尺寸和返回尺寸。"""
    if not size_rounds:
        return "【尺寸历程】尚无已完成轮次。"
    lines = ["【尺寸历程】"]
    for item in size_rounds:
        set_pct = item.get("target_ratio")
        set_cm = item.get("target_cm")
        cross = item.get("cross_name") or "筒径"
        diameter = item.get("diameter_cm")
        visible = item.get("visible_ratio")
        visible_cm = item.get("visible_cm")
        judgment = item.get("size_judgment") or "未知"
        set_txt = f"{set_pct:.1%}（约{set_cm:.0f}厘米）" if isinstance(set_pct, float) else "未知"
        dia_txt = f"，{cross}约{diameter:.1f}厘米" if isinstance(diameter, float) else ""
        if isinstance(visible, float):
            back_cm = f"，约{visible_cm:.0f}厘米" if isinstance(visible_cm, float) else ""
            back_txt = f"返回可见最长边{visible:.1%}{back_cm}"
        else:
            back_txt = "返回尺寸未知"
        lines.append(
            f"第{item.get('round')}轮设定可见最长边{set_txt}{dia_txt}；{back_txt}；判定{judgment}。"
        )
    return "\n".join(lines)

def describe_sampled_position(
    xy: Tuple[float, float],
    center: Tuple[float, float],
) -> str:
    x, y = xy
    dx, dy = x - center[0], y - center[1]
    horizontal = "右侧" if dx >= 0 else "左侧"
    vertical = "下方" if dy >= 0 else "上方"
    quad_h = "右" if x >= 0.5 else "左"
    quad_v = "下" if y >= 0.5 else "上"
    return (
        f"画面{quad_h}{quad_v}区域的废钢堆内部（不含墙壁），"
        f"大约水平{x * 100:.0f}%、垂直{y * 100:.0f}%处（原点为左上角），"
        f"相对废钢几何中心偏{horizontal}{vertical}。"
        f"严禁放在画面正中央，也严禁贴在车壁或挡板上"
    )

_WHOLE_IMAGE_BLUR_HINTS = (
    "柔化整", "整张图模糊", "整图模糊", "降低清晰", "降清晰",
    "降低分辨率", "降分辨率", "降低锐度", "二次渲染", "把画面弄糊",
)

def sanitize_vlm_suggestion(text: str) -> str:
    """去掉让整张图发糊的建议，标签文字模糊可以保留。"""
    if not text:
        return ""
    parts = re.split(r"[。；;\n]+", str(text))
    kept = []
    for part in parts:
        piece = part.strip()
        if not piece:
            continue
        if "标签" in piece or "字迹" in piece or "文字" in piece:
            kept.append(piece)
            continue
        if any(hint in piece for hint in _WHOLE_IMAGE_BLUR_HINTS):
            continue
        kept.append(piece)
    return "。".join(kept).strip()

# ============================================================
# 目录、采样、接口和主流程。
# ============================================================

FUSION_SCORE_AXES = (
    "weathering_score",
    "lighting_score",
    "occlusion_score",
    "style_score",
    "background_preservation_score",
    "identity_score",
)

QUALITY_THRESHOLD = 85.0

def quality_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    scores: dict[str, float] = {}
    reasons = []
    for key in FUSION_SCORE_AXES:
        try:
            score = float(evaluation.get(key, 0))
            if not math.isfinite(score):
                score = 0.0
        except (TypeError, ValueError):
            score = 0.0
        score = min(100.0, max(0.0, score))
        scores[key] = score
        if score <= QUALITY_THRESHOLD:
            reasons.append(f"{key}_not_above_85")
    if evaluation.get("object_present") is not True:
        reasons.append("object_presence_not_confirmed")
    for key in (
        "new_appearance_detected",
        "pasted_appearance_detected",
        "material_violation",
    ):
        if evaluation.get(key) is not False:
            reasons.append(key)
    return {
        "passed": not reasons,
        "scores": scores,
        "minimum_score": min(scores.values()),
        "reasons": reasons,
        "evidence_type": "vlm_visual_opinion_not_human_acceptance",
    }

def edit_region(
    xy: list[float], ratio: float, image_size: tuple[int, int], cross_ratio: float
) -> list[float]:
    """Return a normalized interactive edit hint, including support and shadow context."""
    width, height = image_size
    box_w = min(0.72, max(0.14, ratio * 1.55 + 0.03))
    box_h = min(
        0.60, max(0.16, ratio * (0.60 + min(1.0, cross_ratio)) * width / height)
    )
    cx = min(0.96 - box_w / 2, max(0.04 + box_w / 2, float(xy[0])))
    cy = min(0.84 - box_h / 2, max(0.10 + box_h / 2, float(xy[1])))
    return [
        round(cx - box_w / 2, 5),
        round(cy - box_h / 2, 5),
        round(cx + box_w / 2, 5),
        round(cy + box_h / 2, 5),
    ]

def load_catalog(
    path: str | Path, reference_root: str | Path | None = None
) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != "1.0" or not isinstance(data.get("objects"), list):
        raise ValueError("物品目录格式不符合 1.0 约定")
    catalog_path = Path(path).resolve()
    root = Path(reference_root or data["reference_root"]).expanduser()
    if not root.is_absolute():
        root = catalog_path.parent.parent / root
    root = root.resolve()
    ids, names = set(), set()
    for profile in data["objects"]:
        identifier = profile["id"]
        filename = profile["reference_filename"]
        if identifier in ids or filename in names:
            raise ValueError("物品 ID 和参考图文件名必须唯一")
        ids.add(identifier)
        names.add(filename)
        reference = (root / filename).resolve()
        if not reference.is_relative_to(root) or not reference.is_file():
            raise ValueError(f"找不到目录中的参考图：{filename}")
        policy = profile["size_policy"]
        if policy["reference_length_mm"] <= 0 or policy["cross_mm"] <= 0:
            raise ValueError(f"{identifier} 的尺寸必须为正")
        low, high = policy["visible_ratio_range"]
        if not 0 < low <= high < 1:
            raise ValueError(f"{identifier} 的可见比例范围无效")
    data["reference_root"] = str(root)
    return data

def catalog_items(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "path": Path(catalog["reference_root"]) / profile["reference_filename"],
            "class": profile["category"],
            "weight": profile.get("sampling_weight", 1.0),
            "filename": profile["reference_filename"],
            "profile_id": profile["id"],
            "profile": profile,
        }
        for profile in catalog["objects"]
    ]

def select_profile(catalog: dict[str, Any], identifier: str) -> dict[str, Any]:
    for profile in catalog["objects"]:
        if identifier in (
            profile["id"],
            profile["reference_filename"],
            profile.get("original_filename"),
        ):
            return profile
    raise ValueError(f"目录中没有该物品：{identifier}")

def profile_brief(profile: dict[str, Any]) -> dict[str, Any]:
    policy = profile["size_policy"]
    return {
        "height_cm": policy["reference_length_mm"] / 10,
        "diameter_cm": policy["cross_mm"] / 10,
        "cross_name": policy["cross_name"],
        "expose": profile["exposure_hint"],
        "forbid": "；".join(profile["forbidden_changes"]),
        "parts": [
            {"name": name, "cm": value / 10}
            for name, value in profile["parts_mm"].items()
        ],
        "evidence_status": profile["dimension_evidence"]["status"],
        "physical_name": profile["name"],
        "profile_id": profile["id"],
    }

per_item_catalog_items = catalog_items

per_item_evaluation_prompt = evaluation_user_prompt

per_item_evaluator_prompt = evaluator_prompt

load_per_item_catalog = load_catalog

per_item_brief = profile_brief

render_per_item_prompt = render_prompt

per_item_size_spec = size_spec

fusion_quality_gate = quality_gate

fusion_edit_region = edit_region

# ============ 强制禁用SSL验证 ============
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
requests.packages.urllib3.disable_warnings()
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['CURL_CA_BUNDLE'] = ''

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# ============ 自定义SSL适配器 ============

class SSLAdapter(requests.adapters.HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        kwargs['ssl_context'] = ctx
        return super().init_poolmanager(*args, **kwargs)

def get_session_with_ssl_bypass():
    session = requests.Session()
    adapter = SSLAdapter()
    session.mount('https://', adapter)
    session.verify = False
    
    retry_strategy = urllib3.Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"]
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    return session

# ============ Prompt分析器类 ============

class PromptAnalyzer:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.good_prompts_file = output_dir / "good_prompts.json"
        self.bad_prompts_file = output_dir / "bad_prompts.json"
        self.prompt_stats_file = output_dir / "prompt_stats.json"
        
        self.good_prompts: List[Dict] = []
        self.bad_prompts: List[Dict] = []
        
        self._load_data()
    
    def _load_data(self):
        if self.good_prompts_file.exists():
            try:
                with open(self.good_prompts_file, 'r', encoding='utf-8') as f:
                    self.good_prompts = json.load(f)
            except:
                self.good_prompts = []
        
        if self.bad_prompts_file.exists():
            try:
                with open(self.bad_prompts_file, 'r', encoding='utf-8') as f:
                    self.bad_prompts = json.load(f)
            except:
                self.bad_prompts = []
    
    def add_prompt_record(self, prompt: str, image_id: str, object_class: str, 
                          size_judgment: str, confidence: float, 
                          quality_score: float, fusion_quality: str,
                          round_num: int = 0,
                          full_prompt: str = "", fusion_passed: Optional[bool] = None):
        record = {
            "image_id": image_id,
            "object_class": object_class,
            "size_judgment": size_judgment,
            "confidence": confidence,
            "quality_score": quality_score,
            "fusion_quality": fusion_quality,
            "round_num": round_num,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt_text": prompt,
            "full_prompt": full_prompt if full_prompt else prompt,
            "prompt_length": len(prompt),
            "fusion_passed": fusion_passed,
        }
        
        is_good = (
            size_judgment == "acceptable" and 
            confidence >= 0.5 and 
            quality_score >= 0.5
            and fusion_passed is not False
        )
        
        if is_good:
            self.good_prompts.append(record)
            print(f"  [Prompt分析] ✅ 好Prompt已保存 (第{len(self.good_prompts)}个)")
        else:
            self.bad_prompts.append(record)
            print(f"  [Prompt分析] ❌ 差Prompt已保存 (第{len(self.bad_prompts)}个)")
        
        self._save_prompts()
    
    def _save_prompts(self):
        with open(self.good_prompts_file, 'w', encoding='utf-8') as f:
            json.dump(self.good_prompts, f, ensure_ascii=False, indent=2)
        
        with open(self.bad_prompts_file, 'w', encoding='utf-8') as f:
            json.dump(self.bad_prompts, f, ensure_ascii=False, indent=2)
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_records": len(self.good_prompts) + len(self.bad_prompts),
            "good_count": len(self.good_prompts),
            "bad_count": len(self.bad_prompts),
            "good_prompts_file": str(self.good_prompts_file),
            "bad_prompts_file": str(self.bad_prompts_file),
            "last_update": datetime.now(timezone.utc).isoformat()
        }
    
    def save_stats(self):
        stats = self.get_stats()
        with open(self.prompt_stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    
    def print_summary(self):
        stats = self.get_stats()
        print("\n" + "="*60)
        print("📊 Prompt分析统计")
        print("="*60)
        print(f"总记录数: {stats['total_records']}")
        print(f"好Prompt: {stats['good_count']}")
        print(f"差Prompt: {stats['bad_count']}")
        
        if self.good_prompts:
            print(f"\n✅ 好的Prompt示例 (最新的3条):")
            for i, p in enumerate(self.good_prompts[-3:]):
                print(f"  {i+1}. [{p['object_class']}] {p['prompt_text'][:150]}...")
        
        if self.bad_prompts:
            print(f"\n❌ 差的Prompt示例 (最新的3条):")
            for i, p in enumerate(self.bad_prompts[-3:]):
                print(f"  {i+1}. [{p['object_class']}] {p['prompt_text'][:150]}...")
        print("="*60)

# ============ 配置类 ============

@dataclass
class AppConfig:
    image_api_key: str = field(repr=False)
    model: str
    image_api_url: str
    image_protocol: str
    size: str
    output_format: str
    watermark: bool
    max_images: int
    timeout_seconds: int
    sleep_seconds: float
    total_num: int
    seed: Optional[int]
    
    object_resize_scale: float
    object_resize_scales: Dict[str, Tuple[float, float]]
    object_size_briefs: Dict[str, Dict[str, Any]]
    object_canvas_scale: float
    object_opacity: float
    object_anchor: str
    
    backgrounds_dir: Path
    objects_dir: Path
    prompt_templates_path: Path
    negative_prompt_path: Path
    output_images_dir: Path
    output_final_images_dir: Path
    output_metadata_dir: Path
    output_failed_dir: Path
    output_bbox_dir: Path
    output_analysis_dir: Path
    
    vlm_api_key: str = field(repr=False)
    vlm_api_url: str
    vlm_model: str
    enable_vlm_optimization: bool = True
    max_optimization_rounds: int = 3
    size_tolerance: str = "class_range"
    feed_object_views: bool = True
    max_object_views: int = 3
    six_views_dir: Optional[Path] = None
    object_catalog_path: Optional[Path] = None
    object_catalog: Optional[Dict[str, Any]] = None
    active_object_profile: Optional[Dict[str, Any]] = None
    background_catalog: Optional[Dict[str, Any]] = None
    active_background_profile: Optional[Dict[str, Any]] = None
    active_scene_context: Optional[Dict[str, Any]] = None
    active_edit_region: Optional[List[float]] = None
    active_position_xy: Optional[List[float]] = None
    use_interactive_edit_region: bool = True
    vlm_thinking_mode: str = "disabled"
    prepare_weathered_reference: bool = True
    active_weathered_reference: Optional[Path] = None
    active_reference_kind: str = "six_view_atlas"
    active_reference_preparation: Optional[Dict[str, Any]] = None
    weathered_reference_rounds: int = 2
    
    vlm_evaluator_prompt: str = field(default_factory=lambda: (
        "你是废钢车斗异物质检员。只输出有效JSON。车斗水平视野约240厘米。\n"
        "先估计可见最长边占画面宽度的 visible_ratio，再换成厘米：visible_cm = visible_ratio × 240。\n"
        "尺寸以该类真实厘米为准，包括参照长度、筒径或主横截面，以及握把、喷管、活塞杆、耳环、法兰等结构。\n"
        "不要把真实厘米再缩小成模型。遮挡可以挡住一部分，但露出部分必须仍是真实尺度。\n"
        "太大：可见最长边的厘米明显大于真实参照长度，或画得比真实件更大。\n"
        "太小：没放进去，或露出部分明显小于真实横截面，像缩小的玩具。\n"
        "合格：露出部分落在真实横截面到真实参照长度之间，结构比例符合真实直径/高度。\n"
        "看不见全长是埋入，不要因此判太小；能看见也不要因此判太大。\n"
        "小件如果画得达到更大类别的真实尺寸，判定为太大。\n"
        "文字只要求模糊异物标签；禁止建议柔化整张图或降低分辨率。\n"
        "sharpness 只记整图是否发糊（sharp|ok|soft），不要靠缩小异物来修清晰度。\n"
        "另评五项外观，每项只评自己的内容，取值 bad|ok|good："
        "color_temperature_match 只看白平衡和冷暖；"
        "tone_match 只看色相和明度，不看饱和度；"
        "saturation_match 必须发灰且明显低于废钢，鲜红、橘红、亮黄、翠绿、纯蓝、荧光或与废钢同样鲜艳为 bad；"
        "pixel_match 只看颗粒、压缩块和局部对比；"
        "sharpness_match 只看失焦、拖影和边缘软硬，异物更锐为 bad。\n"
        "tone_fix 只改色相和明度，不要把饱和度写进 tone_fix。\n"
        "另给三项 0 到 100 的整数分：tone_score 色调，size_score 大小，overall_score 整体效果。\n"
        "三项都高于 85 才算这轮合格。每项给一句修正：tone_fix、size_fix、overall_fix。\n"
        "输出JSON：{\"visible_ratio\": 0.0, \"size_judgment\": \"too_large|acceptable|too_small\", "
        "\"tone_score\": 0, \"size_score\": 0, \"overall_score\": 0, "
        "\"tone_fix\": \"\", \"size_fix\": \"\", \"overall_fix\": \"\", "
        "\"color_temperature_match\": \"bad|ok|good\", \"tone_match\": \"bad|ok|good\", "
        "\"saturation_match\": \"bad|ok|good\", \"pixel_match\": \"bad|ok|good\", "
        "\"sharpness_match\": \"bad|ok|good\", "
        "\"sharpness\": \"sharp|ok|soft\", \"confidence\": 0.0, \"reason\": \"\", "
        "\"improvement_suggestion\": \"\", \"quality_score\": 0.0, "
        "\"fusion_quality\": \"very bad|bad|ok|good|very good\"}"
    ))

# ============ 尺寸计算工具 ============

# 废钢车斗水平视野约 2.4 m，用于把宽度百分比换成厘米
HOPPER_FOV_M = 2.4

def image_sharpness_score(path: Path) -> float:
    """Laplacian 方差，数值越高越锐。用于发现后轮二次渲染发糊。"""
    img = Image.open(path).convert("L")
    img = img.resize(
        (max(1, img.width // 4), max(1, img.height // 4)),
        Image.Resampling.BOX,
    )
    edge = img.filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=0))
    pix = list(edge.getdata())
    if not pix:
        return 0.0
    mean = sum(pix) / len(pix)
    return sum((p - mean) ** 2 for p in pix) / len(pix)

OBJECT_SCALE_ALIASES = {
    "fire_extinguisher": "灭火器",
    "gas_cylinder": "气瓶",
    "hydraulic_cylinder": "液压缸",
    "shock_absorber": "减震器",
    "accumulator": "集油器",
    "sealed_container": "储罐",
    "filter": "分离过滤器",
    "jack": "千斤顶",
    "pesticide_can": "杀虫剂罐",
    "pneumatic_cylinder": "气缸",
    "spray_paint_can": "自喷漆罐",
    "ball_valve": "球阀",
    "bearing_housing": "轴承座",
    "sealant_can": "密封胶罐",
}

def parse_object_resize_scales(raw: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    """读取 config.yaml 中每个物体的 [min, max] 或单值，单位为可见最长边/画面宽度。"""
    fallback = float(raw.get("object_resize_scale", 0.022))
    parsed: Dict[str, Tuple[float, float]] = {}
    data = raw.get("object_resize_scales") or {}
    if not isinstance(data, dict):
        raise ValueError("object_resize_scales 必须是按物体类别分组的字典")
    for key, value in data.items():
        name = str(key).strip()
        if not name:
            raise ValueError("object_resize_scales 含有空类别名")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            low = high = float(value)
        elif isinstance(value, dict):
            low = float(value.get("min", fallback))
            high = float(value.get("max", low))
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            low, high = float(value[0]), float(value[1])
        else:
            raise ValueError(f"物体 {name} 的缩放尺寸格式无效，请使用 [min, max]")
        if low > high:
            low, high = high, low
        if not 0 < low <= high < 1:
            raise ValueError(f"物体 {name} 的缩放尺寸须满足 0 < min <= max < 1，当前 {low}–{high}")
        parsed[name] = (low, high)
    return parsed

def actual_scale_range_from_brief(brief: Dict[str, Any]) -> Tuple[float, float]:
    """真实横截面到真实参照长度，换成占 240 厘米车斗视野的比例。"""
    fov_cm = HOPPER_FOV_M * 100.0
    length_cm = float(brief["height_cm"])
    cross_cm = float(brief["diameter_cm"])
    if "轴孔" in str(brief.get("cross_name") or "") or "孔径" in str(brief.get("cross_name") or ""):
        outers = [
            float(part["cm"]) for part in brief.get("parts") or []
            if "孔" not in str(part.get("name") or "")
        ]
        if outers:
            ordered = sorted(outers)
            cross_cm = ordered[len(ordered) // 2]
    low_cm = min(length_cm, cross_cm)
    high_cm = max(length_cm, cross_cm)
    low = max(0.008, low_cm / fov_cm * 0.85)
    high = min(0.95, high_cm / fov_cm * 1.10)
    if low > high:
        low, high = high, low
    return (low, high)

def object_scale_range(cfg: AppConfig, object_class: Optional[str] = None) -> Tuple[float, float]:
    profile = getattr(cfg, "active_object_profile", None)
    if profile and profile["category"] == object_class:
        return tuple(profile["size_policy"]["visible_ratio_range"])
    fallback = (cfg.object_resize_scale, cfg.object_resize_scale)
    if not object_class:
        return fallback
    brief = lookup_size_brief(cfg, object_class)
    if brief:
        return actual_scale_range_from_brief(brief)
    scales = cfg.object_resize_scales or {}
    if object_class in scales:
        return scales[object_class]
    canonical = OBJECT_SCALE_ALIASES.get(object_class)
    if canonical and canonical in scales:
        return scales[canonical]
    for name, rng in scales.items():
        if name and (name in object_class or object_class in name):
            return rng
    return fallback

def sample_object_resize_scale(cfg: AppConfig, object_class: Optional[str] = None) -> float:
    low, high = object_scale_range(cfg, object_class)
    return random.uniform(low, high) if high > low else low

def parse_object_size_briefs(raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """读取每类真实高度、主横截面尺寸和其余结构尺寸。"""
    data = raw.get("object_size_briefs") or {}
    if not isinstance(data, dict):
        raise ValueError("object_size_briefs 必须是按物体类别分组的字典")
    parsed: Dict[str, Dict[str, Any]] = {}
    for key, value in data.items():
        name = str(key).strip()
        if not name or not isinstance(value, dict):
            raise ValueError(f"物体 {name or key} 的尺寸说明格式无效")
        height_cm = float(value.get("height_cm") or 0)
        diameter_cm = float(value.get("diameter_cm") or 0)
        if height_cm <= 0 or diameter_cm <= 0:
            raise ValueError(f"物体 {name} 需要正的 height_cm 和 diameter_cm")
        parts = []
        for part in value.get("parts") or []:
            if not isinstance(part, dict):
                continue
            part_name = str(part.get("name") or "").strip()
            part_cm = float(part.get("cm") or 0)
            if part_name and part_cm > 0:
                parts.append({"name": part_name, "cm": part_cm})
        parsed[name] = {
            "height_cm": height_cm,
            "diameter_cm": diameter_cm,
            "cross_name": str(value.get("cross_name") or "筒径").strip(),
            "expose": str(value.get("expose") or "").strip(),
            "forbid": str(value.get("forbid") or "").strip(),
            "parts": parts,
        }
    return parsed

def lookup_size_brief(cfg: Optional[AppConfig], object_class: Optional[str]) -> Optional[Dict[str, Any]]:
    if cfg is None or not object_class:
        return None
    profile = getattr(cfg, "active_object_profile", None)
    if profile and profile["category"] == object_class:
        return per_item_brief(profile)
    briefs = cfg.object_size_briefs or {}
    if object_class in briefs:
        return briefs[object_class]
    canonical = OBJECT_SCALE_ALIASES.get(object_class)
    if canonical and canonical in briefs:
        return briefs[canonical]
    for name, brief in briefs.items():
        if name and (name in object_class or object_class in name):
            return brief
    return None

def compute_next_target_ratio(spec: Dict[str, Any], evaluation: Dict[str, Any]) -> float:
    """按本轮设定和画面返回的可见比例计算下一轮设定。径高比不变，筒径随可见边同比变化。"""
    current = float(spec["target_ratio"])
    low = float(spec["scale_min"])
    high = float(spec["scale_max"])
    judgment = evaluation.get("size_judgment")
    measured = evaluation.get("visible_ratio")
    measured_f = float(measured) if isinstance(measured, (int, float)) and measured is not None else None
    if judgment == "too_large" and measured_f and measured_f > current > 0:
        nxt = current * (current / measured_f)
        return min(max(nxt, low * 0.88), current)
    if judgment == "too_small":
        if measured_f and measured_f > 0:
            nxt = min(high, max(current * 1.2, measured_f * 1.35))
        else:
            nxt = min(high * 1.12, current * 1.25)
        return min(high * 1.12, max(nxt, current))
    if judgment == "too_large":
        return max(low * 0.88, current * 0.75)
    return min(max(current, low), high)

def calculate_optimal_size(
    background_path: Path,
    object_class: Optional[str] = None,
    cfg: Optional[AppConfig] = None,
    target_ratio: Optional[float] = None,
) -> Dict[str, Any]:
    bg_img = Image.open(background_path)
    bg_w, bg_h = bg_img.size
    scale_min, scale_max = (
        object_scale_range(cfg, object_class) if cfg is not None
        else (0.016, 0.034)
    )
    if target_ratio is None:
        target_ratio = (
            sample_object_resize_scale(cfg, object_class) if cfg is not None
            else random.uniform(scale_min, scale_max)
        )
    target_ratio = min(max(target_ratio, scale_min * 0.8), scale_max * 1.15)
    target_width = max(16, int(round(bg_w * target_ratio)))
    target_height = max(16, int(round(bg_h * target_ratio)))
    result = {
        "bg_size": (bg_w, bg_h),
        "target_width": target_width,
        "target_height": target_height,
        "target_ratio": target_ratio,
        "scale_min": scale_min,
        "scale_max": scale_max,
        "object_class": object_class,
        "description": (
            f"{object_class or '异物'}可见最长边约占画面宽度的{target_ratio*100:.1f}%"
            f"（约{target_ratio * HOPPER_FOV_M * 100:.0f}厘米，配置 {scale_min*100:.1f}–{scale_max*100:.1f}%）"
        ),
    }
    profile = getattr(cfg, "active_object_profile", None) if cfg is not None else None
    if profile:
        result["profile_id"] = profile["id"]
        result["dimension_evidence"] = profile["dimension_evidence"]
        result["description"] = (
            f"{profile['id']} {profile['name']}主体可见最长边目标{target_ratio:.2%}；"
            f"允许{scale_min:.2%}–{scale_max:.2%}；背景尺度尚未标定，厘米为工作假设。"
        )
    return result

# ============ 废钢位置函数（几何中心 + 四角高斯场） ============

POSITION_FIELD_GRID = 64

def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))

def _parse_xy(value: Any) -> Tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        x, y = float(value[0]), float(value[1])
    elif isinstance(value, dict):
        x = float(value.get("x", value.get("cx", 0.5)))
        y = float(value.get("y", value.get("cy", 0.5)))
    else:
        raise ValueError(f"无法解析坐标: {value}")
    if x > 1.5 or y > 1.5:
        x, y = x / 100.0, y / 100.0
    return _clamp01(x), _clamp01(y)

def _order_corners_around_center(
    center: Tuple[float, float],
    corners: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    cx, cy = center
    return sorted(corners, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

def _point_in_polygon(x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside

def assemble_scrap_geometry(
    center: Tuple[float, float],
    corners: List[Tuple[float, float]],
) -> Dict[str, Any]:
    """对角点中点尽量重合到几何中心；四角按绕中心排序。"""
    if len(corners) != 4:
        raise ValueError(f"需要 4 个角落点，实际 {len(corners)}")
    ordered = _order_corners_around_center(center, corners)
    mid_02 = (
        (ordered[0][0] + ordered[2][0]) / 2.0,
        (ordered[0][1] + ordered[2][1]) / 2.0,
    )
    mid_13 = (
        (ordered[1][0] + ordered[3][0]) / 2.0,
        (ordered[1][1] + ordered[3][1]) / 2.0,
    )
    diagonal_center = (
        (mid_02[0] + mid_13[0]) / 2.0,
        (mid_02[1] + mid_13[1]) / 2.0,
    )
    # 两组对角中点的平均，使几何中心尽量成为对角关系的中点
    final_center = (
        _clamp01(diagonal_center[0]),
        _clamp01(diagonal_center[1]),
    )
    ordered = _order_corners_around_center(final_center, ordered)
    gaussian_centers = [
        ((c[0] + final_center[0]) / 2.0, (c[1] + final_center[1]) / 2.0)
        for c in ordered
    ]
    return {
        "center": final_center,
        "corners": ordered,
        "diagonal_midpoints": [mid_02, mid_13],
        "gaussian_centers": gaussian_centers,
        "vlm_center": center,
    }

def fallback_scrap_geometry() -> Dict[str, Any]:
    """无 VLM 结果时：略偏画面下方的废钢区域，避开常见上沿车壁。"""
    return assemble_scrap_geometry(
        center=(0.50, 0.58),
        corners=[(0.16, 0.26), (0.84, 0.26), (0.88, 0.90), (0.12, 0.90)],
    )

def geometry_from_vlm_reply(result: Any) -> Dict[str, Any]:
    """从同一段对话的首轮回复里取几何；失败时用回退四边形。"""
    try:
        if not isinstance(result, dict) or "center" not in result or "corners" not in result:
            raise ValueError(f"VLM 未返回几何字段: {result}")
        center = _parse_xy(result["center"])
        raw_corners = result["corners"]
        if not isinstance(raw_corners, list) or len(raw_corners) < 4:
            raise ValueError(f"角落点数量不足: {raw_corners}")
        corners = [_parse_xy(item) for item in raw_corners[:4]]
        geometry = assemble_scrap_geometry(center, corners)
        print(
            f"  [位置] 废钢几何中心=({geometry['center'][0]:.3f},{geometry['center'][1]:.3f}) "
            f"四角={[(round(x, 3), round(y, 3)) for x, y in geometry['corners']]}"
        )
        return geometry
    except Exception as exc:
        print(f"  [位置] VLM 几何识别失败，使用回退四边形: {exc}")
        return fallback_scrap_geometry()

def _edge_length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def gaussian_center_weights(corners: List[Tuple[float, float]]) -> List[float]:
    """
    相对边按长度比加权。角点绕中心排序后：
    对边 (0-1) 与 (2-3)、(1-2) 与 (3-0)；
    较长的一边提高其相邻两个高斯中心的比重。
    """
    lengths = [
        _edge_length(corners[i], corners[(i + 1) % 4])
        for i in range(4)
    ]
    l01, l12, l23, l30 = lengths
    r01 = l01 / (l01 + l23 + 1e-12)
    r23 = l23 / (l01 + l23 + 1e-12)
    r12 = l12 / (l12 + l30 + 1e-12)
    r30 = l30 / (l12 + l30 + 1e-12)
    raw = [
        r01 * r30,  # 中心0 邻边 0-1 与 3-0
        r01 * r12,  # 中心1 邻边 0-1 与 1-2
        r23 * r12,  # 中心2 邻边 2-3 与 1-2
        r23 * r30,  # 中心3 邻边 2-3 与 3-0
    ]
    total = sum(raw) or 1.0
    return [w / total for w in raw]

def build_position_probability_field(
    geometry: Dict[str, Any],
    grid: int = POSITION_FIELD_GRID,
) -> List[List[float]]:
    """
    四个角落到几何中心的中点各放一个更平坦的各向同性高斯。
    用对边长度比给四个中心加权，叠加为混合密度；四边形外为 0，不做 min-max。
    """
    center = geometry["center"]
    corners = geometry["corners"]
    gcs = geometry["gaussian_centers"]
    weights = gaussian_center_weights(corners)
    geometry["gaussian_weights"] = weights
    geometry["edge_lengths"] = [
        _edge_length(corners[i], corners[(i + 1) % 4])
        for i in range(4)
    ]

    sigmas: List[float] = []
    for gx, gy in gcs:
        dist = math.hypot(gx - center[0], gy - center[1])
        sigmas.append(max(dist * 0.55, 0.05))
    geometry["gaussian_sigmas"] = sigmas

    field: List[List[float]] = []
    for j in range(grid):
        y = (j + 0.5) / grid
        row: List[float] = []
        for i in range(grid):
            x = (i + 0.5) / grid
            if not _point_in_polygon(x, y, corners):
                row.append(0.0)
                continue
            density = 0.0
            for (gx, gy), sigma, weight in zip(gcs, sigmas, weights):
                dx = x - gx
                dy = y - gy
                gauss = math.exp(-0.5 * (dx * dx + dy * dy) / (sigma * sigma))
                density += weight * gauss
            row.append(density)
        field.append(row)
    return field

def sample_from_probability_field(field: List[List[float]]) -> Tuple[Tuple[float, float], float]:
    grid = len(field)
    weights: List[float] = []
    coords: List[Tuple[float, float]] = []
    for j, row in enumerate(field):
        for i, weight in enumerate(row):
            if weight > 0:
                weights.append(weight)
                coords.append(((i + 0.5) / grid, (j + 0.5) / grid))
    if not weights:
        return (0.5, 0.5), 0.0
    total = sum(weights)
    idx = random.choices(range(len(weights)), weights=weights, k=1)[0]
    return coords[idx], weights[idx] / total

def _mixture_density(
    x: float,
    y: float,
    gcs: List[Tuple[float, float]],
    sigmas: List[float],
    weights: List[float],
) -> float:
    density = 0.0
    for (gx, gy), sigma, weight in zip(gcs, sigmas, weights):
        dx = x - gx
        dy = y - gy
        density += weight * math.exp(-0.5 * (dx * dx + dy * dy) / (sigma * sigma))
    return density

def sample_from_weighted_gaussians(
    geometry: Dict[str, Any],
    max_tries: int = 120,
) -> Tuple[Tuple[float, float], float]:
    """
    先按四个中心的权重抽一个高斯，再从该高斯抽样。
    避免把四张密度图直接相加——相加会在几何中心叠出最高峰。
    同时丢掉太靠近几何中心的点。
    """
    gcs = geometry["gaussian_centers"]
    weights = geometry["gaussian_weights"]
    sigmas = geometry["gaussian_sigmas"]
    corners = geometry["corners"]
    center = geometry["center"]
    mean_r = sum(math.hypot(gx - center[0], gy - center[1]) for gx, gy in gcs) / max(len(gcs), 1)
    min_r = mean_r * 0.45

    for _ in range(max_tries):
        idx = random.choices(range(len(gcs)), weights=weights, k=1)[0]
        gx, gy = gcs[idx]
        sigma = sigmas[idx]
        x = _clamp01(random.gauss(gx, sigma))
        y = _clamp01(random.gauss(gy, sigma))
        if not _point_in_polygon(x, y, corners):
            continue
        if math.hypot(x - center[0], y - center[1]) < min_r:
            continue
        if math.hypot(x - 0.5, y - 0.5) < 0.12:
            continue
        return (x, y), _mixture_density(x, y, gcs, sigmas, weights)

    farthest = max(gcs, key=lambda p: math.hypot(p[0] - 0.5, p[1] - 0.5))
    return farthest, _mixture_density(farthest[0], farthest[1], gcs, sigmas, weights)

def sample_foreign_object_position(geometry: Dict[str, Any]) -> Dict[str, Any]:
    """由废钢几何构造加权高斯并采样一个异物出现位置。"""
    build_position_probability_field(geometry)
    xy, density = sample_from_weighted_gaussians(geometry)
    description = describe_sampled_position(xy, geometry["center"])
    return {
        "center": geometry["center"],
        "corners": geometry["corners"],
        "vlm_center": geometry.get("vlm_center"),
        "diagonal_midpoints": geometry.get("diagonal_midpoints"),
        "gaussian_centers": geometry["gaussian_centers"],
        "gaussian_sigmas": geometry.get("gaussian_sigmas"),
        "gaussian_weights": geometry.get("gaussian_weights"),
        "edge_lengths": geometry.get("edge_lengths"),
        "sampled_xy": [round(xy[0], 4), round(xy[1], 4)],
        "sampled_probability": round(float(density), 6),
        "description": description,
    }

CYLINDER_CLASSES = {
    "灭火器", "气瓶", "杀虫剂罐", "自喷漆罐", "密封胶罐", "集油器",
    "气缸", "油缸", "液压缸", "减震器", "千斤顶", "分离过滤器", "储罐",
}

SCORE_AXES = ("tone_score", "size_score", "overall_score")

PASS_SCORE = 85.0

def is_cylinder_class(object_class: Optional[str]) -> bool:
    if not object_class:
        return False
    name = OBJECT_SCALE_ALIASES.get(object_class, object_class)
    return name in CYLINDER_CLASSES or object_class in CYLINDER_CLASSES

def _score100(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if 0 < number <= 1:
        number *= 100.0
    return max(0.0, min(100.0, number))

def score_triplet(evaluation: Dict[str, Any]) -> Dict[str, float]:
    return {key: _score100(evaluation.get(key)) for key in SCORE_AXES}

def average_score(evaluation: Dict[str, Any]) -> float:
    scores = score_triplet(evaluation)
    return sum(scores.values()) / len(SCORE_AXES)

def scores_pass(evaluation: Dict[str, Any]) -> bool:
    return all(score > PASS_SCORE for score in score_triplet(evaluation).values())

def sample_object_pose(object_class: Optional[str] = None) -> Dict[str, Any]:
    """圆柱偏向斜插或躺在坡底；非圆柱水平插入多于斜插，竖直最少。斜插相对水平线 0°–30°。"""
    cylinder = is_cylinder_class(object_class)
    facing_right = random.random() < 0.5
    facing = "right" if facing_right else "left"
    facing_cn = "朝右" if facing_right else "朝左"
    sign = 1.0 if facing_right else -1.0
    if cylinder:
        mode = random.choices(
            ["diagonal_insert", "under_slope", "on_crest"],
            weights=[0.45, 0.40, 0.15],
        )[0]
        shape_cn = "圆柱"
    else:
        mode = random.choices(
            ["horizontal_insert", "diagonal_insert", "vertical_insert"],
            weights=[0.50, 0.35, 0.15],
        )[0]
        shape_cn = "非圆柱"
    if mode == "diagonal_insert":
        tilt = random.uniform(0.0, 30.0)
        pose_cn = "斜插入废钢内部"
        placement = (
            f"长轴相对水平线倾斜{tilt:.1f}度，必须落在0度到30度之间，{facing_cn}斜插入废钢堆内部。"
            "不要平放在堆顶表面，不要竖直立着。"
        )
    elif mode == "vertical_insert":
        tilt = random.uniform(82.0, 90.0)
        pose_cn = "静止竖直插入"
        placement = (
            f"长轴相对水平线约{tilt:.1f}度，接近垂直，静止竖直插入废钢堆内部，不要平躺在表面。"
        )
    elif mode == "under_slope":
        tilt = random.uniform(0.0, 8.0)
        pose_cn = "平躺在废钢坡低处"
        placement = (
            f"长轴相对水平线仅倾斜{tilt:.1f}度，平躺在废钢坡的低处，像从高处滚落后停在坡底。"
            "不要放在堆顶高出的表面上，不要悬浮。"
        )
    elif mode == "on_crest":
        tilt = random.uniform(0.0, 8.0)
        pose_cn = "平放在堆顶"
        placement = (
            f"长轴相对水平线仅倾斜{tilt:.1f}度，平放在废钢堆高出的表面。不要悬浮。"
        )
    else:
        tilt = random.uniform(0.0, 8.0)
        pose_cn = "水平插入废钢内部"
        placement = (
            f"长轴相对水平线仅倾斜{tilt:.1f}度，水平插入废钢堆内部，不要立在表面，不要悬浮。"
        )
    signed = sign * tilt
    description = f"{shape_cn}{pose_cn}，整体{facing_cn}。{placement}"
    return {
        "cylinder": cylinder,
        "pose": mode,
        "pose_cn": pose_cn,
        "facing": facing,
        "facing_cn": facing_cn,
        "tilt_from_horizontal_deg": round(signed, 2),
        "base_angle_deg": 0.0 if mode != "vertical_insert" else 90.0,
        "small_angle_deg": round(signed, 2),
        "final_angle_deg": round(signed, 2),
        "description": description,
    }

def sample_horizontal_orientation() -> Dict[str, Any]:
    """兼容旧调用：按非圆柱抽样。"""
    return sample_object_pose(None)

# ============ 物体图片加载器 ============

def six_view_object_items(six_views_dir: Path, catalog_path: Path) -> List[Dict[str, Any]]:
    """关闭逐件目录时，输入仍是同一套六视图，每张图只代表一件。"""
    data = json.loads(catalog_path.read_text(encoding="utf-8"))
    class_alias = {"阀门": "球阀"}
    items: List[Dict[str, Any]] = []
    missing = []
    for profile in data.get("objects", []):
        path = six_views_dir / profile["reference_filename"]
        if not path.is_file():
            missing.append(profile["reference_filename"])
            continue
        category = str(profile["category"])
        items.append({
            "path": path,
            "class": class_alias.get(category, category),
            "weight": 1.0,
            "filename": path.name,
            "profile_id": profile["id"],
        })
    if missing:
        print("[WARNING] 六视图缺失: " + ", ".join(missing))
    print(f"[DEBUG] 六视图输入: {len(items)} 张，目录 {six_views_dir}")
    return items

def get_all_object_images_with_classes(object_root: Path, prompt_templates: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    object_items = []
    
    if not object_root.exists():
        return object_items
    
    all_images = []
    for img in object_root.rglob("*"):
        if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
            all_images.append(img)
    
    print(f"[DEBUG] 共找到 {len(all_images)} 张物体图片")
    
    for img_path in all_images:
        filename = img_path.stem.lower()
        
        matched_class = None
        for cls in prompt_templates.keys():
            if cls in filename:
                matched_class = cls
                break
        
        if not matched_class:
            for cls in prompt_templates.keys():
                cls_variants = [
                    cls,
                    cls.replace("_", ""),
                    cls.replace("_", "-"),
                ]
                for variant in cls_variants:
                    if variant in filename:
                        matched_class = cls
                        break
                if matched_class:
                    break
        
        if not matched_class:
            print(f"[WARNING] 无法识别类别: {img_path.name}")
            continue
        
        # ===== 提高"罐/瓶/缸"类别的权重 =====
        # 键名必须与 prompt_templates.json 中的类别名称完全一致（中文）
        boosted_classes = {
            '密封胶罐': 4.0,       # 密封胶罐：1张，低权重 → 最高优先级
            '储罐': 3.0,           # 储罐：3张
            '杀虫剂罐': 3.0,       # 杀虫剂罐：3张
            '气瓶': 3.0,           # 气瓶：3张
            '气缸': 3.0,           # 气缸：3张
            '油缸': 3.0,           # 油缸：2张
            '灭火器': 3.0,         # 灭火器：3张
            '自喷漆罐': 4.0,       # 自喷漆罐：1张（最高优先级）
        }

        if matched_class in boosted_classes:
            weight = boosted_classes[matched_class]
            print(f"[DEBUG] ⭐ {img_path.name} 类别 '{matched_class}' 权重提高到 {weight}")
        elif filename.startswith('a'):
            weight = 0.1
            print(f"[DEBUG] {img_path.name} 以 'a' 开头，权重降低到 0.1")
        else:
            weight = 1.0
        
        object_items.append({
            "path": img_path,
            "class": matched_class,
            "weight": weight,
            "filename": img_path.name
        })
    
    print(f"[DEBUG] 有效物体图片: {len(object_items)} 张")
    
    class_counts = {}
    for item in object_items:
        cls = item["class"]
        class_counts[cls] = class_counts.get(cls, 0) + 1
    print(f"[DEBUG] 类别分布: {class_counts}")
    
    weight_counts = {"低权重(以a开头)": 0, "正常权重": 0}
    for item in object_items:
        if item["weight"] < 1.0:
            weight_counts["低权重(以a开头)"] += 1
        else:
            weight_counts["正常权重"] += 1
    print(f"[DEBUG] 权重分布: {weight_counts}")
    
    return object_items

def choose_weighted_object(object_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not object_items:
        return None
    
    weights = [item["weight"] for item in object_items]
    total_weight = sum(weights)
    
    if total_weight == len(object_items):
        return random.choice(object_items)
    
    r = random.random() * total_weight
    cumsum = 0
    for item, weight in zip(object_items, weights):
        cumsum += weight
        if r <= cumsum:
            return item
    
    return object_items[-1]

def collect_object_views(
    object_path: Path,
    object_class: str,
    object_items: List[Dict[str, Any]],
    max_views: int,
    six_views_dir: Optional[Path] = None,
) -> List[Path]:
    """同类实拍和已有六视图格子，只作外形参考，不作为贴图素材。"""
    # Each atlas describes one specific item; never append other items from its broad class.
    matched_item = next((item for item in object_items if item.get("profile_id") and item["path"] == object_path), None)
    if matched_item:
        return [object_path] if max_views > 0 else []
    views: List[Path] = []
    seen = set()

    def add(path: Optional[Path]) -> None:
        if path is None or not path.is_file():
            return
        key = path.resolve()
        if key in seen:
            return
        seen.add(key)
        views.append(path)

    if six_views_dir and six_views_dir.exists():
        needles = [object_path.stem.lower(), object_class.lower()]
        for img in sorted(six_views_dir.rglob("*")):
            if not (img.is_file() and img.suffix.lower() in IMAGE_EXTS):
                continue
            blob = f"{img.parent.name} {img.stem}".lower()
            if any(n and n in blob for n in needles):
                add(img)

    add(object_path)
    for item in object_items:
        if item.get("class") == object_class:
            add(item.get("path"))

    if max_views <= 0:
        return []
    return views[:max_views]

# ============ VLM分析器类 ============

class VLMAnalyzer:
    def __init__(self, cfg: AppConfig, prompt_analyzer: Optional[PromptAnalyzer] = None):
        self.cfg = cfg
        self.optimization_history: List[Dict] = []
        self.prompt_enhancer = PromptEnhancer()
        self.prompt_analyzer = prompt_analyzer
        self.vlm_messages: List[Dict[str, Any]] = []

    def reset_vlm_session(self) -> None:
        self.vlm_messages = []

    def _post_vlm(self, messages: List[Dict[str, Any]], max_tokens: int) -> str:
        api_key = self.cfg.vlm_api_key
        print(f"[DEBUG] VLM 凭据已配置: {bool(api_key)}")
        print(f"[DEBUG] VLM 请求 URL: {self.cfg.vlm_api_url}")
        print(f"[DEBUG] VLM 轮次消息数: {len(messages)}")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.cfg.vlm_model,
            "messages": messages,
            "max_tokens": min(max_tokens, 3072),
            "temperature": 0.3,
        }
        if self.cfg.vlm_model.startswith("doubao-seed"):
            payload["thinking"] = {"type": self.cfg.vlm_thinking_mode}
        session = get_session_with_ssl_bypass()
        response = session.post(
            self.cfg.vlm_api_url,
            headers=headers,
            json=payload,
            timeout=self.cfg.timeout_seconds,
        )
        response.raise_for_status()
        result = response.json()
        content_text = result["choices"][0]["message"]["content"]
        if isinstance(content_text, list):
            content_text = "".join(
                (part.get("text") or "") if isinstance(part, dict) else str(part)
                for part in content_text
            )
        return content_text
        
    def _call_vlm(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        images: List[Path],
        max_tokens: int = 10000,
        expect_json: bool = True,
        continue_session: bool = False,
    ) -> Any:
        if continue_session and self.vlm_messages:
            text = user_prompt
        else:
            text = f"{system_prompt}\n\n{user_prompt}"
        
        content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
        for img_path in images:
            data_url = image_to_data_url_vlm(img_path)
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        
        user_msg = {"role": "user", "content": content}
        if continue_session:
            self.vlm_messages.append(user_msg)
            to_send = self.vlm_messages
        else:
            to_send = [user_msg]
        
        try:
            content_text = self._post_vlm(to_send, max_tokens)
            if continue_session:
                self.vlm_messages.append({"role": "assistant", "content": content_text})
            if expect_json:
                return self._extract_json(content_text)
            return content_text
        except requests.exceptions.SSLError as e:
            print(f"  [SSL Error] {e}，尝试使用更宽松的SSL配置...")
            try:
                session = requests.Session()
                session.verify = False
                payload = {
                    "model": self.cfg.vlm_model,
                    "messages": to_send,
                    "max_tokens": max_tokens,
                    "temperature": 0.3,
                }
                headers = {
                    "Authorization": f"Bearer {self.cfg.vlm_api_key}",
                    "Content-Type": "application/json",
                }
                response = session.post(
                    self.cfg.vlm_api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.cfg.timeout_seconds,
                )
                response.raise_for_status()
                content_text = response.json()["choices"][0]["message"]["content"]
                if continue_session:
                    self.vlm_messages.append({"role": "assistant", "content": content_text})
                if expect_json:
                    return self._extract_json(content_text)
                return content_text
            except Exception as e2:
                print(f"  [重试失败] {e2}")
                if continue_session and self.vlm_messages and self.vlm_messages[-1] is user_msg:
                    self.vlm_messages.pop()
                return self._get_default_evaluation()
        except Exception as e:
            print(f"  [VLM API Error] {e}")
            if continue_session and self.vlm_messages and self.vlm_messages[-1] is user_msg:
                self.vlm_messages.pop()
            return self._get_default_evaluation()
    
    def _call_vlm_with_data_url(
        self,
        system_prompt: str,
        user_prompt: str,
        image_data_url: str,
        max_tokens: int = 4096,
        expect_json: bool = True,
        continue_session: bool = False,
    ) -> Any:
        if continue_session and self.vlm_messages:
            text = user_prompt
        else:
            text = f"{system_prompt}\n\n{user_prompt}"
        content = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]
        user_msg = {"role": "user", "content": content}
        if continue_session:
            self.vlm_messages.append(user_msg)
            to_send = self.vlm_messages
        else:
            to_send = [user_msg]
        try:
            content_text = self._post_vlm(to_send, max_tokens)
            if continue_session:
                self.vlm_messages.append({"role": "assistant", "content": content_text})
            if expect_json:
                return self._extract_json(content_text)
            return content_text
        except Exception as e:
            print(f"  [VLM API Error] {e}")
            if continue_session and self.vlm_messages and self.vlm_messages[-1] is user_msg:
                self.vlm_messages.pop()
            return self._get_default_evaluation()
    
    def _get_default_analysis(self) -> Dict[str, Any]:
        return {
            "object": {
                "category": "unknown",
                "color": "金属色",
                "shape": "不规则",
                "material": "金属",
                "texture": "粗糙",
                "key_features": ""
            },
            "background": {
                "lighting": "自然光",
                "shadows": "废钢阴影",
                "stacking": "杂乱堆叠",
                "gaps": "废钢缝隙",
                "tone": "暗沉",
                "texture_style": "粗糙"
            },
            "synthesis_advice": "将异物塞入废钢缝隙中，被废钢遮挡包裹"
        }
    
    def _get_default_evaluation(self) -> Dict[str, Any]:
        """返回默认评估结果（当 VLM 调用失败时）"""
        return {
            "size_judgment": "acceptable",
            "visible_ratio": None,
            "tone_score": 0,
            "size_score": 0,
            "overall_score": 0,
            "tone_fix": "",
            "size_fix": "",
            "overall_fix": "",
            "sharpness": "ok",
            "confidence": 0.5,
            "reason": "VLM 调用失败，使用默认评估",
            "improvement_suggestion": "",
            "should_retry": True,
            "quality_score": 0.5,
            "fusion_quality": "ok"
        }
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end].strip()
        return json.loads(text)
    
    def open_background_dialogue(
        self,
        background_path: Path,
        base_prompt: str,
        object_class: str,
    ) -> Dict[str, Any]:
        """同一段对话的第一轮：背景图 + 初始 prompt，回复里同时给出几何定位。"""
        user_prompt = (
            f"目标异物类别：{object_class}\n"
            "下面这张图是废钢车斗背景图。随后是准备交给生图模型的初始prompt，此时还不含放置位置。\n"
            f"初始prompt：\n{base_prompt}\n\n"
            "请在同一条JSON里完成两件事：\n"
            "1. reply：说明如何把该异物放进这张背景。\n"
            "2. 几何定位：忽略车斗金属壁、挡板、驾驶室、天空、地面、栏杆，只看车斗内部废钢堆。"
            "给出废钢堆几何中心，以及从该中心展开、落在废钢外边缘（不是墙壁）的四个角落。"
            "四个点尽量构成以几何中心为对角线中点的四边形，并尽量盖住整堆废钢。"
            "坐标为归一化比例，x=0左缘，x=1右缘，y=0上缘，y=1下缘。不要使用像素值。\n"
            "3. scene_context：观察本张真实背景的lighting光源/阴影/高光、contrast明暗和反射、"
            "capture_style监控锐度/噪声、surface_condition邻近钢件旧化程度、support_and_occlusion可承重钢件与可遮挡边缘。"
            "再把外观拆成五项，禁止混进同一句："
            "color_temperature 只写白平衡和冷暖，不写色相；"
            "tone 只写主色相、辅色相和明度，不写饱和度，不写冷暖；"
            "saturation 必须写成发灰、发浊、蒙尘、褪色，并且明显低于周围废钢；"
            "pixel 只写颗粒、压缩块和局部对比，不写清不清楚；"
            "sharpness 先写有没有拍摄晃动的拖影，再写失焦和边缘软硬，不写噪点。"
            "不确定光源方向时明确写不确定，不伪造物理测量。\n"
            "4. placement_xy：选一个能落在钢板/型钢支点上且能被现有前景钢件遮住部分轮廓的堆内位置，"
            "避开墙壁、时间戳、机位文字和完整展示式堆顶摆放，给归一化[x,y]。\n"
            "输出JSON："
            "{\"reply\":\"说明\","
            "\"center\":{\"x\":0.0,\"y\":0.0},"
            "\"corners\":["
            "{\"x\":0.0,\"y\":0.0},{\"x\":0.0,\"y\":0.0},"
            "{\"x\":0.0,\"y\":0.0},{\"x\":0.0,\"y\":0.0}],"
            "\"placement_xy\":[0.5,0.5],\"scene_context\":{\"lighting\":\"观察\","
            "\"contrast\":\"观察\",\"capture_style\":\"观察\",\"surface_condition\":\"观察\",\"support_and_occlusion\":\"观察\","
            "\"color_temperature\":\"色温\",\"tone\":\"色调\",\"saturation\":\"饱和度\",\"pixel\":\"像素\",\"sharpness\":\"清晰度\"}}"
        )
        return self._call_vlm(
            system_prompt=(
                "你是废钢安全质检专家。只输出有效JSON。"
                "几何坐标必须在0到1之间，四个corners必须落在废钢堆边缘而不是墙壁上。"
            ),
            user_prompt=user_prompt,
            images=[background_path],
            expect_json=True,
            continue_session=True,
        )
    
    def compose_generation_prompt(
        self,
        background_path: Path,
        object_class: str,
        spec: Dict[str, Any],
        size_history: str,
        position_hint: str,
        orientation_hint: str,
        scene_hint: str,
        score_notes: str = "",
    ) -> str:
        """VLM 看全程设定/返回尺寸后写提示词。返回值只含本轮设定尺寸。"""
        user_prompt = (
            f"类别：{object_class}\n"
            f"{spec['size_brief']}\n"
            f"{size_history}\n"
            f"位置：{position_hint}\n"
            f"朝向：{orientation_hint}\n"
            f"{score_notes}\n"
            f"场景参考，不要照抄里面的旧尺寸数字：{scene_hint}\n\n"
            + scene_context_prompt(getattr(self.cfg, "active_scene_context", None)) + "\n"
            "请写一条交给生图模型的中文 generation_prompt。\n"
            "生图模型只能看到这条提示词，看不到尺寸历程。\n"
            "生图模型除背景画布外还会看到该件六视图，参考图只认几何和零件，不继承新品光泽、鲜艳漆面、摄影棚光照或干净标签。"
            "必须写成明确报废旧件，按原场景重建材质，并交代实际支点、贴合接触阴影、前景钢件穿过部分轮廓的遮挡关系。\n"
            "背景外观必须分成五句，禁止混成一句颜色或一句清晰度："
            "色温只写白平衡和冷暖；色调只写色相和明度，不要写饱和度；"
            "饱和度必须发灰、发浊、蒙尘、褪色，并且明显低于周围废钢，禁止鲜红、橘红、亮黄、翠绿、纯蓝、荧光色；"
            "像素只写颗粒、压缩块和局部对比；清晰度只写失焦和拖影，异物不能比废钢更锐，不要柔化整张监控图。\n"
            "这些禁令里的颜色词保持原样，不要改写成别的说法。\n"
            "结构尺寸必须使用【本轮设定尺寸】里的真实厘米和直径/高度，不要把筒径、握把、杆径缩成几厘米的模型。\n"
            "本轮可见最长边的厘米和百分比也不要改。\n"
            "不要写上一轮、测得、历程、返回尺寸、得分，也不要写更早轮次的百分比。\n"
            "若给出了色调、大小、整体三项修正，把它们写成这一轮的画面要求，不要写分数。\n"
            "只输出JSON：{\"generation_prompt\":\"...\"}"
        )
        raw = self._call_vlm(
            system_prompt=(
                "你是废钢异物生图提示词编辑。只输出有效JSON。"
                "你可以阅读尺寸历程来决定措辞，但 generation_prompt 里只能出现本轮设定尺寸。"
                "必须交代多视图只是外形参考，禁止把参考图里的物体直接插入场景。"
            ),
            user_prompt=user_prompt,
            images=[] if self.vlm_messages else [background_path],
            expect_json=True,
            continue_session=True,
        )
        text = ""
        if isinstance(raw, dict):
            text = str(raw.get("generation_prompt") or "")
        if not text.strip():
            text = spec["size_brief"]
        cleaned = keep_only_current_size(text, spec)
        print(f"  [VLM] 本轮生图尺寸: {spec['size_brief']}")
        return cleaned

    def evaluate_generated_image(
        self,
        generated_image_path: Path,
        object_class: str,
        original_prompt: str,
        target_ratio: Optional[float] = None,
        size_spec: Optional[Dict[str, Any]] = None,
        size_rounds: Optional[List[Dict[str, Any]]] = None,
        background_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        VLM 评估生成的图片。长边约 1280，便于看 3% 级可见段和整图锐度。
        """
        import io

        try:
            img = Image.open(generated_image_path)
            img.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.convert("RGB").save(buffer, format="JPEG", quality=85)
            buffer.seek(0)
            b64_data = base64.b64encode(buffer.read()).decode("utf-8")
            compressed_data_url = f"data:image/jpeg;base64,{b64_data}"
            print(f"  [VLM] 评估图已缩到长边≤1280 JPEG85")
        except Exception as e:
            print(f"  [VLM] 图片压缩失败: {e}，使用原始图片")
            compressed_data_url = image_to_data_url_vlm(generated_image_path)

        spec = size_spec or round_size_spec(self.cfg, object_class, target_ratio)
        history_text = format_size_history(size_rounds or [])
        user_prompt = (
            f"请评估这张合成图里的{object_class}。\n"
            f"{spec['size_brief']}\n"
            f"允许范围：{spec['scale_min']:.1%}–{spec['scale_max']:.1%}。\n"
            f"{history_text}\n"
            "尺寸以真实厘米为准。车斗水平视野约240厘米，visible_cm = visible_ratio × 240。\n"
            "筒径、握把、喷管、活塞杆、法兰等必须对照上面的真实厘米，不要对照被缩小后的模型尺寸。\n"
            "规则：\n"
            "1. 先给出 visible_ratio（可见最长边/画面宽度，0到1的小数）。\n"
            "2. 可见最长边的厘米明显大于真实参照长度，或整件比真实件更大 → too_large。\n"
            "3. 没放进去，或露出部分明显小于真实横截面、像玩具 → too_small。\n"
            "4. 露出部分落在真实横截面到真实参照长度之间，比例符合直径/高度 → acceptable。\n"
            "5. 埋入导致看不见全长是正常的，不要因此判太小；能看见也不要因此判太大。\n"
            "6. 小件如果达到更大类别的真实尺寸，例如喷雾罐画得像灭火器 → too_large。\n"
            "7. 标签文字可建议模糊标签；禁止建议柔化整张图或降低分辨率。\n"
            "8. sharpness 评估整张图是否发糊；发糊不要改 size_judgment。\n"
            "12. color_temperature_match 只看白平衡和冷暖。\n"
            "13. tone_match 只看色相和明度，不看饱和度；tone_fix 也不写饱和度。\n"
            "14. saturation_match：发灰、褪色且明显低于废钢为 good；鲜红、橘红、亮黄、翠绿、纯蓝、荧光，或和废钢一样鲜，为 bad。\n"
            "15. pixel_match 只看颗粒、压缩块和局部对比。\n"
            "16. sharpness_match 只看失焦、拖影和边缘软硬；异物比废钢更锐为 bad。\n"
            "9. quality_score 只打融合和清晰，不要因尺寸不合格就打到 0.3。\n"
            "10. tone_score、size_score、overall_score 为 0 到 100 的整数。三项都高于 85 才算合格。\n"
            "11. tone_fix、size_fix、overall_fix 各用一句话说明下一轮该怎么改色调、大小和整体埋入效果。\n"
            "只输出JSON："
            "{\"visible_ratio\": 0.0, \"size_judgment\": \"too_large|acceptable|too_small\", "
            "\"tone_score\": 0, \"size_score\": 0, \"overall_score\": 0, "
            "\"tone_fix\": \"\", \"size_fix\": \"\", \"overall_fix\": \"\", "
            "\"color_temperature_match\": \"bad|ok|good\", \"tone_match\": \"bad|ok|good\", "
            "\"saturation_match\": \"bad|ok|good\", \"pixel_match\": \"bad|ok|good\", "
            "\"sharpness_match\": \"bad|ok|good\", "
            "\"sharpness\": \"sharp|ok|soft\", \"confidence\": 0.0, \"reason\": \"\", "
            "\"improvement_suggestion\": \"\", \"quality_score\": 0.0, "
            "\"fusion_quality\": \"very bad|bad|ok|good|very good\"}"
        )

        profile = getattr(self.cfg, "active_object_profile", None)
        if profile and profile["category"] == object_class:
            user_prompt = per_item_evaluation_prompt(profile, spec, history_text)
        eval_user_prompt = (
            "下面这张图是生图模型本轮的输出。请只根据这张生成图做评估，只输出JSON。\n\n"
            + user_prompt
        )
        if profile and background_path:
            eval_user_prompt = (
                "第1张为原监控背景，第2张为当前生成结果。必须对照两张图检查场景适配与背景保持，"
                "所有位置和占比只以第2张完整画幅为准。不要根据提示词声称要求已实现，也不要猜测合成目标占比。\n"
                + f"当前物品：{appearance_neutral_name(profile)}。\n几何身份线索：" + "；".join(geometry_identity(profile))
                + "\n材质旧化要求：" + profile["weathering_prompt"]
                + "\n请独立定位物品最大的连续可见片段，返回visible_bbox_xyxy=[x1,y1,x2,y2]，均为0–1归一化坐标。"
                "不能把被遮住的部位补进框内，不能把分开的两个露出片段之间的隐藏部分计入；没有目标返回null。"
                "visible_ratio依据图像观察给出，尺寸范围由程序后续比较，不向你提供目标数值。"
                "给tone_score、overall_score、sharpness、confidence、reason、improvement_suggestion、quality_score、fusion_quality，"
                "以及具体修正建议；不要求清晰读取旧标签。\n"
                + review_requirements()
                + "\n" + scene_context_prompt(getattr(self.cfg, "active_scene_context", None))
            )
            response = self._call_vlm(
                system_prompt=self.cfg.vlm_evaluator_prompt,
                user_prompt=eval_user_prompt,
                images=[background_path, generated_image_path],
                expect_json=True,
                continue_session=False,
            )
        else:
            response = self._call_vlm_with_data_url(
                system_prompt=self.cfg.vlm_evaluator_prompt,
                user_prompt=eval_user_prompt,
                image_data_url=compressed_data_url,
                expect_json=True,
                continue_session=True,
            )
        if not isinstance(response, dict):
            response = self._get_default_evaluation()
            response["reason"] = "VLM 返回非JSON，使用默认评估"
        response["improvement_suggestion"] = sanitize_vlm_suggestion(
            response.get("improvement_suggestion", "")
        )
        for axis in SCORE_AXES:
            response[axis] = _score100(response.get(axis))
        for axis, value in fusion_quality_gate(response)["scores"].items():
            response[axis] = value
        for fix_key in ("tone_fix", "size_fix", "overall_fix", "weathering_fix", "lighting_fix", "occlusion_fix", "style_fix", "background_fix", "identity_fix"):
            response[fix_key] = sanitize_vlm_suggestion(response.get(fix_key, ""))
        try:
            visible = response.get("visible_ratio")
            response["visible_ratio"] = float(visible) if visible is not None else None
        except (TypeError, ValueError):
            response["visible_ratio"] = None
        sharpness_label = str(response.get("sharpness") or "ok").lower()
        if sharpness_label not in {"sharp", "ok", "soft"}:
            sharpness_label = "ok"
        response["sharpness"] = sharpness_label
        if profile and background_path:
            bbox = response.get("visible_bbox_xyxy")
            observed = None
            if isinstance(bbox, list) and len(bbox) == 4:
                try:
                    x1, y1, x2, y2 = map(float, bbox)
                    if 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1:
                        with Image.open(generated_image_path) as source:
                            width, height = source.size
                        observed = math.hypot(x2-x1, (y2-y1)*height/width)
                except (TypeError, ValueError):
                    pass
            response["visible_ratio"] = observed
            response["ratio_evidence"] = "diagonal_of_vlm_inferred_visible_component_bbox_not_segmentation_or_physical_measurement"
            low, high = spec["scale_min"], spec["scale_max"]
            if observed is None:
                response["size_judgment"] = "unknown"
                response["size_score"] = 0.0
            elif low <= observed <= high:
                response["size_judgment"] = "acceptable"
                response["size_score"] = 90.0
            else:
                response["size_judgment"] = "too_small" if observed < low else "too_large"
                response["size_score"] = 60.0
                response["size_fix"] = "保持实体尺度与零件比例，通过前景宽钢件遮挡调整最大连续可见片段，不能把整件缩成模型。"
        return response
    
    def optimize_prompt(
        self,
        previous_prompt: str,
        evaluation_result: Dict,
        object_class: str,
        round_num: int,
        target_ratio: Optional[float] = None,
    ) -> Tuple[str, float]:
        judgment = evaluation_result.get("size_judgment", "acceptable")
        suggestion = sanitize_vlm_suggestion(evaluation_result.get("improvement_suggestion", ""))
        
        lines = previous_prompt.split("\n")
        core_lines = [l for l in lines if not l.startswith("调整：") and not l.startswith("【尺寸约束】")]
        core = "\n".join(core_lines)
        
        scale_min, scale_max = object_scale_range(self.cfg, object_class)
        current = target_ratio if target_ratio is not None else (scale_min + scale_max) / 2
        if judgment == "too_small":
            new_ratio = min(scale_max, max(current * 1.25, scale_min * 1.1))
            fix = (
                f"按{object_class}的真实厘米放大到能看出横截面，不要再画成玩具。"
                f"{class_size_clause(self.cfg, object_class, new_ratio)}"
            )
            occlusion = "留出能辨认真实尺度的一段，不要完全埋进废钢里"
        elif judgment == "too_large":
            new_ratio = max(scale_min, min(current, scale_max) * 0.85)
            fix = (
                f"可见最长边不要超过该件真实参照长度，结构厘米保持真实值，不要整件缩成模型。"
                f"{class_size_clause(self.cfg, object_class, new_ratio)}"
            )
        else:
            new_ratio = min(max(current, scale_min), scale_max)
            fix = class_size_clause(self.cfg, object_class, new_ratio)
        
        positions = self.prompt_enhancer.get_position_descriptions("")
        if judgment != "too_small":
            occlusion = random.choice(self.prompt_enhancer.get_occlusion_descriptions())
        
        shape_fix = (
            "【形状强制约束】必须严格保持物体原本的形状、结构、比例和轮廓特征，"
            "不允许任何形式的变形、拉伸、扭曲或改变。圆形保持圆形，圆柱形保持圆柱形，方形保持方形。"
        )
        if judgment == "too_small":
            fusion_fix = "异物必须出现在画面里，露出能辨认的一小段，不能再交回一张没有异物的背景。"
        else:
            fusion_fix = random.choice([
                "加强异物的嵌入感，确保被废钢挤压包裹，边缘有遮挡，纹理与周围废钢完全一致，光影匹配背景。",
                "让异物看起来像是长期存在于废钢堆中，与周围环境融合，同时仍能辨认出它是目标异物。",
                "让异物与废钢的锈蚀纹理和色调一致，看起来像是自然嵌入的废弃部件，并留下能辨认的一小段。",
            ])
        
        return f"{core}\n调整：{fix}。{suggestion}。{occlusion}。{shape_fix}{fusion_fix}", new_ratio
    
    def run_optimization_loop(
        self,
        object_path: Path,
        background_path: Path,
        object_class: str,
        base_prompt: str = None,
        initial_generation_func: callable = None,
        image_id: str = None,
        size_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.optimization_history = []
        self.reset_vlm_session()
        final_confidence = 0.0
        final_quality_score = 0.0
        final_fusion_quality = "unknown"
        final_size_judgment = "unknown"
        current_prompt = base_prompt or ""
        prompt_for_image = current_prompt
        last_image_path: Optional[Path] = None
        last_api_result: Optional[Dict[str, Any]] = None
        last_full_prompt = current_prompt
        success = False
        round_sharpness: List[float] = []
        size_rounds: List[Dict[str, Any]] = []

        if size_state is None:
            size_state = {"ratio": sample_object_resize_scale(self.cfg, object_class)}
        print(f"  [VLM] 异物原图不送入对话，类别={object_class}，文件={object_path.name}")
        print(f"  [VLM] 首轮：背景图与初始prompt，并做几何定位")
        opening: Dict[str, Any] = {}
        try:
            raw_opening = self.open_background_dialogue(background_path, current_prompt, object_class)
            if isinstance(raw_opening, dict):
                opening = raw_opening
        except Exception as e:
            print(f"  [VLM] 首轮对话失败: {e}")
        geometry = geometry_from_vlm_reply(opening)
        scene_context = opening.get("scene_context")
        self.cfg.active_scene_context = scene_context if isinstance(scene_context, dict) else None
        observed = appearance_from_context(self.cfg.active_scene_context)
        print(
            "  [VLM] 背景外观: "
            + ", ".join(f"{key}={observed[key] or '未观察'}" for key in observed)
        )
        self.optimization_history.append({"stage": "opening", "data": opening})

        position_sample = sample_foreign_object_position(geometry)
        placement = opening.get("placement_xy")
        if isinstance(placement, list) and len(placement) == 2:
            try:
                px, py = map(float, placement)
                if 0.04 <= px <= 0.96 and 0.10 <= py <= 0.84:
                    position_sample["sampled_xy"] = [px, py]
                    position_sample["description"] = (
                        f"使用本张背景中建议的钢件支撑区，中心约水平{px:.1%}、垂直{py:.1%}；"
                        "物品嵌在已有前景钢件之后，避免完整摆放在堆顶。"
                    )
                    position_sample["selection_method"] = "vlm_support_and_occlusion_hint"
            except (TypeError, ValueError):
                pass
        self.cfg.active_edit_region = None
        self.cfg.active_position_xy = position_sample["sampled_xy"]
        orientation_sample = sample_object_pose(object_class)
        position_hint = position_sample["description"]
        orientation_hint = orientation_sample["description"]
        print(
            f"  [位置] 采样 ({position_sample['sampled_xy'][0]:.3f},"
            f"{position_sample['sampled_xy'][1]:.3f}) "
            f"p={position_sample['sampled_probability']:.3f} | {position_hint}"
        )
        print(
            f"  [姿态] {orientation_sample['pose_cn']} "
            f"{orientation_sample['facing_cn']} "
            f"相对水平{orientation_sample['tilt_from_horizontal_deg']:+.1f}°"
        )

        self.optimization_history.append({"stage": "initial_prompt", "prompt": current_prompt})
        print(f"  [VLM] 初始Prompt: {current_prompt}")

        judgment_history = []
        round_candidates: List[Dict[str, Any]] = []
        score_notes = ""

        for round_num in range(1, self.cfg.max_optimization_rounds + 1):
            print(f"  [VLM] 第{round_num}轮生成与评估...")

            if not initial_generation_func:
                break

            spec = round_size_spec(self.cfg, object_class, size_state.get("ratio"))
            image_prompt = self.compose_generation_prompt(
                background_path,
                object_class,
                spec,
                format_size_history(size_rounds),
                position_hint,
                orientation_hint,
                current_prompt,
                score_notes,
            )
            prompt_for_image = image_prompt
            # 不把上一轮生成图送入生图：wan 会当 img2img 二次渲染，第3轮明显发糊
            generated_path, api_result, full_prompt = initial_generation_func(
                image_prompt, round_num, None, position_hint, orientation_hint,
            )
            last_image_path = generated_path
            last_api_result = api_result
            last_full_prompt = full_prompt
            sharpness = image_sharpness_score(generated_path)
            round_sharpness.append(sharpness)
            print(f"  [清晰度] 第{round_num}轮 laplacian={sharpness:.1f}")

            evaluation = self.evaluate_generated_image(
                generated_path,
                object_class,
                image_prompt,
                spec["target_ratio"],
                size_spec=spec,
                size_rounds=size_rounds,
                background_path=background_path,
            )
            
            size_judgment = evaluation.get("size_judgment", "acceptable")
            confidence = evaluation.get("confidence", 0.5)
            quality_score = evaluation.get("quality_score", 0.5)
            fusion_quality = evaluation.get("fusion_quality", "ok")
            visible_ratio = evaluation.get("visible_ratio")
            vlm_sharpness = evaluation.get("sharpness", "ok")
            scores = score_triplet(evaluation)
            mean_score = average_score(evaluation)
            appearance_bad = appearance_is_bad(evaluation)
            passed = scores_pass(evaluation) and not appearance_bad
            gate = fusion_quality_gate(evaluation) if self.cfg.active_object_profile else None
            if gate is not None:
                passed = passed and gate["passed"] and size_judgment == "acceptable"
            
            final_confidence = confidence
            final_quality_score = quality_score
            final_fusion_quality = fusion_quality
            final_size_judgment = size_judgment
            
            judgment_record = {
                "round": round_num,
                "judgment": size_judgment,
                "confidence": confidence,
                "quality_score": quality_score,
                "fusion_quality": fusion_quality,
                "sharpness": sharpness,
                "visible_ratio": visible_ratio,
                "vlm_sharpness": vlm_sharpness,
                "tone_score": scores["tone_score"],
                "size_score": scores["size_score"],
                "overall_score": scores["overall_score"],
                "color_temperature_match": evaluation.get("color_temperature_match"),
                "tone_match": evaluation.get("tone_match"),
                "saturation_match": evaluation.get("saturation_match"),
                "pixel_match": evaluation.get("pixel_match"),
                "sharpness_match": evaluation.get("sharpness_match"),
                "average_score": mean_score,
                "passed": passed,
                "fusion_gate": gate,
                "image_path": str(generated_path),
            }
            judgment_history.append(judgment_record)
            round_candidates.append({
                "round": round_num,
                "path": generated_path,
                "prompt": image_prompt,
                "full_prompt": full_prompt,
                "api_result": api_result,
                "average_score": mean_score,
                "passed": passed,
                "fusion_gate": gate,
                "evaluation": evaluation,
                "edit_region": self.cfg.active_edit_region,
            })
            
            if len(judgment_history) >= 2:
                last_two = [j["judgment"] for j in judgment_history[-2:]]
                if all(j == "too_small" for j in last_two):
                    print(f"  [VLM] ⚠️ 连续2轮太小，下一轮将要求稍微放大")
                    evaluation["improvement_suggestion"] = (
                        f"{evaluation.get('improvement_suggestion', '')} 前两轮都太小，这次稍微放大。"
                    )
                elif all(j == "too_large" for j in last_two):
                    print(f"  [VLM] ⚠️ 连续2轮太大，下一轮将要求稍微缩小")
                    evaluation["improvement_suggestion"] = (
                        f"{evaluation.get('improvement_suggestion', '')} 前两轮都太大，这次明显缩小。"
                    )
            
            visible_cm = (
                float(visible_ratio) * HOPPER_FOV_M * 100.0
                if isinstance(visible_ratio, (int, float)) else None
            )
            next_ratio = compute_next_target_ratio(spec, evaluation)
            size_record = {
                "round": round_num,
                "target_ratio": spec["target_ratio"],
                "target_cm": spec["target_cm"],
                "diameter_cm": spec.get("diameter_cm"),
                "diameter_to_height": spec.get("diameter_to_height"),
                "cross_name": spec.get("cross_name"),
                "scale_min": spec["scale_min"],
                "scale_max": spec["scale_max"],
                "size_brief": spec["size_brief"],
                "generation_prompt": image_prompt,
                "visible_ratio": visible_ratio,
                "visible_cm": visible_cm,
                "size_judgment": size_judgment,
                "tone_score": scores["tone_score"],
                "size_score": scores["size_score"],
                "overall_score": scores["overall_score"],
                "average_score": mean_score,
                "next_target_ratio": next_ratio,
            }
            size_rounds.append(size_record)
            self.optimization_history.append({
                "stage": f"evaluation_round_{round_num}",
                "evaluation": evaluation,
                "prompt_used": image_prompt,
                "size": size_record,
            })
            visible_txt = f"{visible_ratio:.1%}" if isinstance(visible_ratio, (int, float)) else "?"
            print(
                f"  [VLM] 评估结果: size={size_judgment}, visible={visible_txt}, "
                f"色调={scores['tone_score']:.0f}, 大小={scores['size_score']:.0f}, "
                f"整体={scores['overall_score']:.0f}, 平均={mean_score:.1f}, "
                f"色温={evaluation.get('color_temperature_match')}, "
                f"色相={evaluation.get('tone_match')}, "
                f"饱和度={evaluation.get('saturation_match')}, "
                f"像素={evaluation.get('pixel_match')}, "
                f"清晰={evaluation.get('sharpness_match')}"
            )
            if scores_pass(evaluation) and appearance_bad:
                print("  [VLM] 分数已过线，但色温、色调、饱和度、像素或清晰度仍不一致，继续改提示词")
            
            if passed:
                print(f"  [VLM] ✅ 尺寸与场景融合复核通过，采用第{round_num}轮；仍需人工看图")
                selected_reason = "all_above_85"
                success = True
                if self.prompt_analyzer and image_id:
                    self.prompt_analyzer.add_prompt_record(
                        prompt=prompt_for_image,
                        image_id=image_id,
                        object_class=object_class,
                        size_judgment=final_size_judgment,
                        confidence=final_confidence,
                        quality_score=final_quality_score,
                        fusion_quality=final_fusion_quality,
                        round_num=round_num,
                        full_prompt=last_full_prompt,
                        fusion_passed=passed,
                    )
                break
            
            score_notes = (
                f"色调分{scores['tone_score']:.0f}，大小分{scores['size_score']:.0f}，"
                f"整体分{scores['overall_score']:.0f}。三项都需要高于85。\n"
                f"色调修正：{evaluation.get('tone_fix', '')}\n"
                f"大小修正：{evaluation.get('size_fix', '')}\n"
                f"整体修正：{evaluation.get('overall_fix', '')}\n"
                f"旧化修正：{evaluation.get('weathering_fix', '')}\n"
                f"光照修正：{evaluation.get('lighting_fix', '')}\n"
                f"遮挡接触修正：{evaluation.get('occlusion_fix', '')}\n"
                f"风格修正：{evaluation.get('style_fix', '')}\n"
                f"背景修正：{evaluation.get('background_fix', '')}\n"
                f"身份修正：{evaluation.get('identity_fix', '')}\n"
                f"{appearance_correction_text(evaluation, appearance_from_context(self.cfg.active_scene_context))}\n"
                "写给生图模型的提示词不要出现分数，也不要写上一轮。"
                "色温、色调、饱和度、像素、清晰度分开写，不要把饱和度写进色调，也不要改写禁止鲜红、青蓝、橙色这类原词。"
            )
            size_state["ratio"] = next_ratio
            next_spec = round_size_spec(self.cfg, object_class, next_ratio)
            print(
                f"  [尺寸] 下一轮设定 {next_spec['target_ratio']:.1%}"
                f"（约{next_spec['target_cm']:.0f}厘米），只写入 VLM 历程，不写入生图提示词"
            )
            self.optimization_history.append({
                "stage": f"next_size_round_{round_num}",
                "next_target_ratio": next_ratio,
                "size_brief": next_spec["size_brief"],
            })
        
        selected_reason = "size_and_fusion_above_85" if success else "best_fusion_candidate_unaccepted"
        selected_round = round_candidates[-1]["round"] if success and round_candidates else None
        if not success and round_candidates:
            best = max(round_candidates, key=lambda item: (
                (item.get("fusion_gate") or {}).get("minimum_score", item["average_score"]),
                item["average_score"],
            ))
            selected_round = best["round"]
            last_image_path = best["path"]
            prompt_for_image = best["prompt"]
            last_full_prompt = best["full_prompt"]
            last_api_result = best["api_result"]
            self.cfg.active_edit_region = best.get("edit_region")
            selected_evaluation = best["evaluation"]
            final_confidence = selected_evaluation.get("confidence", 0.0)
            final_quality_score = selected_evaluation.get("quality_score", 0.0)
            final_fusion_quality = selected_evaluation.get("fusion_quality", "unknown")
            final_size_judgment = selected_evaluation.get("size_judgment", "unknown")
            print(
                f"  [VLM] 没有一轮三项都高于{PASS_SCORE:.0f}，"
                f"采用第{best['round']}轮，平均分{best['average_score']:.1f}"
            )
        if not success:
            print(f"  [VLM] ⚠️ 达到最大优化轮数 ({self.cfg.max_optimization_rounds})")
            if self.prompt_analyzer and image_id:
                self.prompt_analyzer.add_prompt_record(
                    prompt=prompt_for_image,
                    image_id=image_id,
                    object_class=object_class,
                    size_judgment=final_size_judgment,
                    confidence=final_confidence,
                    quality_score=final_quality_score,
                    fusion_quality=final_fusion_quality,
                    round_num=self.cfg.max_optimization_rounds,
                    full_prompt=last_full_prompt,
                    fusion_passed=False,
                )
        return {
            "final_prompt": prompt_for_image,
            "full_prompt": last_full_prompt,
            "history": self.optimization_history,
            "success": success,
            "last_image_path": last_image_path,
            "api_result": last_api_result,
            "position_sample": position_sample,
            "orientation_sample": orientation_sample,
            "opening": opening,
            "round_sharpness": round_sharpness,
            "judgment_history": judgment_history,
            "size_rounds": size_rounds,
            "selected_round": selected_round,
            "selected_reason": selected_reason,
            "fusion_gate": next((item.get("fusion_gate") for item in round_candidates if item["round"] == selected_round), None),
            "selected_evaluation": next((item.get("evaluation") for item in round_candidates if item["round"] == selected_round), None),
            "scene_context": self.cfg.active_scene_context,
            "edit_region_xyxy": self.cfg.active_edit_region,
        }
    
    def generate_initial_prompt(
        self,
        object_class: str,
        object_analysis: Dict,
        background_analysis: Dict,
        base_prompt: str = None,
        position_hint: str = None,
    ) -> str:
        if base_prompt:
            return self._refine_prompt_with_analysis(
                base_prompt, object_analysis, background_analysis, object_class,
                position_hint=position_hint,
            )
        
        obj = object_analysis or {}
        bg = background_analysis or {}
        
        color = obj.get("color", "金属色")
        shape = obj.get("shape", "不规则")
        key_features = obj.get("key_features", "")
        gaps = bg.get("gaps", "废钢缝隙")
        lighting = bg.get("lighting", "自然光")
        tone = bg.get("tone", "暗沉")
        
        enhancer = self.prompt_enhancer
        position = position_hint or random.choice(enhancer.get_position_descriptions(gaps))
        occlusion = random.choice(enhancer.get_occlusion_descriptions())
        weathering = random.choice(enhancer.get_weathering_descriptions(object_class))
        lighting_desc = random.choice(enhancer.get_lighting_descriptions(lighting))
        tone_desc = random.choice(enhancer.get_tone_descriptions(tone))
        verb = random.choice(enhancer.get_embedding_verbs())
        scenario = random.choice(enhancer.get_scenario_descriptions())
        
        return (
            f"在{scenario}中，将一个真实大小的{color}{shape}的{object_class}{verb}{position}。\n"
            f"【精确尺寸】{class_size_clause(self.cfg, object_class)}\n"
            f"【形态保持】必须严格保持{key_features}等关键形态特征，不能改变物体原本的形状。\n"
            f"【遮挡方式】{occlusion}\n"
            f"【表面质感】{weathering}\n"
            f"【光照环境】{lighting_desc}\n"
            f"【色彩融合】{tone_desc}\n"
            f"【融合要求】异物的颜色、纹理、光影必须与周围废钢完全一致，不能有突兀感。\n"
            f"【禁止项】禁止改变原始背景，禁止生成干净崭新的物体，禁止让异物悬浮或尺寸过大。"
        )
    
    def _refine_prompt_with_analysis(
        self,
        base_prompt: str,
        object_analysis: Dict,
        background_analysis: Dict,
        object_class: str = None,
        position_hint: str = None,
    ) -> str:
        bg = background_analysis or {}
        obj = object_analysis or {}
        
        enhancer = self.prompt_enhancer
        
        gaps = bg.get("gaps", "废钢缝隙")
        lighting = bg.get("lighting", "自然光")
        tone = bg.get("tone", "暗沉")
        color = obj.get("color", "金属色")
        shape = obj.get("shape", "不规则")
        key_features = obj.get("key_features", "")
        
        core = base_prompt.replace("保持背景不变，", "").replace("不要改变背景，", "")
        
        position = position_hint or random.choice(enhancer.get_position_descriptions(gaps))
        occlusion = random.choice(enhancer.get_occlusion_descriptions())
        weathering = random.choice(enhancer.get_weathering_descriptions(object_class or ""))
        lighting_desc = random.choice(enhancer.get_lighting_descriptions(lighting))
        tone_desc = random.choice(enhancer.get_tone_descriptions(tone))
        verb = random.choice(enhancer.get_embedding_verbs())
        
        cls_name = object_class if object_class else "异物"
        
        return (
            f"{core}\n"
            f"将一个真实大小的{color}{shape}的{cls_name}{verb}{position}。\n"
            f"【尺寸约束】{class_size_clause(self.cfg, cls_name)}\n"
            f"【遮挡约束】{occlusion}\n"
            f"【形态约束】必须严格保持{key_features}等关键形态特征，不允许任何形式的变形。\n"
            f"【质感做旧】{weathering}\n"
            f"【光影融合】{lighting_desc}\n"
            f"【色调融合】{tone_desc}\n"
            f"【整体要求】异物必须与废钢环境完美融合，看起来像是自然存在于废钢堆中的废弃部件。"
        )

# ============ 工具函数 ============

def image_to_data_url_vlm(path: Path) -> str:
    import io
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")

def load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    return yaml.safe_load(text) or {}

def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

def choose_image(paths: List[Path]) -> Path:
    if not paths:
        raise FileNotFoundError("No images found")
    return random.choice(paths)

def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

# ============================================================
#  build_request_payload - 按图像协议组请求（自动压缩到 2K）
# ============================================================

def _encode_input_images(image_paths: List[Path]) -> Tuple[List[str], int, int]:
    """压缩到约 2K 并编码为 data URL。返回 (images, width, height)。"""
    from PIL import Image
    import io

    images: List[str] = []
    canvas_w = 0
    canvas_h = 0

    for index, path in enumerate(image_paths):
        img = Image.open(path)
        w, h = img.size
        total_pixels = w * h

        max_pixels = 4194304  # 2K
        if total_pixels > max_pixels:
            ratio = (max_pixels / total_pixels) ** 0.5
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            new_w = new_w if new_w % 2 == 0 else new_w + 1
            new_h = new_h if new_h % 2 == 0 else new_h + 1
            print(f"  [压缩] 图片从 {w}x{h} ({total_pixels}px) 压缩到 {new_w}x{new_h} ({new_w*new_h}px)")
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            w, h = new_w, new_h

        if index == 0:
            canvas_w, canvas_h = w, h

        buffer = io.BytesIO()
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        img.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)
        b64_data = base64.b64encode(buffer.read()).decode('utf-8')
        images.append(f"data:image/jpeg;base64,{b64_data}")

    return images, canvas_w, canvas_h

def build_request_payload(image_paths: List[Path], prompt: str, cfg: AppConfig) -> Dict[str, Any]:
    """
    ali_wan：input.messages，size 使用 宽*高
    huoshan_seedream：prompt/image 扁平字段，size 使用 宽x高
    """
    images, final_w, final_h = _encode_input_images(image_paths)

    if cfg.image_protocol == "ali_wan":
        content: List[Dict[str, str]] = [{"text": prompt}]
        content.extend({"image": data_url} for data_url in images)
        payload = {
            "model": cfg.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ]
            },
            "parameters": {
                "size": f"{final_w}*{final_h}",
                "n": cfg.max_images,
                "watermark": cfg.watermark,
            },
        }
        region = getattr(cfg, "active_edit_region", None)
        if getattr(cfg, "use_interactive_edit_region", True) and region and images:
            x1, y1, x2, y2 = region
            bbox = [round(x1 * final_w), round(y1 * final_h), round(x2 * final_w), round(y2 * final_h)]
            payload["parameters"]["bbox_list"] = [[bbox], *[[] for _ in images[1:]]]
        return payload

    if cfg.image_protocol == "huoshan_seedream":
        payload: Dict[str, Any] = {
            "model": cfg.model,
            "prompt": prompt,
            "size": f"{final_w}x{final_h}",
            "watermark": cfg.watermark,
            "response_format": "url",
        }
        if len(images) == 1:
            payload["image"] = images[0]
        elif images:
            payload["image"] = images
        return payload

    raise ValueError(f"未知图像协议: {cfg.image_protocol}")

def generate_single_call(
    cfg: AppConfig,
    background_path: Path,
    prompt: str,
    previous_path: Optional[Path] = None,
    reference_paths: Optional[List[Path]] = None,
) -> Tuple[Dict[str, Any], str]:
    """每轮把原始背景当画布；可选附带多视图参考。不把上一轮生成图再输入。"""
    del previous_path
    refs = [p for p in (reference_paths or []) if p and p.is_file()]
    aged_reference = getattr(cfg, "active_weathered_reference", None)
    labeled = generation_role_prompt(prompt, view_count=len(refs), references_weathered=bool(aged_reference and aged_reference in refs), reference_kind=cfg.active_reference_kind)
    image_paths = [background_path, *refs]
    print(f"  [生图] 单次调用，输入 {len(image_paths)} 张图（1张背景画布 + {len(refs)} 张多视图参考）")
    for idx, path in enumerate(refs, start=2):
        print(f"  [生图] 第{idx}张参考: {path.name}")
    payload = build_request_payload(image_paths, labeled, cfg)
    return call_image_api(payload, cfg), labeled

def prepare_scene_weathered_reference(
    cfg: AppConfig, reference_path: Path, background_path: Path
) -> Path:
    """Generate and refine an aged reference; each stage is a recorded API request."""
    profile = cfg.active_object_profile
    if not profile or not cfg.prepare_weathered_reference:
        return reference_path
    if cfg.active_weathered_reference and cfg.active_weathered_reference.is_file():
        return cfg.active_weathered_reference
    prompt = (
        "把输入图中的同一物品重新生成成明确的长期报废旧件。此次只输出一件完整的旧化物品身份参考，"
        "采用灰褐色无纹理哑光背景；不生成废钢场景、不输出六格、文字或尺寸线。\n"
        f"目标：{appearance_neutral_name(profile)}。必须保留的几何：" + "；".join(geometry_identity(profile)) + "。\n"
        "必须实际改变可见材质状态，不能继续输出原图的新品或只加少量装饰痕迹。\n"
        + profile["weathering_prompt"] + "\n"
        "对钢/铁主体，生成清楚的深褐不规则锈层、剥漆、灰尘和旧油泥；"
        "原来鲜艳的红/蓝/绿只作少量暗淡残漆，不保留大面积亮漆；移除清晰广告字样，"
        "无干净银白表面、镜面灯条和亮黑新品质感。铝、不锈钢、塑料及橡胶按上面的专属材质旧化，不强行画铁锈。\n"
        "完整保留主体和安装端的数量、连接和比例；少量擦伤/磨损不能把原结构改成其它型号。"
    )
    old_region = cfg.active_edit_region
    cfg.active_edit_region = None
    directory = cfg.output_images_dir.parent / "weathered_references"
    directory.mkdir(parents=True, exist_ok=True)
    stages = []
    current = reference_path
    try:
        for stage in range(1, max(1, min(3, cfg.weathered_reference_rounds)) + 1):
            stage_prompt = prompt if stage == 1 else (
                "对当前同一物品做进一步报废旧化，完整形体和连接不变。必须清除剩余亮漆、橙红大色块和干净反射，"
                "以低饱和灰褐积尘、深褐旧氧化和旧油泥覆盖可见主体；"
                "钢/铁可见主体约80–90%呈明显锈蚀/污垢，原漆只剩零散暗淡残片，此为合成外观目标。"
                "铝、不锈钢、塑料和橡胶严格按下面专属旧化处理，不能画成铁锈。"
                + profile["weathering_prompt"]
                + "。只输出一件完整旧物的参考照片，平整灰褐背景；无其它钢件、无文字、无宫格、无摄影棚灯条反射。"
            )
            payload = build_request_payload([current], stage_prompt, cfg)
            result = call_image_api(payload, cfg)
            urls = extract_image_urls(result, cfg.image_protocol)
            if not urls:
                raise RuntimeError("旧化参考生成未返回图片")
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            path = directory / f"{profile['id']}_stage{stage}_{stamp}.{cfg.output_format}"
            path.write_bytes(download_image(urls[0]))
            stages.append({"stage": stage, "source": str(current), "output": str(path), "prompt": stage_prompt})
            current = path
    finally:
        cfg.active_edit_region = old_region
    path = current
    cfg.active_weathered_reference = path
    cfg.active_reference_kind = "weathered_single_view"
    cfg.active_reference_preparation = {
        "stage": "generated_discarded_state_reference",
        "source_reference": str(reference_path),
        "background_source": str(background_path),
        "generated_reference": str(path),
        "model": cfg.model,
        "prompt": prompt,
        "stages": stages,
        "acceptance": "assessed_with_final_scene_not_assumed_from_filename",
    }
    save_json(path.with_suffix(".json"), cfg.active_reference_preparation)
    print(f"  [旧化参考] 已生成 {path.name}；最终场景仍需视觉复核")
    return path

# ============================================================
#  call_image_api
# ============================================================

def call_image_api(payload: Dict[str, Any], cfg: AppConfig) -> Dict[str, Any]:
    """按已绑定的图像地址和密钥发起请求。"""
    print(f"[DEBUG] 图像生成凭据已配置: {bool(cfg.image_api_key)}")
    print(f"[DEBUG] 使用模型: {cfg.model}")
    print(f"[DEBUG] 图像协议: {cfg.image_protocol}")

    endpoint = cfg.image_api_url
    print(f"[DEBUG] 请求端点: {endpoint}")

    headers = {
        "Authorization": f"Bearer {cfg.image_api_key}",
        "Content-Type": "application/json",
    }

    session = get_session_with_ssl_bypass()

    try:
        response = session.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=cfg.timeout_seconds
        )

        if response.status_code != 200:
            print(f"  [API Error] 状态码: {response.status_code}")
            print(f"  [API Error] 响应内容: {response.text}")
            try:
                error_json = response.json()
                err = error_json.get("error") if isinstance(error_json.get("error"), dict) else error_json
                if err.get("code"):
                    print(f"  [API Error] 错误代码: {err.get('code')}")
                if err.get("message"):
                    print(f"  [API Error] 错误消息: {err.get('message')}")
            except Exception:
                pass
            response.raise_for_status()

        result = response.json()
        print(f"  [API Response] 响应键: {list(result.keys())}")
        return result

    except Exception as e:
        print(f"  [API Error] {e}")
        raise

# ============================================================
#  extract_image_urls - 按图像协议解析
# ============================================================

def extract_image_urls(result: Dict[str, Any], image_protocol: str) -> List[str]:
    """ali_wan 读 output.choices；huoshan_seedream 读 data[].url。"""
    urls: List[str] = []

    if image_protocol == "ali_wan":
        output = result.get("output", {})
        for choice in output.get("choices", []):
            message = choice.get("message", {})
            for item in message.get("content", []):
                if isinstance(item, dict) and item.get("image"):
                    urls.append(item["image"])
    elif image_protocol == "huoshan_seedream":
        for item in result.get("data") or []:
            if isinstance(item, dict) and item.get("url"):
                urls.append(item["url"])
    else:
        raise ValueError(f"未知图像协议: {image_protocol}")

    if not urls:
        debug_path = Path("debug_response.json")
        save_json(debug_path, result)
        print(f"  [提取URL] 完整响应已保存到: {debug_path}")
        raise ValueError(f"无法从响应中提取图片URL，响应键: {list(result.keys())}")

    print(f"  [提取URL] 找到 {len(urls)} 个图片URL")
    return urls

def download_image(url: str) -> bytes:
    session = get_session_with_ssl_bypass()
    response = session.get(url, timeout=60)
    response.raise_for_status()
    return response.content

def _origin_of(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"无法解析地址: {url}")
    return f"{parsed.scheme}://{parsed.netloc}"

def _chat_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    suffix = "/chat/completions"
    if base.endswith(suffix):
        return base
    return base + suffix

def load_bailian_raw() -> Dict[str, Any]:
    """中转函数直接读取 config_bailian.yaml。"""
    config_path = Path(__file__).resolve().parent / "config_bailian.yaml"
    print(f"[DEBUG] 读取配置文件: {config_path}")
    print(f"[DEBUG] 配置文件是否存在: {config_path.exists()}")
    return load_yaml_config(config_path)

def load_app_raw() -> Dict[str, Any]:
    """读取 config.yaml，用于选择生图模型和 VLM 模型。"""
    config_path = Path(__file__).resolve().parent / "config.yaml"
    print(f"[DEBUG] 读取应用配置: {config_path}")
    print(f"[DEBUG] 应用配置是否存在: {config_path.exists()}")
    return load_yaml_config(config_path)

def route_models(raw_bailian: Dict[str, Any], raw_app: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """系列由入口按模型名决定，这里不再跨家混选。"""
    del raw_bailian, raw_app
    return None

def select_models(raw_bailian: Dict[str, Any], raw_app: Dict[str, Any]) -> Tuple[str, str]:
    del raw_bailian, raw_app
    print(f"[DEBUG] 模型来源: {THIS_SERIES} -> {CHOSEN_IMAGE} / {CHOSEN_VLM}")
    return CHOSEN_IMAGE, CHOSEN_VLM

def models_for_this_series(image_model: Optional[str], vlm_model: Optional[str]) -> Tuple[str, str]:
    """只接受本系列、并且十一点配置里已经出现过的模型名。"""
    image = image_model or DEFAULT_IMAGE
    vlm = vlm_model or DEFAULT_VLM
    if image not in ALLOWED_IMAGES or vlm not in ALLOWED_VLMS:
        raise ValueError(
            f"{THIS_SERIES} 只接受生图 {list(ALLOWED_IMAGES)}，VLM {list(ALLOWED_VLMS)}"
        )
    return image, vlm

def bind_by_model(raw: Dict[str, Any], image_model: str, vlm_model: str) -> Dict[str, str]:
    """按模型名绑定密钥、地址和协议。图像与 VLM 互不影响。"""
    if image_model.startswith("wan"):
        image_protocol = "ali_wan"
        image_api_key = raw.get("api_key_ali") or ""
        image_base = raw.get("photo_generate_api_url_ali") or ""
        image_api_url = _origin_of(image_base) + "/api/v1/services/aigc/multimodal-generation/generation"
    elif image_model.startswith("doubao-seedream"):
        image_protocol = "huoshan_seedream"
        image_api_key = raw.get("api_key_huoshan") or ""
        image_api_url = raw.get("photo_generate_api_url_huoshan") or ""
    else:
        raise ValueError(f"无法为图像模型选择协议: {image_model}")

    if vlm_model.startswith("qwen"):
        vlm_api_key = raw.get("api_key_ali") or ""
        vlm_api_url = _chat_endpoint(raw.get("vlm_api_ali_url_") or raw.get("vlm_api_url") or "")
    elif vlm_model.startswith("doubao-seed"):
        vlm_api_key = raw.get("api_key_huoshan") or ""
        vlm_api_url = _chat_endpoint(raw.get("vlm_api_huoshan_url") or "")
    else:
        raise ValueError(f"无法为 VLM 模型选择协议: {vlm_model}")

    if not image_api_key:
        raise ValueError(f"图像模型 {image_model} 未配置 API Key")
    if not vlm_api_key:
        raise ValueError(f"VLM 模型 {vlm_model} 未配置 API Key")
    if not image_api_url:
        raise ValueError(f"图像模型 {image_model} 未配置地址")
    if not vlm_api_url:
        raise ValueError(f"VLM 模型 {vlm_model} 未配置地址")

    return {
        "image_api_key": image_api_key,
        "image_api_url": image_api_url,
        "image_protocol": image_protocol,
        "model": image_model,
        "vlm_api_key": vlm_api_key,
        "vlm_api_url": vlm_api_url,
        "vlm_model": vlm_model,
    }

def load_app_config() -> AppConfig:
    base_dir = Path(__file__).resolve().parent
    raw = load_bailian_raw()
    raw_app = load_app_raw()
    image_model, vlm_model = select_models(raw, raw_app)
    bound = bind_by_model(raw, image_model, vlm_model)

    print(f"[DEBUG] 图像模型: {bound['model']}")
    print(f"[DEBUG] 图像协议: {bound['image_protocol']}")
    print(f"[DEBUG] 图像端点: {bound['image_api_url']}")
    print(f"[DEBUG] 图像凭据已配置: {bool(bound['image_api_key'])}")
    print(f"[DEBUG] VLM 模型: {bound['vlm_model']}")
    print(f"[DEBUG] VLM 端点: {bound['vlm_api_url']}")
    print(f"[DEBUG] VLM 凭据已配置: {bool(bound['vlm_api_key'])}")

    cfg = AppConfig(
        image_api_key=bound["image_api_key"],
        model=bound["model"],
        image_api_url=bound["image_api_url"],
        image_protocol=bound["image_protocol"],
        size=raw.get("size", "2K"),
        output_format=raw.get("output_format", "png"),
        watermark=bool(raw.get("watermark", False)),
        max_images=int(raw.get("max_images", 1)),
        object_resize_scale=float(raw_app.get("object_resize_scale", raw.get("object_resize_scale", 0.022))),
        object_resize_scales=parse_object_resize_scales(raw_app),
        object_size_briefs=parse_object_size_briefs(raw_app),
        object_canvas_scale=float(raw_app.get("object_canvas_scale", raw.get("object_canvas_scale", 1.0))),
        object_opacity=float(raw.get("object_opacity", 0.5)),
        object_anchor=raw.get("object_anchor", "center"),
        timeout_seconds=int(raw.get("timeout_seconds", 300)),
        sleep_seconds=float(raw.get("sleep_seconds", 1.0)),
        total_num=int(raw.get("total_num", 100)),
        seed=raw.get("seed"),
        backgrounds_dir=base_dir / raw.get("backgrounds_dir", "input/backgrounds"),
        objects_dir=base_dir / raw.get("objects_dir", "input/objects"),
        prompt_templates_path=base_dir / raw.get("prompt_templates_path", "prompts/prompt_templates.json"),
        negative_prompt_path=base_dir / raw.get("negative_prompt_path", "prompts/negative_prompt.txt"),
        output_images_dir=base_dir / raw.get("output_images_dir", "output/images"),
        output_final_images_dir=base_dir / raw.get("output_final_images_dir", "output/final_images"),
        output_metadata_dir=base_dir / raw.get("output_metadata_dir", "output/metadata"),
        output_failed_dir=base_dir / raw.get("output_failed_dir", "output/failed"),
        output_bbox_dir=base_dir / raw.get("output_bbox_dir", "output/bbox"),
        output_analysis_dir=base_dir / raw.get("output_analysis_dir", "output/analysis"),
        vlm_api_key=bound["vlm_api_key"],
        vlm_api_url=bound["vlm_api_url"],
        vlm_model=bound["vlm_model"],
        enable_vlm_optimization=raw_app.get("enable_vlm_optimization", raw.get("enable_vlm_optimization", True)),
        max_optimization_rounds=int(raw_app.get("max_optimization_rounds", raw.get("max_optimization_rounds", 3))),
        size_tolerance=raw_app.get("size_tolerance", raw.get("size_tolerance", "class_range")),
        feed_object_views=bool(raw_app.get("feed_object_views", raw.get("feed_object_views", True))),
        max_object_views=int(raw_app.get("max_object_views", raw.get("max_object_views", 3))),
        six_views_dir=(base_dir / raw_app.get("six_views_dir", raw.get("six_views_dir", "input/objects/six_views"))).resolve(),
        use_interactive_edit_region=bool(raw_app.get("use_interactive_edit_region", True)),
        vlm_thinking_mode=raw_app.get("vlm_thinking_mode", "disabled"),
        prepare_weathered_reference=bool(raw_app.get("prepare_weathered_reference", True)),
        weathered_reference_rounds=int(raw_app.get("weathered_reference_rounds", 2)),
    )
    if raw_app.get("use_object_catalog", False):
        cfg.object_catalog_path = base_dir / raw_app.get("object_catalog_path", "profiles/object_profiles.json")
        cfg.object_catalog = load_per_item_catalog(cfg.object_catalog_path, cfg.six_views_dir)
        background_catalog_path = base_dir / raw_app.get("background_catalog_path", "profiles/background_profiles.json")
        cfg.background_catalog = json.loads(background_catalog_path.read_text(encoding="utf-8"))
        print(f"[INFO] 已启用逐件目录：{len(cfg.object_catalog['objects'])} 张参考图，独立尺寸/prompt")
    print(f"[DEBUG] 已加载 {len(cfg.object_resize_scales)} 类物体真实缩放尺寸")
    print(f"[DEBUG] 已加载 {len(cfg.object_size_briefs)} 类物体结构尺寸")
    return cfg

# ============ 主函数 ============

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch generate scrap foreign-object synthesis images with VLM optimization (Bailian Platform)")
    parser.add_argument("--total", type=int, default=None, help="总生成数量")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--no-vlm", action="store_true", help="禁用VLM优化，使用传统模式")
    parser.add_argument("--with-vlm", action="store_true", help="本次运行启用背景分析与生成结果复核")
    parser.add_argument("--max-rounds", type=int, default=None, help="最大优化轮数")
    parser.add_argument("--backgrounds", type=str, nargs="+", help="指定背景图片路径或文件名")
    parser.add_argument("--objects", type=str, nargs="+", help="指定物品 ID（S01–S38）、图片路径或文件名；多件时加权选取")
    parser.add_argument("--random", action="store_true", help="随机选取背景图（默认顺序遍历）")
    parser.add_argument("--analyze-only", action="store_true", help="仅分析已有的Prompt数据，不生成")
    parser.add_argument("--model", type=str, default=None, help="覆盖本提供方的生图模型")
    parser.add_argument("--vlm-model", type=str, default=None, help="覆盖本提供方的 VLM")
    parser.add_argument("--output-root", type=Path, default=None, help="本次运行的独立结果目录")
    parser.add_argument("--rust-reference", type=Path, default=None, help="指定当前单件物品的现成生锈六视图，直接融合")
    args = parser.parse_args()
    handoff_to_series(args, parser)

    cfg = load_app_config()
    supplied_rust_reference = None
    if args.rust_reference:
        supplied_rust_reference = args.rust_reference.expanduser().resolve()
        if not supplied_rust_reference.is_file() or supplied_rust_reference.suffix.lower() not in IMAGE_EXTS:
            parser.error("生锈六视图文件不存在或格式不支持")
        if not args.objects or len(args.objects) != 1 or not cfg.object_catalog:
            parser.error("rust-reference 需与一个物品 ID 配合使用")
        cfg.prepare_weathered_reference = False
    if args.with_vlm and args.no_vlm:
        parser.error("with-vlm 和 no-vlm 不能同时使用")
    if args.with_vlm:
        cfg.enable_vlm_optimization = True
    if args.output_root:
        run_root = args.output_root.expanduser().resolve()
        for field_name, folder in (
            ("output_images_dir", "images"),
            ("output_final_images_dir", "final_images"),
            ("output_metadata_dir", "metadata"),
            ("output_failed_dir", "failed"),
            ("output_bbox_dir", "bbox"),
            ("output_analysis_dir", "analysis"),
        ):
            setattr(cfg, field_name, run_root / folder)

    if args.model or args.vlm_model:
        try:
            image_model, vlm_model = models_for_this_series(args.model, args.vlm_model)
        except ValueError as exc:
            parser.error(str(exc))
        global CHOSEN_IMAGE, CHOSEN_VLM
        CHOSEN_IMAGE, CHOSEN_VLM = image_model, vlm_model
        bound = bind_by_model(load_bailian_raw(), image_model, vlm_model)
        cfg.image_api_key = bound["image_api_key"]
        cfg.image_api_url = bound["image_api_url"]
        cfg.image_protocol = bound["image_protocol"]
        cfg.model = bound["model"]
        cfg.vlm_api_key = bound["vlm_api_key"]
        cfg.vlm_api_url = bound["vlm_api_url"]
        cfg.vlm_model = bound["vlm_model"]
        cfg.output_analysis_dir = cfg.output_analysis_dir / cfg.model
        print(f"[DEBUG] 命令行覆盖生图模型: {cfg.model} ({cfg.image_protocol})")
        print(f"[DEBUG] 图像端点: {cfg.image_api_url}")
        print(f"[DEBUG] 命令行覆盖 VLM: {cfg.vlm_model}")
        print(f"[DEBUG] VLM 端点: {cfg.vlm_api_url}")
        print(f"[DEBUG] Prompt 统计目录: {cfg.output_analysis_dir}")
    
    if not cfg.image_api_key or not cfg.vlm_api_key:
        print("❌ 错误：未设置API Key")
        print("请在 config_bailian.yaml 中配置 api_key_ali 与 api_key_huoshan")
        return
    
    # 创建分析目录
    cfg.output_analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化Prompt分析器
    prompt_analyzer = PromptAnalyzer(cfg.output_analysis_dir)
    
    # 如果仅分析模式，打印统计后退出
    if args.analyze_only:
        prompt_analyzer.print_summary()
        prompt_analyzer.save_stats()
        return
    
    if args.no_vlm:
        cfg.enable_vlm_optimization = False
    if args.max_rounds is not None:
        cfg.max_optimization_rounds = args.max_rounds
    
    seed = args.seed if args.seed is not None else cfg.seed
    if seed is not None:
        random.seed(seed)

    cfg.output_images_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_final_images_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_metadata_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_failed_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_bbox_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_analysis_dir.mkdir(parents=True, exist_ok=True)

    prompt_templates = load_prompt_templates(cfg.prompt_templates_path)
    negative_prompt = load_negative_prompt(cfg.negative_prompt_path)

    backgrounds = list_images(cfg.backgrounds_dir)
    if not backgrounds:
        raise FileNotFoundError(f"No background images found in {cfg.backgrounds_dir}")

    if args.backgrounds:
        specified_bg = [Path(p) for p in args.backgrounds]
        backgrounds = [bg for bg in backgrounds if bg in specified_bg or bg.name in args.backgrounds]
        if not backgrounds:
            print(f"Warning: No specified backgrounds found, using all backgrounds")
            backgrounds = list_images(cfg.backgrounds_dir)

    total_to_process = args.total if args.total is not None else cfg.total_num
    
    if args.random:
        if total_to_process > len(backgrounds):
            print(f"⚠️ 指定数量 {total_to_process} 超过背景图总数 {len(backgrounds)}，将使用放回采样")
            selected_backgrounds = [random.choice(backgrounds) for _ in range(total_to_process)]
        else:
            selected_backgrounds = random.sample(backgrounds, total_to_process)
        print(f"📁 随机选取 {len(selected_backgrounds)} 张背景图")
    else:
        if total_to_process > len(backgrounds):
            print(f"⚠️ 指定数量 {total_to_process} 超过背景图总数 {len(backgrounds)}，将从头开始重复")
            selected_backgrounds = []
            for i in range(total_to_process):
                selected_backgrounds.append(backgrounds[i % len(backgrounds)])
        else:
            selected_backgrounds = backgrounds[:total_to_process]
        print(f"📁 顺序处理 {len(selected_backgrounds)} 张背景图")

    object_items = (
        per_item_catalog_items(cfg.object_catalog) if cfg.object_catalog
        else six_view_object_items(
            cfg.six_views_dir,
            Path(__file__).resolve().parent / "profiles/object_profiles.json",
        )
    )
    if not object_items:
        raise FileNotFoundError(f"No object images found in {cfg.objects_dir}")
    catalog_items = list(object_items)

    if args.objects:
        specified = set(args.objects)
        matched = [
            item for item in object_items
            if str(item["path"]) in specified
            or item["path"].name in specified
            or item["filename"] in specified
            or item.get("profile_id") in specified
            or (item.get("profile") or {}).get("original_filename") in specified
        ]
        if not matched:
            if cfg.object_catalog:
                raise ValueError("指定物品不在逐件目录中；使用S01–S38或准确六视图文件名")
            print(f"Warning: 未找到指定异物 {args.objects}，改用全部异物图")
        else:
            object_items = matched
            print(f"📁 指定异物 {len(object_items)} 张: {[it['filename'] for it in object_items]}")

    available_classes = list(set([item["class"] for item in object_items]))
    print(f"[INFO] 可用类别: {available_classes}")
    
    class_items = {}
    for item in object_items:
        cls = item["class"]
        if cls not in class_items:
            class_items[cls] = []
        class_items[cls].append(item)
    
    print(f"[INFO] 类别分布:")
    for cls, items in class_items.items():
        low_weight_count = len([it for it in items if it["weight"] < 1.0])
        print(f"  {cls}: {len(items)} 张 (其中低权重 {low_weight_count} 张)")

    vlm_analyzer = VLMAnalyzer(cfg, prompt_analyzer) if cfg.enable_vlm_optimization else None

    for bg_idx, background_path in enumerate(selected_backgrounds):
        print(f"\n{'='*60}")
        print(f"📷 处理第 {bg_idx+1}/{len(selected_backgrounds)} 张背景图: {background_path.name}")
        print(f"{'='*60}")
        
        selected_item = choose_weighted_object(object_items)
        if not selected_item:
            print(f"❌ 没有可用物体图片，跳过")
            continue
        
        object_class = selected_item["class"]
        object_path = selected_item["path"]
        cfg.active_object_profile = selected_item.get("profile")
        cfg.active_scene_context = None
        cfg.active_edit_region = None
        cfg.active_position_xy = None
        cfg.active_weathered_reference = None
        cfg.active_reference_kind = "six_view_atlas"
        cfg.active_reference_preparation = None
        if supplied_rust_reference:
            import hashlib
            object_path = supplied_rust_reference
            cfg.active_object_profile = dict(cfg.active_object_profile)
            cfg.active_reference_kind = "weathered_six_view_atlas"
            cfg.active_weathered_reference = object_path
            cfg.active_reference_preparation = {
                "stage": "user_supplied_weathered_six_view_atlas",
                "source_reference": str(object_path),
                "sha256": hashlib.sha256(object_path.read_bytes()).hexdigest(),
                "reference_generation_calls": 0,
            }
            cfg.active_object_profile["prompt"] = render_per_item_prompt(
                cfg.active_object_profile, reference_kind=cfg.active_reference_kind,
            )
        cfg.active_background_profile = next(
            (item for item in (cfg.background_catalog or {}).get("backgrounds", []) if item["filename"] == background_path.name), None
        )
        if cfg.active_object_profile:
            cfg.vlm_evaluator_prompt = per_item_evaluator_prompt(cfg.active_object_profile)
        view_paths: List[Path] = []
        size_data = calculate_optimal_size(background_path, object_class, cfg)
        size_state = {"ratio": size_data["target_ratio"]}
        print(f"  [物体] 选择: {object_path.name} (类别: {object_class}, 权重: {selected_item['weight']})")
        print(f"  [尺寸] {size_data['description']}")
        if cfg.feed_object_views:
            view_paths = [object_path] if supplied_rust_reference else collect_object_views(
                object_path,
                object_class,
                catalog_items,
                cfg.max_object_views,
                cfg.six_views_dir,
            )
            print(
                f"  [多视图] 参考 {len(view_paths)} 张（只认外形，不贴图）: "
                f"{[p.name for p in view_paths]}"
            )
        
        base_prompt = (
            cfg.active_object_profile["prompt"] if cfg.active_object_profile
            else random.choice(prompt_templates[object_class])
        )
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        image_id = f"{bg_idx:06d}_{background_path.stem}_{object_class}_{cfg.model}_{timestamp}"
        meta_name = f"{image_id}.json"

        try:
            final_prompt = base_prompt
            optimization_history = []
            vlm_used = False
            vlm_success = False
            vlm_analysis_json = None

            full_prompt = base_prompt
            result = None
            position_sample = None
            orientation_sample = None
            last_image_path: Optional[Path] = None
            outcome: Dict[str, Any] = {}

            if vlm_analyzer and object_path:
                vlm_used = True
                print(f"\n[VLM优化] 处理背景 {bg_idx+1}/{len(selected_backgrounds)}: {background_path.name} -> {object_class}")
                print(f"  [VLM] 背景图片: {background_path}")

                def generate_with_prompt(
                    prompt: str,
                    round_num: int,
                    previous_path: Optional[Path],
                    position_hint: str,
                    orientation_hint: str,
                    cls: str = object_class,
                ):
                    current_references = view_paths
                    if cfg.active_object_profile and cfg.prepare_weathered_reference:
                        aged_reference = prepare_scene_weathered_reference(cfg, object_path, background_path)
                        current_references = [aged_reference]
                    if cfg.active_object_profile:
                        with Image.open(background_path) as source:
                            canvas_size = source.size
                        policy = cfg.active_object_profile["size_policy"]
                        cfg.active_edit_region = fusion_edit_region(
                            cfg.active_position_xy or [0.5, 0.55],
                            size_state["ratio"], canvas_size,
                            policy["cross_mm"] / policy["reference_length_mm"],
                        )
                    round_prompt = build_prompt(
                        cls, prompt, negative_prompt,
                        background_path=background_path,
                        position_hint=position_hint,
                        orientation_hint=orientation_hint,
                        cfg=cfg,
                        target_ratio=size_state["ratio"],
                    )
                    api_result, labeled = generate_single_call(
                        cfg,
                        background_path,
                        round_prompt,
                        previous_path,
                        reference_paths=current_references,
                    )
                    urls = extract_image_urls(api_result, cfg.image_protocol)
                    image_bytes = download_image(urls[0])
                    out_name = f"{image_id}_round{round_num}.{cfg.output_format}"
                    final_round_path = cfg.output_images_dir / out_name
                    final_round_path.write_bytes(image_bytes)
                    return final_round_path, api_result, labeled

                outcome = vlm_analyzer.run_optimization_loop(
                    object_path=object_path,
                    background_path=background_path,
                    object_class=object_class,
                    base_prompt=base_prompt,
                    initial_generation_func=generate_with_prompt,
                    image_id=image_id,
                    size_state=size_state,
                )
                final_prompt = outcome["final_prompt"]
                full_prompt = outcome["full_prompt"]
                optimization_history = outcome["history"]
                vlm_success = outcome["success"]
                last_image_path = outcome["last_image_path"]
                result = outcome["api_result"]
                position_sample = outcome["position_sample"]
                orientation_sample = outcome["orientation_sample"]
                vlm_analysis_json = outcome["opening"]
                print(f"  [VLM] 优化结果: {'成功' if vlm_success else '达到最大轮数'}")
            else:
                if cfg.active_object_profile and cfg.prepare_weathered_reference:
                    aged_reference = prepare_scene_weathered_reference(cfg, object_path, background_path)
                    view_paths = [aged_reference]
                scrap_geometry = fallback_scrap_geometry()
                position_sample = sample_foreign_object_position(scrap_geometry)
                orientation_sample = sample_object_pose(object_class)
                position_hint = position_sample["description"]
                orientation_hint = orientation_sample["description"]
                if cfg.active_object_profile:
                    with Image.open(background_path) as source:
                        canvas_size = source.size
                    policy = cfg.active_object_profile["size_policy"]
                    cfg.active_edit_region = fusion_edit_region(
                        position_sample["sampled_xy"], size_state["ratio"], canvas_size,
                        policy["cross_mm"] / policy["reference_length_mm"],
                    )
                    cfg.active_position_xy = position_sample["sampled_xy"]
                print(
                    f"  [位置] 采样 ({position_sample['sampled_xy'][0]:.3f},"
                    f"{position_sample['sampled_xy'][1]:.3f}) "
                    f"p={position_sample['sampled_probability']:.3f} | {position_hint}"
                )
                print(
                    f"  [姿态] {orientation_sample['pose_cn']} "
                    f"{orientation_sample['facing_cn']} "
                    f"相对水平{orientation_sample['tilt_from_horizontal_deg']:+.1f}°"
                )
                round_prompt = build_prompt(
                    object_class, base_prompt, negative_prompt,
                    background_path=background_path,
                    position_hint=position_hint,
                    orientation_hint=orientation_hint,
                    cfg=cfg,
                    target_ratio=size_state["ratio"],
                )
                result, full_prompt = generate_single_call(
                    cfg, background_path, round_prompt, reference_paths=view_paths,
                )
                final_prompt = full_prompt
                urls = extract_image_urls(result, cfg.image_protocol)
                image_bytes = download_image(urls[0])
                last_image_path = cfg.output_images_dir / f"{image_id}.{cfg.output_format}"
                last_image_path.write_bytes(image_bytes)

            if last_image_path is None or not last_image_path.exists():
                raise RuntimeError("生图未产出图片")

            final_name = f"{image_id}.{cfg.output_format}"
            final_path = cfg.output_final_images_dir / final_name
            final_path.write_bytes(last_image_path.read_bytes())
            saved_paths = [str(final_path)]

            metadata = {
                "image_id": image_id,
                "index": bg_idx,
                "background_image": str(background_path),
                "object_image": str(object_path) if object_path else None,
                "object_views": [str(p) for p in view_paths],
                "object_class": object_class,
                "object_profile_id": (cfg.active_object_profile or {}).get("id"),
                "object_physical_name": (cfg.active_object_profile or {}).get("name"),
                "working_dimensions_mm": (cfg.active_object_profile or {}).get("dimensions_mm"),
                "dimension_evidence": (cfg.active_object_profile or {}).get("dimension_evidence"),
                "background_calibration": cfg.active_background_profile,
                "object_scale": {
                    "target_ratio": (
                        ((outcome or {}).get("size_rounds") or [{}])[-1].get("target_ratio", size_state["ratio"])
                        if vlm_used else size_state["ratio"]
                    ),
                    "scale_min": size_data["scale_min"],
                    "scale_max": size_data["scale_max"],
                    "description": size_data["description"],
                },
                "base_prompt": base_prompt,
                "final_prompt": final_prompt,
                "full_prompt": full_prompt,
                "negative_prompt": negative_prompt,
                "model": cfg.model,
                "size": cfg.size,
                "output_format": cfg.output_format,
                "watermark": cfg.watermark,
                "generated_images": saved_paths,
                "vlm_enabled": cfg.enable_vlm_optimization,
                "vlm_used": vlm_used,
                "vlm_success": vlm_success,
                "vlm_analysis": vlm_analysis_json,
                "scrap_position": position_sample,
                "object_orientation": orientation_sample,
                "optimization_rounds": len([h for h in optimization_history if h.get("stage", "").startswith("evaluation_round")]),
                "optimization_history": optimization_history if optimization_history else None,
                "round_sharpness": (outcome or {}).get("round_sharpness") if vlm_used else None,
                "judgment_history": (outcome or {}).get("judgment_history") if vlm_used else None,
                "size_rounds": (outcome or {}).get("size_rounds") if vlm_used else None,
                "selected_round": (outcome or {}).get("selected_round") if vlm_used else None,
                "selected_reason": (outcome or {}).get("selected_reason") if vlm_used else None,
                "scene_context": cfg.active_scene_context,
                "reference_preparation": cfg.active_reference_preparation,
                "fusion_reference_images": [str(cfg.active_weathered_reference)] if cfg.active_weathered_reference else [str(p) for p in view_paths],
                "edit_region_xyxy": cfg.active_edit_region,
                "fusion_gate": outcome.get("fusion_gate"),
                "selected_evaluation": outcome.get("selected_evaluation"),
                "fusion_pipeline_version": "1.1.0",
                "quality_status": "vlm_passed_pending_human_review" if vlm_success else "unaccepted_requires_review",
                "review_result": None,
                "notes": f"Generated with {cfg.model} (HTTP), VLM: {cfg.vlm_model}",
                "api_result": result,
            }
            save_json(cfg.output_metadata_dir / meta_name, metadata)
            print(f"[OK] {image_id} -> {len(saved_paths)} image(s) (VLM: {vlm_success if vlm_used else 'disabled'})")
            time.sleep(cfg.sleep_seconds)
            
        except Exception as exc:
            failed = {
                "image_id": image_id,
                "index": bg_idx,
                "background_image": str(background_path),
                "object_image": str(object_path) if object_path else None,
                "object_views": [str(p) for p in view_paths],
                "object_class": object_class,
                "error": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            save_json(cfg.output_failed_dir / f"{image_id}.json", failed)
            print(f"[FAILED] {image_id}: {exc}")

    print(f"\n{'='*60}")
    print(f"✅ 全部完成！共处理 {len(selected_backgrounds)} 张背景图")
    print(f"{'='*60}")
    
    prompt_analyzer.save_stats()
    prompt_analyzer.print_summary()

if __name__ == "__main__":
    main()
