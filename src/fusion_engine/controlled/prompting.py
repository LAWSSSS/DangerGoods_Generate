"""Readable scene-first prompt documents shared by export, preview and generation."""

from __future__ import annotations

import hashlib
import re
from importlib.resources import files
from typing import Any

import yaml

from fusion_engine.config import reject_secrets
from fusion_engine.controlled.appearance import material_plan
from fusion_engine.controlled.framing import map_placement, map_point
from fusion_engine.errors import ConfigError

SECTION_IDS = (
    "task",
    "inputs",
    "scene_style",
    "identity",
    "placement",
    "material",
    "interaction",
    "negative",
    "output",
)
MATERIAL_NAMES = {
    "carbon_steel": "碳钢",
    "painted_steel": "喷漆钢",
    "cast_iron": "铸铁",
    "stainless_steel": "不锈钢",
    "aluminium": "铝",
    "galvanized_steel": "镀锌钢",
    "copper": "铜",
    "brass": "黄铜",
    "plastic": "塑料",
    "rubber": "橡胶",
    "wood": "木材",
    "paper": "纸/纸板",
    "glass": "玻璃",
    "fabric": "织物",
    "ceramic": "陶瓷",
    "unknown": "未核验基材",
}
AGEING_LEVELS = {
    0: "近新；不额外添加重锈，材质响应仍服从现场光照",
    1: "轻度使用痕迹；少量失光、擦痕和积灰",
    2: "明显使用与风化；旧化按材料和暴露位置分布",
    3: "长期弃置外观；残漆失光、局部腐蚀/污垢分层，保持现场低对比或高对比的实际条件",
    4: "严重表面风化；结构损伤只限明确允许的部位",
}


def unique(values: list[str]) -> list[str]:
    result, seen = [], set()
    for value in values:
        value = re.sub(r"\s+", " ", value).strip(" \n-；;")
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def percent(value: float) -> str:
    return f"{value * 100:.4f}".rstrip("0").rstrip(".") + "%"


def prompt_policy(version: str = "2.1") -> tuple[dict, str]:
    if version not in ("2.1", "2.2", "2.3", "3.0"):
        raise ConfigError("未支持的提示词版本")
    raw = (
        files("fusion_engine")
        .joinpath(f"templates/scene_fusion_v{version.replace('.', '_')}.yaml")
        .read_bytes()
    )
    try:
        policy = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError("提示词模板不是有效 YAML；未输出原始内容") from exc
    reject_secrets(policy, "prompt_template")
    if not isinstance(policy, dict) or not isinstance(policy.get("version"), str):
        raise ConfigError("提示词模板缺少版本信息")
    sections = policy.get("sections", [])
    if (
        not isinstance(sections, list)
        or len(sections) != len(SECTION_IDS)
        or any(not isinstance(value, dict) for value in sections)
        or [value.get("id") for value in sections] != list(SECTION_IDS)
    ):
        raise ConfigError("提示词模板必须按顺序包含九个已定义区块")
    for section in sections:
        if (
            not isinstance(section.get("title"), str)
            or not section["title"].strip()
            or not isinstance(section.get("rules"), list)
            or any(not isinstance(rule, str) or not rule.strip() for rule in section["rules"])
        ):
            raise ConfigError("提示词区块必须包含标题与文本规则列表")
    return policy, hashlib.sha256(raw).hexdigest()


def build_prompt_document(
    data: dict[str, Any],
    object_plans: list[dict] | None = None,
    *,
    photometry: dict | None = None,
    frame: dict | None = None,
) -> dict:
    policy, template_hash = prompt_policy(data.get("prompt_version", "2.1"))
    materials = (
        {value["id"]: value["material"] for value in object_plans}
        if object_plans is not None
        else {item["id"]: material_plan(item) for item in data["objects"]}
    )
    additions = {name: [] for name in SECTION_IDS}
    global_directives = data.get("prompt_directives", {})
    associations = []
    for item in data["objects"]:
        for reference in item["references"]:
            index = len(associations) + 1
            associations.append(
                {"image_index": index, "object_id": item["id"], "view_id": reference["view_id"]}
            )
            label = "生锈六视图参考" if reference["source_kind"] == "rusted_atlas" else "物体参考"
            additions["inputs"].append(
                f"图{index}：{item['id']} / {item['category']} / {reference['view_id']}，{label}。"
            )
    scene_index = len(associations) + 1
    hinted = data["generator"].get("layout_hint") == "procedural_proxy"
    additions["inputs"].append(
        f"图{scene_index}（原始场景）：以此图的空间、光色和成像作为统一依据。"
        if hinted
        else f"图{scene_index}（最后一张）：原始场景；以此图的空间、光色和成像作为统一依据。"
    )
    if hinted:
        additions["inputs"].append(
            f"图{scene_index + 1}（最后一张）是完整场景上的低细节落位引导：简单色块只表示物体占据的位置、完整尺度、倾角和开口。"
            "请在该位置重绘参考中的真实旧物，替换全部简单色块；依原始背景重新生成局部遮挡、接触和成像。"
            "引导不是融合成品、材质照片或实测三维模型；不保留平面色块，不复制参考白底，不新增第二件物体。"
        )
    cropped = bool(frame and frame["kind"] == "context_crop")
    if cropped:
        additions["output"].append(
            "只返回与当前局部输入图同画幅、同长宽比的完整场景；不要补回局部输入以外的背景，不重新取景，不添加原图裁切之外的叠字。"
        )
        additions["inputs"].append(
            "最后一张是原场景的局部上下文裁切。以下中心、宽高和接触点已经换算为这张输入图的坐标；不要再次套用整张原图的百分比。保持这张局部输入图的全部画幅与长宽比，不扩大、移动或重新裁切相机视野。"
        )
    additions["task"].append(
        "新增物体清单："
        + "；".join(f"{item['id']} / {item['category']}，1件" for item in data["objects"])
        + "。"
    )
    additions["scene_style"] += [
        "场景光照提示：" + data["scene"]["lighting"],
        "场景成像提示：" + data["scene"]["camera_style"],
    ]
    additions["scene_style"] += global_directives.get("scene_style", [])
    additions["negative"] += global_directives.get("negative", [])
    if policy["version"] in ("2.2", "2.3"):
        for anchor in (photometry or {}).get("anchors", []):
            rgb = "、".join(f"{value:.1f}" for value in anchor["rgb_median"])
            additions["scene_style"].append(
                f"原图局部采样 {anchor['id']}：{anchor['description']}；RGB8中位数约({rgb})。"
                "数值只描述该区域，不能用作整件物体的填充色、统一白平衡或反射率。"
            )
        if data["harmonization"]["mode"] == "paired_local":
            additions["scene_style"].append(
                "局部对照区："
                + "；".join(
                    value["description"]
                    for value in data["harmonization"].get("paired_anchors", [])
                )
                + "。这些对照区保持原有形状、纹理与明暗，新增物体服从其所在位置的现场成像。"
            )
    supports = {value["id"]: value for value in data["scene"]["support_regions"]}
    for item in data["objects"]:
        oid, directives = item["id"], item.get("prompt_directives", {})
        features = unique(
            item["identity"]["positive"]
            + item["identity"]["critical_parts"]
            + directives.get("identity", [])
        )
        additions["identity"].append(
            f"{oid} 必须保留：" + "；".join(value.rstrip("。；;") for value in features) + "。"
        )
        additions["negative"] += [
            f"{oid} 禁止：{value}" for value in unique(item["identity"]["negative"])
        ]
        p = map_placement(item["placement"], frame) if cropped else item["placement"]
        support_first = item["placement"].get("priority") == "support_first"
        placement_line = (
            f"{oid}：完整框中心 x={percent(p['center'][0])}、y={percent(p['center'][1])}；"
            f"宽={percent(p['width'])}×原图宽，高={percent(p['height'])}×原图高；"
            f"旋转={p['rotation_deg']:+g}°；"
            + ("允许边界裁切。" if p.get("allow_crop") else "完整框保留在画布内。")
        )
        if support_first:
            limits = item["placement"]["flexibility"]
            placement_line = placement_line.replace("完整框中心", "初始名义框中心")
            placement_line += (
                "以上是原区域与尺度参考，真实承托优先，不强制照搬初始倾角和中心。"
                f"在已有编辑包络内，中心在原归一化坐标平面移动距离最多{percent(limits['max_center_shift'])}，"
                f"各尺寸比例为初始值{limits['scale_range'][0]:g}–{limits['scale_range'][1]:g}倍，"
                f"投影转角最多微调±{limits['max_rotation_delta_deg']:g}°。"
                "微调是预览请求，实际姿态和接触需重新核验；不扩大接触或编辑授权。"
                "原钢面不能移动、加宽或变平，不新增垫块/托架/遮挡钢条，不用黑影填空气。"
            )
        additions["placement"].append(placement_line)
        if p.get("pose_3d_notes"):
            additions["placement"].append(f"{oid} 姿态提示：{p['pose_3d_notes']}")
        if policy["version"] in ("2.2", "2.3"):
            spatial = item.get("spatial_constraints", {})
            if spatial.get("occluded_fraction_range"):
                low, high = spatial["occluded_fraction_range"]
                additions["interaction"].append(
                    f"{oid} 计划由原前景遮挡约{percent(low)}–{percent(high)}的完整投影面积；"
                    "按实际交叉位置发生遮挡，不为凑比例移动钢条、扩粗钢条或缩短物体。"
                )
            if "forbidden" in spatial:
                additions["placement"].append(
                    f"{oid} 禁入约束：{spatial['forbidden']['evidence']['notes']}"
                )
            for name in ("obj_full", "fg_occ", "contact"):
                descriptor = item["masks"].get(name)
                if descriptor:
                    label = {
                        "obj_full": "完整形状目标",
                        "fg_occ": "原前景遮挡",
                        "contact": "局部接触交互",
                    }[name]
                    additions["interaction"].append(
                        f"{oid} {label}区域说明：{descriptor['evidence']['notes']}"
                    )
        if (
            policy["version"] != "3.0"
            and data["background"].get("width")
            and data["background"].get("height")
        ):
            image_width, image_height = (
                frame["size"]
                if cropped
                else (data["background"]["width"], data["background"]["height"])
            )
            ratio = p["width"] * image_width / (p["height"] * image_height)
            additions["placement"].append(
                f"{oid} 完整框在原图像素坐标中的宽/高约 {ratio:.3f}；遮挡后仍保持该完整投影目标。"
            )
        ageing = item["ageing"]
        additions["material"].append(
            f"{oid}：{AGEING_LEVELS[ageing['level']]}；暴露历史提示：{ageing['exposure_history']}。"
        )
        additions["material"] += [
            f"{oid} 外观：{value}" for value in unique(directives.get("appearance", []))
        ]
        for part in materials[oid]["parts"]:
            requested = (
                part["weathering_positive"]
                if ageing["level"]
                else "保留近新基材与涂层，仅按实际暴露添加轻微表面痕迹"
            )
            if (
                ageing["level"]
                and part["base_material"] == "unknown"
                and any(
                    reference["source_kind"] == "rusted_atlas" for reference in item["references"]
                )
            ):
                requested = "保留生锈参考可见的腐蚀、污旧与积灰分布，按现场重新表现光色和反射；不推断未经核验的基材或新增腐蚀产物"
            line = f"{oid}/{part['id']}（{MATERIAL_NAMES[part['base_material']]}）：{requested}。"
            if part.get("coating"):
                line += f"涂层：{part['coating']}。"
            line += "位置：" + "；".join(part["locations"]) + "。"
            additions["material"].append(line)
            additions["negative"].append(f"{oid}/{part['id']}：{part['weathering_negative']}")
            if part["geometry_change_allowed"]:
                additions["material"].append(
                    f"{oid}/{part['id']} 仅允许这些局部结构损伤："
                    + "；".join(part["allowed_damage"])
                )
        if not ageing["allow_structural_damage"]:
            additions["identity"].append(
                f"{oid} 保持结构完整；旧化限于表面，保留接口、孔位和连接。"
            )
        for contact in item["support_contacts"]:
            point = map_point(contact["point"], frame) if cropped else contact["point"]
            text = f"{oid} 接触目标：支撑 {contact['support_id']}，点位 x={percent(point[0])}、y={percent(point[1])}。"
            if contact["support_id"] in supports:
                text += supports[contact["support_id"]]["description"]
            if support_first:
                text += (
                    "这是原承载面的初始接触搜索参考，不是必须把物体架到该像素点。"
                    "在已有授权区域按实际实体下缘形成承托；无法落稳时需重规划，不能虚构支撑。"
                )
            additions["interaction"].append(text)
        additions["interaction"] += [
            f"{oid} 接触：{value}" for value in unique(directives.get("contact", []))
        ]
        additions["interaction"] += [
            f"{oid} 遮挡：{value}" for value in unique(directives.get("occlusion", []))
        ]
    feedback = data["harmonization"].get("feedback_correction", {})
    if feedback.get("enabled"):
        additions["scene_style"].append(
            "用户已指出目标物体颜色鲜艳、细节过于清晰："
            "仅对指定物体匹配同深度钢件的积灰、低彩度残漆、弱高光和监控有限细节。"
            "不增加细密锈点、锐利白字、干净蓝边或金属亮边，不对全图去色或模糊。"
            "像素偏高指物体有效清晰度，不改变原图尺寸与视野；轮辐、阀杆、瓶帽连接和真实孔洞完整保留。"
        )
    if data["constraints"]["require_foreground_occlusion"]:
        additions["interaction"].append(
            "按原图中确实存在的前景形成局部遮挡，保留身份所需部分可辨；不为了遮挡凭空新增钢条、枝叶或其它物体。"
        )
    if data["scene"].get("gravity_image_direction"):
        additions["material"].append(
            f"流痕跟随当前摆放的重力投影提示 {data['scene']['gravity_image_direction']}，不固定沿参考图的上下方向。"
        )
    for edge in data["scene"]["occlusion_edges"]:
        additions["interaction"].append(
            f"指定交叠区域：{edge['front_id']} 在 {edge['back_id']} 前，遮挡处保持前方物体连续。"
        )
    if policy["version"] in ("2.2", "2.3"):
        for patch in data["harmonization"].get("surface_adjustments", []):
            additions["material"].append(
                f"{patch['object_id']}/{patch['id']} 表面要求：{patch['region']['evidence']['notes']}"
            )
    if data.get("prompt_notes"):
        additions["interaction"] += [
            "补充场景要求：" + value
            for value in unique(re.split(r"(?<=[。；])\s*", data["prompt_notes"]))
        ]
    sections = [
        {
            "id": section["id"],
            "title": section["title"],
            "rules": unique(section["rules"] + additions[section["id"]]),
        }
        for section in policy["sections"]
    ]
    if cropped:
        for section in sections:
            section["rules"] = [
                value.replace("原图宽", "当前输入图宽")
                .replace("原图高", "当前输入图高")
                .replace("原图像素坐标", "当前输入图像素坐标")
                .replace("原图左上角", "当前输入图左上角")
                .replace("原图像素平面", "当前输入图像素平面")
                for value in section["rules"]
            ]
    text = (
        "\n\n".join(
            f"【{index:02d} {section['title']}】\n"
            + "\n".join("- " + rule for rule in section["rules"])
            for index, section in enumerate(sections, 1)
        )
        + "\n"
    )
    return {
        "format_version": policy["version"],
        "template_sha256": template_hash,
        "reference_associations": associations,
        "background_image_index": scene_index,
        **(
            {"edit_target_image_index": scene_index + 1 if hinted else scene_index}
            if policy["version"] in ("2.3", "3.0")
            else {}
        ),
        **({"request_frame": frame} if policy["version"] in ("2.2", "2.3", "3.0") else {}),
        "sections": sections,
        "text": text,
    }
