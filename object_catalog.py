"""Per-reference dimensions and prompts; no API calls or inferred physical calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fusion_quality import (
    appearance_neutral_name,
    geometry_identity,
    region_prompt,
    review_requirements,
    scene_context_prompt,
)


def load_catalog(
    path: str | Path, reference_root: str | Path | None = None
) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != "1.0" or not isinstance(data.get("objects"), list):
        raise ValueError("物品目录格式不符合 1.0 约定")
    root = Path(reference_root or data["reference_root"]).expanduser().resolve()
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
        size_spec(profile, target_ratio)["size_brief"],
        "身份和几何必须保留：" + "；".join(geometry_identity(profile)) + "。",
        "材质和旧化：" + profile["weathering_prompt"],
        "姿态和支撑：" + profile["pose_hint"] + "；" + profile["support_hint"] + "。",
        "遮挡：" + profile["exposure_hint"] + "。",
        "【接触与遮挡】先选真实钢板/型钢支点，把主体大部分置于废钢缝隙深处；至少一块原有宽钢板/型钢遮住主体中段并截断两侧外轮廓，不以一条细杆轻轻跨过表面代替真正埋入。露出一段可识别旧材质及安装端之一，保持完整几何尺度。接触阴影只在真实支点附近贴合产生，不能完整摆在堆顶或出现整圈描边。允许小幅重建接触处和前景遮挡边缘。",
        "匹配原场景的明暗、色温、阴影软硬、景深、锐度、传感器噪声和压缩；接触阴影贴合真实支点，投影服从原场景光源。",
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="查看单件物品的尺寸与专属 prompt；不调用模型"
    )
    parser.add_argument("--catalog", default="profiles/object_profiles.json")
    parser.add_argument("--item", help="S01–S38 或准确参考图文件名")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--output", type=Path, help="保存指定物品的文本提示词")
    args = parser.parse_args()
    catalog = load_catalog(args.catalog)
    if args.list:
        for profile in catalog["objects"]:
            print(profile["id"], profile["name"], profile["dimensions_mm"])
    elif args.item:
        text = render_prompt(select_profile(catalog, args.item))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text)
    else:
        parser.error("请使用 --list 或 --item")


if __name__ == "__main__":
    main()
