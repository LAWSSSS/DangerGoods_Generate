"""Build the authored per-item catalog and reusable prompts without model/API calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image, ImageOps, ImageStat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from object_catalog import render_prompt  # noqa: E402
from fusion_quality import WEATHERING  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def build(seeds_path: Path, destination: Path) -> dict:
    seeds = json.loads(seeds_path.read_text(encoding="utf-8"))
    reference_root = Path(seeds["reference_root"]).expanduser()
    if not reference_root.is_absolute():
        reference_root = (ROOT / reference_root).resolve()
    background_root = (ROOT / seeds["background_root"]).resolve()
    assumed_width = seeds["assumed_scene_width_mm"]
    projection = [0.7, 1.0]
    profiles = []
    for seed in seeds["objects"]:
        reference = reference_root / seed["reference"]
        if not reference.is_file():
            raise ValueError(f"参考图缺失：{seed['id']} {seed['reference']}")
        length = seed["dimensions"][seed["reference_axis"]]
        cross = seed["dimensions"][seed["cross"]]
        low = round(
            length * seed["visible_fraction"][0] * projection[0] / assumed_width, 6
        )
        high = round(
            min(
                0.85,
                length * seed["visible_fraction"][1] * projection[1] / assumed_width,
            ),
            6,
        )
        material = seed["material"]
        support = (
            "大件接触两处可承重的厚钢板/型钢，不由细铁丝或尖角单点托起"
            if length >= 650
            else "与附近钢件存在真实接触和前后遮挡，避免整圈黑描边"
        )
        profile = {
            "id": seed["id"],
            "name": seed["name"],
            "category": seed["category"],
            "reference_filename": seed["reference"],
            "reference_sha256": digest(reference),
            "reference_image_size": list(Image.open(reference).size),
            "original_filename": seed.get("original"),
            "reference_kind": "six_view_atlas_not_scale_measurement",
            "dimensions_mm": seed["dimensions"],
            "parts_mm": seed["parts"],
            "dimension_ranges_mm": {seed["reference_axis"]: seed["primary_range"]},
            "dimension_evidence": {
                "status": seed.get("status", seeds["dimension_status_default"]),
                "instance_measured": False,
                "prior_note": "尺寸描述/" + seed["note"] if seed.get("note") else None,
                "sources": seed.get("sources", []),
                "note": seed.get(
                    "evidence_note",
                    "工作值取自既有说明和参考外形，具体实例尺寸/型号尚未实测确认。",
                ),
            },
            "size_policy": {
                "reference_axis": seed["reference_axis"],
                "reference_length_mm": length,
                "cross_name": seed["cross"],
                "cross_mm": cross,
                "assumed_scene_width_mm": assumed_width,
                "scene_scale_status": "inherited_2_4m_assumption_not_calibrated",
                "visible_length_fraction_range": seed["visible_fraction"],
                "projection_factor_range": projection,
                "visible_ratio_range": [low, high],
                "default_visible_ratio": round((low + high) / 2, 6),
                "formula": "reference_length_mm * visible_length_fraction * projection_factor / assumed_scene_width_mm",
                "large_object_requires_partial_occlusion_or_crop": length
                >= assumed_width * 0.8,
                "warning": "可见长度比例不是可见面积比例；完整横截面不得随遮挡比例缩小。",
            },
            "material": material,
            "identity_features": seed["features"],
            "weathering_prompt": WEATHERING[material],
            "fusion_policy": {
                "appearance": "scene_matched_discarded_material",
                "clean_reference_appearance_allowed": False,
                "explicit_weathering_required": True,
                "partial_foreground_occlusion_required": True,
                "visual_review_axes": [
                    "weathering",
                    "lighting",
                    "occlusion",
                    "style",
                    "background",
                    "identity",
                ],
            },
            "exposure_hint": seed["exposure"],
            "pose_hint": seed["pose"],
            "support_hint": support,
            "forbidden_changes": seed["forbid"],
            "sampling_weight": 1.0,
        }
        profile["prompt"] = render_prompt(profile)
        profiles.append(profile)
    background_records = []
    groups = {}
    for path in sorted(background_root.iterdir()):
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            width, height = image.size
            image.thumbnail((400, 400))
            luma = ImageStat.Stat(image.convert("L"))
        group = path.stem.rsplit("_", 1)[0]
        groups[group] = groups.get(group, 0) + 1
        background_records.append(
            {
                "filename": path.name,
                "sha256": digest(path),
                "width": width,
                "height": height,
                "filename_group": group,
                "luma_mean_downsampled": round(luma.mean[0], 2),
                "luma_std_downsampled": round(luma.stddev[0], 2),
                "assumed_scene_width_mm": assumed_width,
                "calibration_status": "unmeasured_assumption",
                "suggested_pile_roi_xywh": [0.16, 0.22, 0.68, 0.56],
                "roi_status": "generic_hint_requires_per_image_confirmation_not_mask",
                "preserve_regions": ["上部时间戳", "底部机位标识", "料斗边框"],
                "lighting_policy": "按本张原图推断；不预设所有图共用固定主光方向",
            }
        )
    result = {
        "schema_version": "1.0",
        "version": "1.1.0",
        "date": "2026-10-14",
        "units": "mm",
        "reference_root": seeds["reference_root"],
        "background_root": str(background_root),
        "object_count": len(profiles),
        "background_count": len(background_records),
        "background_scale_evidence": "沿用旧程序 HOPPER_FOV_M=2.4 的假设；单图无法证实这一真实跨度。",
        "objects": profiles,
    }
    save(destination / "object_profiles.json", result)
    save(
        destination / "background_profiles.json",
        {
            "schema_version": "1.0",
            "version": "1.1.0",
            "groups": groups,
            "backgrounds": background_records,
        },
    )
    save(
        destination / "object_prompts.json",
        {
            p["id"]: {
                "name": p["name"],
                "reference_filename": p["reference_filename"],
                "prompt": p["prompt"],
            }
            for p in profiles
        },
    )
    prompt_dir = destination / "prompts"
    for profile in profiles:
        path = prompt_dir / f"{profile['id']}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(profile["prompt"] + "\n", encoding="utf-8")
    report = [
        "# 38 件物品尺寸参数与配套提示词",
        "",
        "版本：1.1.0；日期：2026-10-14；单位：mm。",
        "",
        "逐张查看了38张六视图，并与既有原物照片/尺寸说明核对。73张背景均读取了分辨率、哈希和亮度统计，人工抽查了3个文件名前缀组共12张背景。背景尚未进行真实尺度标定。",
        "",
        "下表为可直接用于合成的工作值，不是对所有实物的测量结论。未确认型号的部件只提供估计范围；有型号目录来源的项目注明条件。六视图中的铭文也可能由生成过程改写，不能一概当作实物认证。",
        "",
        "旧代码的2.4m背景水平跨度仅作为可编辑初值。可见占比由主体尺度、遮挡和投影假设共同决定，不将整件物品全部缩小；大储罐可被遮挡或局部出画。",
        "",
        "## 总表",
        "",
        "| ID | 物品 | 工作尺寸 | 参照轴可调整范围 | 可见最长边/画面宽度 |",
        "|---|---|---|---|---|",
    ]
    for profile in profiles:
        dims = "；".join(
            f"{name}{value:g}" for name, value in profile["dimensions_mm"].items()
        )
        working_range = profile["dimension_ranges_mm"][
            profile["size_policy"]["reference_axis"]
        ]
        low, high = profile["size_policy"]["visible_ratio_range"]
        report.append(
            f"| {profile['id']} | {profile['name']} | {dims} | {working_range[0]:g}–{working_range[1]:g} | {low:.1%}–{high:.1%} |"
        )
    report.extend(
        [
            "",
            "## 主要更正与证据边界",
            "",
            "- DN25螺纹小球阀、DN100双法兰截止阀、DN200双法兰球阀、DN350大型角式阀各用独立尺寸；不继续共用55cm阀门设定。",
            "- 红色卧式储气罐和青绿色压力储液罐与小集油杯分开；虽然旧文件名叫集油器，外形并不是小滤杯。",
            "- 立式/卧式工业储罐按米级工作范围设置，取消统一75cm小罐假设；姿态、可见部分和遮挡单独控制。",
            "- 轮式/螺旋/瓶式千斤顶、短/长空气避震与摩托车弹簧减震分别设置；塑料、铝、不锈钢和橡胶不统一处理成铁锈。",
            "- S25白色圆罐无标尺/容量；110×95mm仅为临时工作设定，实际尺寸应优先核实。",
            "- S35按UCP210条件引用旭精工目录尺寸：底座206、宽60、高114、轴孔50、孔距159；不代表实物已经测量。来源：[旭精工轴承座目录](https://www.asahiseiko.co.jp/english/img/cata07_stand.pdf)。",
            "",
            "## 逐件说明与可复制 Prompt",
            "",
        ]
    )
    for profile in profiles:
        report.extend(
            [
                f"### {profile['id']} — {profile['name']}",
                "",
                f"参考图：`{profile['reference_filename']}`。",
                "",
                f"尺寸证据：`{profile['dimension_evidence']['status']}`。{profile['dimension_evidence']['note']}",
                "",
                "工作尺寸："
                + "；".join(
                    f"{key} {value:g} mm"
                    for key, value in profile["dimensions_mm"].items()
                )
                + "。",
                "",
                "零件尺寸："
                + "；".join(
                    f"{key} {value:g} mm" for key, value in profile["parts_mm"].items()
                )
                + "。",
                "",
                "```text",
                profile["prompt"],
                "```",
                "",
            ]
        )
    report.extend(
        [
            "## 程序使用",
            "",
            "```bash",
            "python object_catalog.py --list",
            "python object_catalog.py --item S15",
            "python object_catalog.py --item S26 --output output/item_prompts/S26.txt",
            "# 完成配置后再明确运行真实生成任务；本次只配置，不调用模型",
            "python generate_aliyun.py --objects S15 --total 1",
            "```",
            "",
            "修改单件工作值：编辑 `profiles/object_profile_seeds.json` 后运行 `python tools/build_object_profiles.py` 重建。运行目录 `profiles/object_profiles.json`、占比范围、独立文本和本手册均为派生文件，应由种子统一更新，避免尺寸与prompt不一致。",
            "",
            "每件种子中，`dimensions` 是主体工作尺寸，`parts` 是细节工作尺寸，`primary_range` 是参照轴可调整范围，`visible_fraction` 是可见长度比例范围。`reference_axis` 指定占比计算使用哪条轴；背景标定后修改顶层 `assumed_scene_width_mm`。范围不会自行抽样改变实体型号，当前生成使用设定的名义工作值。",
            "",
            "原始图片没有重命名、移动或复制；运行只给模型当前选中物品的六视图，不混入同大类其他型号。旧原物照片暂未匹配的储罐2、黑色HCIC液压缸、小集油杯不被冒充成已有六视图。",
            "",
        ]
    )
    (destination / "物品尺寸与Prompt逐件手册.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", type=Path, default=ROOT / "profiles/object_profile_seeds.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "profiles")
    args = parser.parse_args()
    result = build(args.seeds, args.output)
    print(
        json.dumps(
            {
                "objects": result["object_count"],
                "backgrounds": result["background_count"],
                "status": "configured_without_model_calls",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
