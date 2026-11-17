"""Build full-frame trials from independent scene support proposals, never calibrated geometry."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
from create_v2_2_cases import build as baseline
from create_v2_2_cases import evidence, polygon_mask, rectangle, transform


def remap(descriptor: dict, before: dict, after: dict) -> dict:
    value = copy.deepcopy(descriptor)
    angle = math.radians(before["rotation_deg"])
    matrix = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    for field in ("polygons", "exclude_polygons"):
        if field in value:
            local = [
                ((np.asarray(poly) - before["center"]) * [3840, 2160])
                @ matrix
                / [before["width"] * 3840, before["height"] * 2160]
                for poly in value[field]
            ]
            value[field] = [transform(poly, after) for poly in local]
    value["evidence"] = evidence(
        value["evidence"]["notes"] + "；2.3仅按新目标姿态重投影，仍非生成分割真值。"
    )
    return value


def build(oid: str, root: Path) -> dict:
    data = baseline(oid, root)
    data["task_id"] = f"fullframe_grounded_v2_3_{oid}"
    data["prompt_version"] = "2.3"
    data["workspace"] = "../../.."
    data["output_dir"] = "output/controlled-v2_3"
    item = data["objects"][0]
    if oid == "S01":
        for key in ("critical_parts", "positive"):
            item["identity"][key] = [
                value.replace("细长红色圆柱", "细长、残留污旧红漆的圆柱")
                for value in item["identity"][key]
            ]
    if oid == "S20":
        item["identity"]["critical_parts"] = [
            "旧蓝漆手轮及可见轮辐",
            "升降阀杆与开放轭架",
            "一体阀体和两侧法兰",
        ]
        item["identity"]["positive"] = list(item["identity"]["critical_parts"])
    old = copy.deepcopy(item["placement"])
    p = item["placement"]
    p.update(
        {
            "center": [0.275, 0.412] if oid == "S01" else [0.626, 0.322],
            "width": 0.225 if oid == "S01" else 0.07,
            "height": 0.0625 if oid == "S01" else 0.15,
            "rotation_deg": 29 if oid == "S01" else -12,
        }
    )
    p["evidence"] = evidence(
        "先观察原背景独立钢面后选择落位。VLM最初建议存在角度/位置矛盾，现为原图视觉核对后的保守预览建议，无真实比例或三维承载标定。"
    )
    p["pose_3d_notes"] = (
        "侧卧在左中部原有宽槽钢的灰褐内表面，长轴从左上向右下倾斜29度，保护帽在左上、瓶底在右下。筒体紧贴槽内表面，主体位于槽内，不跨在槽口上方悬空。"
        if oid == "S01"
        else "落在中上部原有宽灰褐钢板的可见上表面，阀杆投影向上并稍向左倾12度，双法兰下缘紧贴原钢板；相机俯视能看到手轮圆面、轮毂与阀盖上表面，不能复制参考平视展示面。允许轻微倾靠以贴合钢板，不得以手轮悬挂。"
    )
    item["masks"]["obj_full"] = remap(item["masks"]["obj_full"], old, p)
    item["masks"]["fg_occ"] = {
        "empty": True,
        "evidence": evidence(
            "新落位不强制让原斜钢条从物体中部经过；没有独立确认的前景分割，明确不伪造精确遮挡Mask。"
        ),
    }
    item["support_contacts"] = []
    support = (
        [[0.151, 0.295], [0.198, 0.270], [0.405, 0.514], [0.347, 0.569]]
        if oid == "S01"
        else [
            [0.55, 0.273],
            [0.64, 0.246],
            [0.67, 0.376],
            [0.656, 0.452],
            [0.603, 0.444],
            [0.571, 0.375],
        ]
    )
    description = (
        "原图左中部斜向宽槽钢内的覆灰钢面，既有边缘从左上向右下延伸。保留槽钢的侧壁和内面，气瓶侧卧落在内面并与近下侧灰褐钢面相接。"
        if oid == "S01"
        else "原图中上部宽灰褐钢板的可见上表面，位于大圆筒右上方、长斜钢条左上方；两处法兰下缘落在此原有宽板面上，板边和原有灰尘纹理在接触旁可辨，不能站在右侧暗区或虚构支架上。"
    )
    sid = f"scene_receiver_{oid}"
    data["scene"]["support_regions"] = [
        {
            "id": sid,
            "mask": polygon_mask(
                [support],
                "在原图上独立选取的可见承载候选面，不由物体OBB或预定底边扩张产生；面内高度与真实载荷未知。",
            ),
            "load_bearing": "unknown",
            "description": description,
        }
    ]
    contact_polys = []
    for local in [[-0.22, 0.5], [0.28, 0.5]] if oid == "S01" else [[-0.32, 0.5], [0.32, 0.5]]:
        point = transform([local], p)[0]
        item["support_contacts"].append(
            {
                "support_id": sid,
                "point": point,
                "evidence": evidence(
                    "落位代理下缘与独立承载面附近的建议接触，实际生成边界须重新检测，不能当成已接触。"
                ),
            }
        )
        x, y = point
        contact_polys.append(rectangle(x - 0.013, y - 0.012, x + 0.013, y + 0.014))
    item["masks"]["contact"] = polygon_mask(
        contact_polys, "接触附近窄范围交互，仅承托面与实体边界相接后允许补有限接触遮蔽。"
    )
    expanded = {**p, "width": p["width"] * 1.35, "height": p["height"] * 1.65}
    corners = np.asarray(transform(rectangle(-0.5, -0.5, 0.5, 0.5), expanded))
    bounds = rectangle(*corners.min(axis=0), *corners.max(axis=0))
    item["spatial_constraints"] = {
        "edit_envelope": polygon_mask(
            [bounds, *contact_polys],
            "明确局部编辑包络，含实体轮廓小幅变化和接触；不裁切模型输入画幅，不冒充对象分割。",
        ),
        "minimum_contact_count": 2,
        "occluded_fraction_range": [0, 0.35],
    }
    for patch in data["harmonization"]["surface_adjustments"]:
        patch["region"] = remap(patch["region"], old, p)
        if patch["id"] in ("paint", "body"):
            patch["fill_unassigned_visible"] = True
        if patch["id"] == "paint":
            patch.update(
                {
                    "chroma_scale": 0.62,
                    "max_lab_chroma": 22,
                    "highlight_knee_l": 58,
                    "highlight_slope": 0.6,
                }
            )
        elif patch["id"] == "old_marking":
            patch.update({"highlight_knee_l": 48, "highlight_slope": 0.28})
        elif patch["id"] == "wheel":
            patch.update(
                {
                    "chroma_scale": 0.55,
                    "max_lab_chroma": 14,
                    "highlight_knee_l": 58,
                    "highlight_slope": 0.5,
                }
            )
        elif patch["id"] == "body":
            patch.update(
                {
                    "chroma_scale": 0.72,
                    "max_lab_chroma": 16,
                    "highlight_knee_l": 58,
                    "highlight_slope": 0.48,
                }
            )
    anchors = (
        [
            ("near_left", rectangle(0.09, 0.42, 0.11, 0.46), "左侧槽钢之外未编辑区域"),
            ("near_right", rectangle(0.45, 0.40, 0.47, 0.43), "右侧邻近原有灰褐钢面"),
            ("below", rectangle(0.29, 0.64, 0.31, 0.67), "下方未编辑原钢面"),
        ]
        if oid == "S01"
        else [
            ("left", rectangle(0.52, 0.34, 0.54, 0.36), "左侧未编辑灰钢面"),
            ("right", rectangle(0.72, 0.33, 0.74, 0.35), "右侧未编辑覆灰钢面"),
            ("above", rectangle(0.65, 0.12, 0.67, 0.14), "上方原灰钢面"),
        ]
    )
    data["harmonization"]["paired_anchors"] = [
        {
            "id": aid,
            "mask": polygon_mask([poly], description + "；同位置前后配对，不作为物体本色。"),
            "description": description,
        }
        for aid, poly, description in anchors
    ]
    data["harmonization"].update(
        {
            "anchor_failure_policy": "skip_unreliable",
            "min_reliable_anchors": 1,
            "feather_radius_px": 2,
            "blur_sigma_px": 0.45,
        }
    )
    item["prompt_directives"]["identity"] = (
        ["保护帽在左上端、圆底在右下端；保持连续细长筒体与原参考帽孔。"]
        if oid == "S01"
        else [
            "手轮、阀杆、开放轭架、阀盖、阀体、双法兰依次真实连接；轭架中的空隙不能被填成短实心脖子。"
        ]
    )
    item["prompt_directives"]["contact"] = [
        description,
        "相机视角下实体下缘与承载面有明确相接或局部掩埋；各接触点旁保留原钢面纹理，不画一整圈黑底来掩盖空间间隙。",
    ]
    item["prompt_directives"]["occlusion"] = [
        "原图长斜钢条远离新的主体落位，不让它穿过瓶壳、阀体、手轮或轭架；仅在实际交叉处表现合理前后关系，不为了遮挡比例添加废钢。"
    ]
    data.pop("prompt_notes", None)
    data["constraints"].update(
        {"require_foreground_occlusion": False, "enforce_spatial_plan": True}
    )
    data["generator"].update(
        {
            "context_crop": False,
            "preserve_full_frame": True,
            "refinement_source": "previous_candidate",
            "bbox_padding_px": 12,
            "layout_hint": "procedural_proxy",
        }
    )
    data["candidate_geometry"]["method"] = "vlm_seeded_grabcut"
    data["candidate_geometry"]["analysis_context"] = True
    data["registration"] = {
        "max_side": 1800,
        "min_inliers": 35,
        "max_rotation_deg": 2,
        "max_scale_log": 0.04,
        "max_translation_ratio": 0.025,
        "max_reprojection_px": 4,
    }
    data["contact_rendering"] = {
        "enabled": True,
        "search_radius_px": 40,
        "sigma_px": 3,
        "strength": 0.12,
    }
    data["budget"] = {"max_image_calls": 2, "max_vlm_calls": 4, "max_candidates": 2}
    return data


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    directory = root / "config/controlled/v2_3"
    directory.mkdir(parents=True, exist_ok=False)
    for oid in ("S01", "S20"):
        (directory / f"{oid}.json").write_text(
            json.dumps(build(oid, root), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print("Created two full-frame configurations; model calls: 0")
