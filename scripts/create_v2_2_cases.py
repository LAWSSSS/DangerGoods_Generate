"""Reproduce explicitly proposed S01/S20 control geometry; no API, segmentation or physical claim."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np


def evidence(notes: str) -> dict:
    return {"source": "proposal", "confirmed": False, "confidence": 0.3, "notes": notes}


def polygon_mask(polygons: list, notes: str, exclude: list | None = None) -> dict:
    value = {
        "polygons": polygons,
        "coordinate_space": "normalized_original_background",
        "evidence": evidence(notes),
    }
    if exclude:
        value["exclude_polygons"] = exclude
    return value


def rectangle(x0: float, y0: float, x1: float, y1: float) -> list:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def ellipse(cx: float, cy: float, rx: float, ry: float, n: int = 32) -> list:
    return [
        [cx + rx * math.cos(t * 2 * math.pi / n), cy + ry * math.sin(t * 2 * math.pi / n)]
        for t in range(n)
    ]


def transform(poly: list, placement: dict) -> list:
    angle = math.radians(placement["rotation_deg"])
    matrix = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    points = np.asarray(poly) * [placement["width"] * 3840, placement["height"] * 2160]
    return np.round((points @ matrix.T) / [3840, 2160] + placement["center"], 8).tolist()


def build(oid: str, root: Path) -> dict:
    data = json.loads((root / f"config/controlled/rusted_{oid}.prompt-v2_1.json").read_text())
    data = copy.deepcopy(data)
    data["task_id"] = f"controlled_rust_v2_2_{oid}"
    data["prompt_version"] = "2.2"
    data["output_dir"] = "output/controlled-v2_2"
    item = data["objects"][0]
    p = item["placement"]
    if oid == "S20":
        # Move the proposed valve upward so the real diagonal bar crosses lower body/flanges.
        p["center"] = [0.66, 0.54]
        p["evidence"] = evidence(
            "相对2.1将中心y从0.61提至0.54，减少斜钢条穿轭架的构图风险；不是已核验三维摆放"
        )
        p["pose_3d_notes"] = (
            "阀杆朝上，整体顺时针倾斜18°，法兰在下部承托；横向双法兰与阀体连续，手轮平面按俯视角形成椭圆。"
        )
    else:
        p["pose_3d_notes"] = (
            "长轴在画面内向右上方12°延伸，瓶帽位于左端，瓶底位于右端；瓶身不折弯，近下侧两处获得承托。"
        )
    if oid == "S01":
        body = [
            [-0.49, -0.13],
            [-0.39, -0.13],
            [-0.36, -0.32],
            [-0.32, -0.46],
            [-0.27, -0.5],
            [0.40, -0.5],
            [0.47, -0.38],
            [0.5, -0.18],
            [0.5, 0.18],
            [0.47, 0.38],
            [0.40, 0.5],
            [-0.27, 0.5],
            [-0.32, 0.46],
            [-0.36, 0.32],
            [-0.39, 0.13],
            [-0.49, 0.13],
        ]
        local_polys, holes = [body], []
        desired_contacts = [[-0.23, 0.5], [0.30, 0.5]]
        label = transform(rectangle(-0.20, -0.32, 0.17, 0.06), p)
        paint = polygon_mask(
            [transform(body, p)],
            "残留红漆失光、蒙灰，按原色相保留局部红色；限制商品式鲜红与均匀白亮筒身。",
            [label],
        )
        patches = [
            {
                "id": "paint",
                "object_id": oid,
                "region": paint,
                "chroma_scale": 0.72,
                "max_lab_chroma": 29,
                "highlight_knee_l": 74,
                "highlight_slope": 0.55,
                "strength": 0.85,
            },
            {
                "id": "old_marking",
                "object_id": oid,
                "region": polygon_mask(
                    [label],
                    "CCS区域仅保留污旧字迹，白字与邻近受光表面相容，不独立发白、补字或锐化。",
                ),
                "highlight_knee_l": 65,
                "highlight_slope": 0.42,
                "strength": 0.85,
            },
        ]
        anchors = [
            (
                "left_dust",
                rectangle(0.448, 0.578, 0.478, 0.602),
                "气瓶左上邻近的覆灰钢面，避开绿色篷布与文字",
            ),
            ("right_dust", rectangle(0.748, 0.645, 0.765, 0.670), "气瓶右侧同一区域的灰褐废钢面"),
            ("lower_dust", rectangle(0.530, 0.823, 0.553, 0.843), "气瓶下方覆灰钢板的中间调表面"),
        ]
        forbidden = rectangle(0.775, 0.595, 0.799, 0.795)
        note = (
            "右侧可见螺纹杆保留其原有位置；气瓶不能伸入此二维保护带。投影禁入不等于完整三维碰撞体。"
        )
        extra = {
            "identity": [
                "瓶帽在左端、瓶底在右端，完整长轴与指定倾角一致；不能用更短更粗的气瓶替换。"
            ],
            "appearance": [
                "红漆仅作不连续的暗灰红/灰粉残漆；锈蚀、旧漆和覆灰各有不同粗糙度，受光上缘也不能形成独立亮带。",
                "CCS一带字迹为污旧痕迹，局部断裂、蒙灰、低对比；不把白字做成全图最清晰最亮的主体标识。",
            ],
            "contact": [
                "瓶身下侧在两个分开的建议接触区与废钢相接；两点之间可有局部间隙，避免整条下缘等宽发黑。承托应依靠可辨钢件，不能凭空添加支架。"
            ],
            "occlusion": [
                "原图斜长钢条连续位于瓶身前方；瓶身被挡处保持相同轴线和直径。钢条孔洞只显示实际视线后的内容，不能让瓶体从钢条实体表面穿出。"
            ],
        }
    else:
        wheel = ellipse(0, -0.34, 0.49, 0.15)
        stem = rectangle(-0.045, -0.22, 0.045, 0.04)
        yoke = [[-0.28, -0.18], [0.28, -0.18], [0.20, 0.13], [-0.20, 0.13]]
        body = ellipse(0, 0.25, 0.29, 0.23)
        left, right = ellipse(-0.32, 0.29, 0.17, 0.21), ellipse(0.32, 0.29, 0.17, 0.21)
        local_polys = [wheel, stem, yoke, body, left, right]
        holes = []
        for start in (18, 108, 198, 288):
            angles = np.radians(np.linspace(start, start + 53, 8))
            arc = [[0.38 * math.cos(t), -0.34 + 0.104 * math.sin(t)] for t in angles]
            arc += [[0.14 * math.cos(t), -0.34 + 0.040 * math.sin(t)] for t in reversed(angles)]
            holes.append(arc)
        holes += [rectangle(-0.18, -0.10, -0.075, 0.025), rectangle(0.075, -0.10, 0.18, 0.025)]
        desired_contacts = [[-0.32, 0.50], [0.32, 0.50]]
        wheel_world = transform(wheel, p)
        patches = [
            {
                "id": "wheel",
                "object_id": oid,
                "region": polygon_mask(
                    [wheel_world],
                    "手轮保留低纯度灰蓝残漆；外圈、轮辐和轮毂同样失光积灰，不保留干净亮蓝色圈或白色锐边。",
                ),
                "chroma_scale": 0.65,
                "max_lab_chroma": 18,
                "highlight_knee_l": 68,
                "highlight_slope": 0.6,
                "strength": 0.9,
            },
            {
                "id": "body",
                "object_id": oid,
                "region": polygon_mask(
                    [transform(poly, p) for poly in local_polys[1:]],
                    "阀体、阀盖、法兰和轭架呈参考中可见的暗褐腐蚀及灰尘层；压制棚拍金铜色高光，基材仍未知。",
                    [wheel_world],
                ),
                "chroma_scale": 0.84,
                "max_lab_chroma": 20,
                "highlight_knee_l": 72,
                "highlight_slope": 0.6,
                "strength": 0.85,
            },
        ]
        anchors = [
            ("left_neutral", rectangle(0.584, 0.497, 0.605, 0.520), "阀门左侧邻近的灰褐粗糙钢面"),
            (
                "right_neutral",
                rectangle(0.723, 0.580, 0.739, 0.595),
                "阀门右侧邻近覆灰钢面的中间调",
            ),
            (
                "lower_neutral",
                rectangle(0.708, 0.662, 0.727, 0.682),
                "阀门下方的覆灰钢件表面，避开深黑孔洞",
            ),
        ]
        forbidden = rectangle(0.773, 0.560, 0.801, 0.770)
        note = (
            "右侧原有管件与螺纹杆所在区域不允许阀门伸入；禁入区仅为二维构图约束，未重建三维占据体。"
        )
        extra = {
            "identity": [
                "手轮轮辐之间与轭架两侧保持真实开口；钢条不能穿过轮辐、阀杆或轭架实体，阀杆与阀盖连接不断裂。"
            ],
            "appearance": [
                "轮缘、轮辐、中心轮毂只保留局部灰蓝残漆与灰尘，不出现高亮蓝色完整圆环；亮边强度不超过相同局部照明下的邻近表面。",
                "阀体底色按既有锈蚀与污垢表现为灰褐/污褐层次，压制淡金、铜黄棚拍反光；法兰孔、凸字与螺纹按监控有效清晰度表现。",
            ],
            "contact": [
                "阀门下部两侧法兰在建议支点附近获得承托；底部空隙可以局部存在，但不能整件架空，不能以手轮、阀杆或前景斜钢条悬挂阀门。"
            ],
            "occlusion": [
                "新落位让原图斜长钢条主要经过阀体/法兰下部，手轮和阀杆位于钢条上方的可见空间；逐段保持钢条实体在前。不能为露出DN100标识把阀体抬到钢条前面。"
            ],
        }
    full_polys = [transform(poly, p) for poly in local_polys]
    full_holes = [transform(poly, p) for poly in holes]
    # A supplied design proxy is deliberately marked proposal, not a segmentation observation.
    item["masks"]["obj_full"] = polygon_mask(
        full_polys,
        f"{oid}按参考结构和目标姿态构造的完整投影代理；不是实际生成分割，不是CAD或实测轮廓。",
        full_holes,
    )
    foreground = [[0.765, 0.463], [0.778, 0.486], [0.478, 0.842], [0.465, 0.819]]
    slot = [[0.601, 0.672], [0.606, 0.678], [0.620, 0.663], [0.616, 0.655]]
    item["masks"]["fg_occ"] = polygon_mask(
        [foreground],
        "原场景斜长钢条的建议轮廓，保持从右上到左下连续；镂空槽不作为实体遮挡。边缘为人工式视觉估计，待用户审核。",
        [slot],
    )
    item["support_contacts"] = []
    regions, contact_polys = [], []
    for index, local in enumerate(desired_contacts, 1):
        point = transform([local], p)[0]
        sid = f"support_{oid}_{index}"
        x, y = point
        region = rectangle(x - 0.014, y - 0.002, x + 0.014, y + 0.014)
        contact_polys.append(rectangle(x - 0.015, y - 0.005, x + 0.015, y + 0.014))
        regions.append(
            {
                "id": sid,
                "mask": polygon_mask(
                    [region], "依据画面与目标下缘提出的承托区；没有载荷、深度或三维稳定性证据。"
                ),
                "load_bearing": "unknown",
                "description": f"建议承托区{index}，位于指定支点附近的原有废钢表面；先形成局部相接，再产生窄缝接触暗部，不能画成整圈黑底。",
            }
        )
        item["support_contacts"].append(
            {
                "support_id": sid,
                "point": point,
                "evidence": evidence("投影代理下缘的建议接触点；不是实际生成结果的已确认接触。"),
            }
        )
    item["masks"]["contact"] = polygon_mask(
        contact_polys, "只允许在下缘两处建议接触区域重建接触与局部阴影；不允许整片重绘或黑影托底。"
    )
    envelope_placement = dict(p)
    envelope_placement["width"] *= 1.10 if oid == "S01" else 1.40
    envelope_placement["height"] *= 1.45 if oid == "S01" else 1.15
    envelope = polygon_mask(
        [transform(rectangle(-0.5, -0.5, 0.5, 0.5), envelope_placement), *contact_polys],
        "独立允许编辑包络，容纳少量轮廓偏差与接触交互；不是物体分割。最终处理采用候选估计可见轮廓，不能按投影代理裁掉主体。",
    )
    item["spatial_constraints"] = {
        "edit_envelope": envelope,
        "forbidden": polygon_mask([forbidden], note),
        "minimum_contact_count": 2,
        "occluded_fraction_range": [0.02, 0.45],
    }
    for key, values in extra.items():
        item["prompt_directives"][key] = values
    data["scene"]["support_regions"] = regions
    data["constraints"].update({"background_protection": "strict", "enforce_spatial_plan": True})
    data["generator"].update(
        {
            "route": "guided_generation_then_restore",
            "context_crop": True,
            "context_padding_ratio": 0.25,
            "alpha_policy": "near_opaque_unassociated_rgb",
            "near_opaque_min_alpha": 251,
        }
    )
    data["budget"] = {"max_image_calls": 1, "max_vlm_calls": 0, "max_candidates": 1}
    data["harmonization"] = {
        "mode": "paired_local",
        "paired_anchors": [
            {
                "id": aid,
                "mask": polygon_mask(
                    [poly], description + "；建议配对区，使用同像素前后差分，不把其颜色当物体颜色。"
                ),
                "description": description,
            }
            for aid, poly, description in anchors
        ],
        "allow_proposed_anchors": True,
        "min_anchor_pixels": 128,
        "max_exposure_stops": 0.4,
        "max_chromatic_stops": 0.18,
        "strength": 0.75,
        "max_pair_residual": 0.06,
        "min_pair_inlier_fraction": 0.6,
        "surface_adjustments": patches,
        "blur_sigma_px": 0.35,
        "noise_std": 0,
        "feather_radius_px": 4,
    }
    data["candidate_geometry"] = {
        "method": "grabcut_difference",
        "seed": 42,
        "iterations": 3,
        "max_side": 1000,
        "difference_floor": 12,
        "max_background_difference": 40,
        "min_visible_iou": 0.4,
        "max_escape_fraction": 0.03,
        "contact_tolerance_px": 14,
        "reject_on_risk": True,
        "min_seed_density": 0.03,
        "min_area_ratio": 0.2,
        "max_area_ratio": 3,
        "max_component_count": 8,
    }
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="必须使用新目录；生成配置不启动模型"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = args.output.resolve()
    if not destination.is_relative_to(root) or not any(
        destination.is_relative_to(root / name) for name in ("work", "config/controlled")
    ):
        raise SystemExit("输出目录必须位于仓库 work/ 或 config/controlled/ 内")
    destination.mkdir(parents=True, exist_ok=False)
    for oid in ("S01", "S20"):
        data = build(oid, root)
        data["workspace"] = str(root)
        (destination / f"{oid}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        )
    print(json.dumps({"output": str(destination), "model_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
