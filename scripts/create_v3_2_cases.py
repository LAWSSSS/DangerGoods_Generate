"""Create optional user-feedback correction previews; never calls models."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    destination = root / "config/controlled/v3_2"
    targets = [destination / f"{oid}.json" for oid in ("S01", "S20")]
    if any(path.exists() for path in targets):
        raise FileExistsError("不覆盖既有V3.2配置")
    destination.mkdir(exist_ok=True)
    for oid, target in zip(("S01", "S20"), targets):
        data = json.loads((root / f"config/controlled/v3_1/{oid}.json").read_text())
        data["task_id"] = f"feedback_v3_2_{oid}"
        data["output_dir"] = "output/controlled-v3_2"
        data["harmonization"]["feedback_correction"] = {
            "enabled": True,
            "reason": "2026-10-19用户明确指出颜色鲜艳和物体像素偏高；仅对当前实体作有界校正",
            "object_ids": [oid],
            "chroma_scale": 0.78 if oid == "S01" else 0.74,
            "detail_sigma_px": 1.2,
        }
        item = data["objects"][0]
        item["placement"].update(
            priority="support_first",
            flexibility={
                "max_center_shift": 0.03,
                "scale_range": [0.85, 1.15],
                "max_rotation_delta_deg": 25,
            },
            pose_3d_notes=(
                "气瓶沿原左中部槽钢内面自然侧卧，帽在左上、圆底在右下。"
                "圆柱下腹落在槽内，不骑跨槽口或管口；只有确实交叉的原钢沿才能位于瓶腹前方，不能新画钢带。"
                if oid == "S01"
                else "阀门在原中上部宽钢面顺坡自然倾靠，以双法兰下缘和可受力阀体承托。"
                "手轮、轮辐和阀杆不承重；手轮、法兰和轭架保持同一刚体姿态，不能强行端正陈列。"
            ),
        )
        item["prompt_directives"]["appearance"].append(
            "依用户本次反馈收敛鲜明残漆和高频细纹：保留低彩度固有色、灰尘与粗糙体积，"
            "不生成鲜橙红密锈斑、干净白字、发亮蓝边或金铜亮带；有效清晰度取同深度原废钢。"
        )
        item["prompt_directives"]["contact"].append(
            "以原区域内真实承托优先于初始中心/倾角；原支撑不能变平、加宽、移动或增加。"
            "找不到现有授权内的稳定承托时标明需重规划，不用黑底或新钢条掩盖。"
        )
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        print(target)


if __name__ == "__main__":
    main()
