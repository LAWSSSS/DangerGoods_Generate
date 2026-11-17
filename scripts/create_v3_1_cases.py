"""Create opt-in structure-preserving cases; never starts model requests."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    destination = root / "config/controlled/v3_1"
    destination.mkdir(exist_ok=True)
    for oid in ("S01", "S20"):
        data = json.loads((root / f"config/controlled/v3/{oid}.json").read_text())
        data["task_id"] = f"structure_v3_1_{oid}"
        data["output_dir"] = "output/controlled-v3_1"
        data["feature_flags"].update(
            {"v3_structure_preservation": True, "v3_causal_iteration": True}
        )
        data["fusion_v3"].update(
            {"surface_policy": "paired_only", "max_structure_loss_fraction": 0.02}
        )
        data["harmonization"]["blur_sigma_px"] = 0
        data["generator"]["layout_hint"] = "none"
        obj = data["objects"][0]
        obj["prompt_directives"]["appearance"].append(
            "保留参考残漆与锈层的固有色及层次；只按现场受光重表现，不统一压成灰粉色。"
            "优先保住完整封闭轮廓、细连接和真实孔洞，修旧不等于删除结构。"
        )
        target = destination / f"{oid}.json"
        if target.exists():
            raise FileExistsError(f"不覆盖现有配置：{target}")
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
        print(target)


if __name__ == "__main__":
    main()
