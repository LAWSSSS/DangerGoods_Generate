"""Create opt-in V3 cases from existing v2.3 designs. Does not call any model."""

from __future__ import annotations

import copy
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    destination = root / "config/controlled/v3"
    destination.mkdir(exist_ok=True)
    for oid in ("S01", "S20"):
        data = json.loads((root / f"config/controlled/v2_3/{oid}.json").read_text(encoding="utf8"))
        data["pipeline_version"] = "3.0"
        data["prompt_version"] = "3.0"
        data["task_id"] = f"layered_v3_{oid}"
        data["output_dir"] = "output/controlled-v3"
        data["budget"] = {"max_image_calls": 3, "max_vlm_calls": 6, "max_candidates": 3}
        data["generator"]["near_opaque_min_alpha"] = 250
        data["generator"]["near_opaque_outlier_min_alpha"] = 240
        data["generator"]["near_opaque_max_outlier_fraction"] = 0.00001
        data["iteration"] = {
            "enabled": True,
            "max_rounds": 3,
            "prompt_mode": "append_only",
            "stop_on_visual_accept": False,
            "continue_on_unknown": True,
        }
        data["feature_flags"] = {
            "v3_layer_compositor": True,
            "v3_mask_refinement": True,
            "v3_interaction_layers": True,
            "v3_diagnostics": True,
            "v3_repair_router": True,
        }
        obj = data["objects"][0]
        obj["prompt_directives"]["occlusion"] = [
            "原图长斜钢条不与新主体强制交叉；不得让任何原钢材无孔穿过主体。"
            "仅在确实交叉处表现连续、明确的前后关系，不为遮挡比例添加废钢。"
        ]
        support = next(
            r
            for r in data["scene"]["support_regions"]
            if r["id"] == obj["support_contacts"][0]["support_id"]
        )
        data["fusion_v3"] = {
            "allow_proposed_interactions": True,
            "shadow_receivers": [
                {
                    "id": f"receiver_{oid}",
                    "object_id": oid,
                    "receiver": copy.deepcopy(support["mask"]),
                    "allowed": copy.deepcopy(obj["masks"]["contact"]),
                    "surface_kind": "locally_planar",
                }
            ],
            "shadow": {
                "max_darkening": 0.22,
                "min_darkening": 0.02,
                "max_chromatic_error": 0.035,
                "max_texture_log_error": 0.12,
                "min_valid_pixels": 128,
                "min_valid_fraction": 0.35,
                "min_component_pixels": 16,
            },
        }
        path = destination / f"{oid}.json"
        if path.exists():
            raise FileExistsError(f"不会覆盖已有配置：{path}")
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
        print(path)


if __name__ == "__main__":
    main()
