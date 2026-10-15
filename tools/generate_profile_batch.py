"""Generate each requested item once using the configured scene-fusion entry point."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def save(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="逐件生成背景适配的废旧物品并进行 VLM 复核"
    )
    parser.add_argument("--items", nargs="+", default=["S01", "S21", "S22"])
    parser.add_argument("--background", required=True, help="背景文件名或绝对路径")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--reference-map", type=Path, help="JSON：物品 ID 到现成生锈六视图路径"
    )
    args = parser.parse_args()
    if not 1 <= args.max_rounds <= 10:
        parser.error("max-rounds 必须在 1–10 之间")
    catalog = json.loads(
        (ROOT / "profiles/object_profiles.json").read_text(encoding="utf-8")
    )
    profiles = {item["id"]: item for item in catalog["objects"]}
    identifiers = (
        list(profiles) if args.items == ["all"] else list(dict.fromkeys(args.items))
    )
    if any(identifier not in profiles for identifier in identifiers):
        parser.error("物品必须使用 S01–S38；全部逐件生成可用 --items all")
    reference_map = {}
    if args.reference_map:
        reference_map = json.loads(args.reference_map.read_text(encoding="utf-8"))
        for identifier in identifiers:
            if (
                identifier not in reference_map
                or not Path(reference_map[identifier]).is_file()
            ):
                parser.error(f"生锈参考映射缺少有效文件：{identifier}")
    background = Path(args.background)
    if not background.is_absolute():
        background = ROOT / "input/backgrounds" / background
    if not background.is_file():
        parser.error("背景文件不存在")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    run_root = (args.output_root or ROOT / "output/fusion-v1.1" / stamp).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "run_manifest.json"
    if manifest_path.exists():
        parser.error("此目录已有运行记录，请使用新的结果目录保留原记录")
    manifest = {
        "pipeline_version": "1.1.0",
        "status": "running",
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "background": str(background),
        "requested_items": identifiers,
        "max_rounds_per_item": args.max_rounds,
        "vlm_enabled": True,
        "supplied_rust_references": reference_map,
        "items": [],
    }
    save(manifest_path, manifest)
    for index, identifier in enumerate(identifiers):
        item_root = run_root / identifier
        item_root.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-B",
            str(ROOT / "generate_aliyun.py"),
            "--objects",
            identifier,
            "--backgrounds",
            str(background),
            "--total",
            "1",
            "--with-vlm",
            "--max-rounds",
            str(args.max_rounds),
            "--seed",
            str(args.seed + index),
            "--output-root",
            str(item_root),
        ]
        if reference_map:
            command.extend(
                ["--rust-reference", str(Path(reference_map[identifier]).resolve())]
            )
        print(
            json.dumps(
                {"item": identifier, "stage": "generating_and_reviewing"},
                ensure_ascii=False,
            ),
            flush=True,
        )
        with (item_root / "run.log").open("w", encoding="utf-8") as log:
            process_env = dict(os.environ)
            process_env["PYTHONUNBUFFERED"] = "1"
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                env=process_env,
            )
        metadata_paths = sorted((item_root / "metadata").glob("*.json"))
        if metadata_paths:
            data = json.loads(metadata_paths[-1].read_text(encoding="utf-8"))
            record = {
                "id": identifier,
                "name": profiles[identifier]["name"],
                "metadata": str(metadata_paths[-1]),
                "images": data.get("generated_images", []),
                "quality_status": data.get("quality_status"),
                "vlm_used": data.get("vlm_used"),
                "vlm_passed": data.get("vlm_success"),
                "fusion_gate": data.get("fusion_gate"),
                "selected_round": data.get("selected_round"),
                "rounds": data.get("optimization_rounds"),
                "process_exit_code": result.returncode,
            }
        else:
            record = {
                "id": identifier,
                "quality_status": "generation_failed",
                "process_exit_code": result.returncode,
                "log": str(item_root / "run.log"),
            }
        manifest["items"].append(record)
        save(manifest_path, manifest)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    manifest["status"] = "complete_requires_human_visual_review"
    manifest["finished_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    save(manifest_path, manifest)


if __name__ == "__main__":
    main()
