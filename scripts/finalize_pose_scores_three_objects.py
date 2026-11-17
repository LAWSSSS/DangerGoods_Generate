#!/usr/bin/env python3
"""Check the completed historical snapshot test without making model calls."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
from datetime import datetime


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = args.run_dir.resolve()
    manifest, report = read(run / "manifest.json"), read(run / "report.json")
    ledger = read(run / "requests.json")["requests"]
    assert manifest["head"] == "ec23f39c3851f1dd7f61517020d10c5129701b73"
    assert manifest["branch"] == "rollback/current-pose-scores"
    assert manifest["models"] == {"generation": "wan2.7-image", "vlm": "doubao-seed-2.1-pro"}
    assert len(manifest["jobs"]) == report["completed"] == 6
    assert report["images"] == 18
    assert not any(r["status"] == "submitted" for r in ledger)
    assert sum(r["kind"] == "wan" for r in ledger) <= 18
    for name, expected in manifest["source_hashes"].items():
        assert sha(root / name) == expected, name
        assert sha(run / "source_snapshot" / name) == expected, name
    for name in list(manifest["source_hashes"])[:5]:
        assert (root / name).read_bytes() == subprocess.check_output(
            ["git", "show", manifest["head"] + ":" + name], cwd=root), name
    for job in report["jobs"]:
        assert job["status"] == "completed"
        assert sha(Path(job["object_path"])) == job["object_hash"]
        assert sha(Path(job["background_path"])) == job["background_hash"]
        assert job["reference_paths"] == [job["object_path"]]
        assert job["reference_hashes"] == [sha(Path(job["object_path"]))]
        assert [r["round"] for r in job["rounds"]] == [1, 2, 3]
        for row in job["rounds"]:
            assert row["status"] == "returned"
            assert sha(Path(row["image"])) == row["raw_image_sha256"]
            assert row["request_parameters"]["seed"] == job["seed"]
            assert row["request_parameters"]["n"] == 1
            from PIL import Image
            with Image.open(job["background_path"]) as bg:
                ratio = bg.width / bg.height
            requested_w, requested_h = map(int, row["request_parameters"]["size"].split("*"))
            assert abs(requested_w / requested_h - ratio) < .01
            assert abs(row["output_dimensions"][0] / row["output_dimensions"][1] - ratio) < .01
            assert row["previous_image"] is None
            assert row["generation_image_inputs"] == [job["background_path"], job["object_path"]]
            assert row["generation_input_hashes"] == [job["background_hash"], job["object_hash"]]
            assert row["size_spec"]["target_ratio"] == row["target_ratio"]
            assert (run / "cases" / job["id"] / f"round{row['round']}_prompt.txt").read_text() == row["full_prompt"]
        chosen = max(job["rounds"], key=lambda r: r["baseline_mean_score"])
        assert chosen["round"] == job["selected_round"]
        assert sha(Path(job["selected_image"])) == chosen["raw_image_sha256"] == job["selected_image_sha256"]
    actual = []
    for request in ledger:
        if request["kind"] == "wan":
            model = "wan2.7-image"
        else:
            path = run / "cases" / request["job"] / f"qwen_{request['id']:04d}_request.json"
            model = read(path)["model"]
            assert model == "doubao-seed-2.1-pro", path
        actual.append({**request, "kind": "generation" if request["kind"] == "wan" else "vlm", "model": model})
    # Preserve the live driver's original ledger and expose model names explicitly.
    (run / "requests_by_model.json").write_text(json.dumps({"requests": actual}, ensure_ascii=False, indent=2)+"\n")
    (run / "report_driver_original.json").write_bytes((run / "report.json").read_bytes())
    report["vlm_requests"] = report.pop("qwen_requests", report.get("vlm_requests"))
    report["vlm_model"] = "doubao-seed-2.1-pro"
    report["models"] = manifest["models"]
    (run / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    summary = run / "summary.md"
    summary.write_text(summary.read_text().replace("Qwen请求", "VLM请求"), encoding="utf-8")
    rows = [r for j in report["jobs"] for r in j["rounds"]]
    stamp = datetime.fromisoformat(manifest["created_at"]).strftime("%Y%m%dT%H%M%SZ")
    exports = []
    for job in report["jobs"]:
        stem = f"retest_ec23f39_{job['id']}_wan2.7-image_{stamp}"
        round_paths = []
        for row in job["rounds"]:
            target = root / "output/images" / f"{stem}_round{row['round']}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert sha(target) == row["raw_image_sha256"]
            else:
                shutil.copyfile(row["image"], target)
            round_paths.append(str(target))
        chosen = next(r for r in job["rounds"] if r["round"] == job["selected_round"])
        target = root / "output/final_images" / f"{stem}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert sha(target) == job["selected_image_sha256"]
        else:
            shutil.copyfile(job["selected_image"], target)
        metadata = dict(image_id=stem, source_commit=manifest["head"], source_branch=manifest["branch"],
            test_run=str(run), background_image=job["background_path"], object_image=job["object_path"],
            object_views=job["reference_paths"], object_class=job["object_class"],
            model="wan2.7-image", vlm_model="doubao-seed-2.1-pro", seed=job["seed"],
            generated_images=[str(target)], round_images=round_paths,
            selected_round=job["selected_round"], selected_reason=job["selected_reason"],
            historical_stop_round=job["historical_stop_round"], forced_rounds=3,
            optimization_rounds=3, vlm_enabled=True, vlm_used=True,
            object_scale={"target_ratio":chosen["target_ratio"],
                "scale_min":chosen["size_spec"]["scale_min"],
                "scale_max":chosen["size_spec"]["scale_max"],
                "description":chosen["size_spec"]["size_brief"]},
            selected_scores_pass=chosen["baseline_would_stop"],
            scrap_position=job["position"], object_orientation=job["orientation"],
            optimization_history=read(run / "cases" / job["id"] / "history.json"),
            size_rounds=job["size_rounds"], judgment_history=job["judgment_history"],
            round_sharpness=job["round_sharpness"], full_prompt=chosen["full_prompt"],
            api_result=read(run / "cases" / job["id"] / f"round{chosen['round']}_api.json"),
            independent_evaluations=[r.get("independent_audit", {}) for r in job["rounds"]],
            quality_status="pending_review", independent_selected_passed=chosen.get("independent_passed",False),
            notes="Generated with wan2.7-image (HTTP), VLM: doubao-seed-2.1-pro; three full rounds; selected primary reference only.")
        md_path = root / "output/metadata" / f"{stem}.json"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n")
        exports.append(dict(case=job["id"], rounds=round_paths, final=str(target), metadata=str(md_path)))
    (run / "exports.json").write_text(json.dumps(exports, ensure_ascii=False, indent=2)+"\n")
    final = dict(complete_cases=6, raw_images=18, selected_images=6,
        source_and_snapshot_hashes_verified=True, input_and_output_hashes_verified=True,
        exactly_three_rounds_verified=True, previous_outputs_not_inputs=True,
        background_request_and_output_aspect_ratios_verified=True,
        selected_primary_references_verified=True, best_mean_selections_verified=True,
        wan_attempts=report["wan_requests"], wan_returned=report["wan_returned"], wan_failed=report["wan_failed"],
        vlm_requests=report["vlm_requests"], models=manifest["models"],
        vlm_returned=sum(r["kind"] == "vlm" and r["status"] == "returned" for r in actual),
        vlm_failed=sum(r["kind"] == "vlm" and r["status"] == "failed" for r in actual),
        valid_baseline_evaluations=sum(r.get("baseline_evaluation_valid", False) for r in rows),
        independently_passed=report["independently_passed"], no_pending_requests=True)
    (run / "final_validation.json").write_text(json.dumps(final, ensure_ascii=False, indent=2)+"\n")
    names = ["scripts/report_pose_scores_three_objects.py", "scripts/finalize_pose_scores_three_objects.py", "tests/test_pose_scores_three_objects.py"]
    for name in names:
        target = run / "report_source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    print(json.dumps(final, ensure_ascii=False))


if __name__ == "__main__":
    main()
