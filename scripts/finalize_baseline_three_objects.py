#!/usr/bin/env python3
"""Verify saved outputs and compare only the same six cases from the prior test."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import statistics


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(jobs):
    result = []
    for n in (1, 2, 3):
        rows = [next(r for r in j["rounds"] if r["round"] == n) for j in jobs]
        valid = [r for r in rows if r.get("baseline_evaluation_valid")]
        audits = [r.get("independent_audit", {}) for r in rows]
        result.append(dict(round=n, evaluated_cases=len(rows),
            valid_baseline_evaluations=len(valid),
            size_acceptable=sum(r["baseline_evaluation"].get("size_judgment") == "acceptable" for r in valid),
            mean_quality=statistics.mean(r["baseline_evaluation"]["quality_score"] for r in valid) if valid else None,
            reference_match=sum(a.get("reference_match") == "match" for a in audits),
            reference_mismatch=sum(a.get("reference_match") == "mismatch" for a in audits),
            reference_uncertain=sum(a.get("reference_match") == "uncertain" for a in audits),
            stable_support=sum(a.get("support_judgment") == "stable" for a in audits),
            natural_integration=sum(a.get("integration_judgment") == "natural" for a in audits),
            independently_passed=sum(r.get("independent_passed", False) for r in rows)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--previous-run", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = args.run_dir.resolve()
    manifest = read(run / "manifest.json")
    report = read(run / "report.json")
    ledger = read(run / "requests.json")["requests"]
    assert manifest["head"] == "0fded2f6a66a64cea18aafdc193bb31fde8b9dfb"
    assert len(manifest["jobs"]) == report["completed"] == 6
    assert report["images"] == 18
    assert not any(r["status"] == "submitted" for r in ledger)
    assert sum(r["kind"] == "wan" for r in ledger) <= 18
    source_checks = {}
    for name in manifest["source_hashes"]:
        assert sha(root / name) == manifest["source_hashes"][name], name
        source_checks[name] = True
    for name in list(manifest["source_hashes"])[:5]:
        assert (root / name).read_bytes() == subprocess.check_output(
            ["git", "show", manifest["head"] + ":" + name], cwd=root), name
    for job in report["jobs"]:
        assert job["status"] == "completed"
        assert sha(Path(job["object_path"])) == job["object_hash"]
        assert sha(Path(job["background_path"])) == job["background_hash"]
        assert [r["round"] for r in job["rounds"]] == [1, 2, 3]
        for index, row in enumerate(job["rounds"]):
            assert row["status"] == "returned"
            assert sha(Path(row["image"])) == row["raw_image_sha256"]
            assert row["request_parameters"]["seed"] == job["seed"]
            assert row["request_parameters"]["n"] == 1
            assert row["generation_image_inputs"] == [job["background_path"]] + (
                [job["rounds"][index-1]["image"]] if index else [])
            assert row["size_calculation"]["target_ratio"] == row["target_ratio"]
            assert row.get("baseline_evaluation_valid"), (job["id"], row["round"])
            assert (run / "cases" / job["id"] / f"round{row['round']}_prompt.txt").read_text() == row["full_prompt"]
    keys = {(j["object_class"], j["background_label"]) for j in report["jobs"]}
    previous = read(args.previous_run.resolve() / "report.json")
    previous_jobs = [j for j in previous["jobs"] if (j["object_class"], j["background_label"]) in keys]
    assert len(previous_jobs) == 6
    comparison = dict(baseline_commit=manifest["head"], previous_commit=previous["source_commit"],
        baseline=stats(report["jobs"]), feat_same_six_cases=stats(previous_jobs),
        caveats=["Model judgments are not physical measurements.",
            "The two commits use different baseline size-evaluation criteria; those scores are not a common quality scale.",
            "Same object files, backgrounds and API seeds; placement RNG order and prompts differ by version.",
            "Two separately generated runs are not a controlled fixed-placement comparison.",
            "Independent audit never changes the generation feedback chain."])
    (run / "version_comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2)+"\n")
    final = dict(complete_cases=6, raw_images=18, source_checks=source_checks,
        input_and_output_hashes_verified=True, three_rounds_and_reference_order_verified=True,
        wan_attempts=report["wan_requests"], wan_returned=report["wan_returned"],
        wan_failed=report["wan_failed"], qwen_requests=report["qwen_requests"],
        no_pending_requests=True, independently_passed=report["independently_passed"])
    (run / "final_validation.json").write_text(json.dumps(final, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(dict(validation=final, comparison=comparison), ensure_ascii=False))


if __name__ == "__main__":
    main()
