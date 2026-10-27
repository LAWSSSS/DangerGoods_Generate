"""Parent choice and append-only iteration checks; zero real API requests."""

# ruff: noqa: E402
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusion_engine.controlled.fusion_v3.causal import diagnose, select_parent
from fusion_engine.controlled.iteration import IterationPolicy, PromptJournal, feedback_from_record
from fusion_engine.controlled.job import ControlledJob
from fusion_engine.controlled.workflow import run
from fusion_engine.errors import AdapterError, ConfigError


def record(root, name, *, integrity="PASS", identity="PASS"):
    for suffix in ["raw", "master"]:
        Image.new("RGB", (20, 12), "gray").save(root / f"{name}_{suffix}.png")
    return {
        "id": name,
        "aligned_path": f"{name}_raw.png",
        "master_path": f"{name}_master.png",
        "master_sha256": "a" * 64,
        "raw_sha256": "b" * 64,
        "processing": {"status": "APPLIED"},
        "candidate_observation": {
            "result": {
                "objects": [
                    {
                        "id": "X",
                        "identity_status": identity,
                        "support_status": "PASS",
                        "occlusion_status": "PASS",
                        "color_status": "PASS",
                        "fix_prompt": "保留正确轮辐。",
                    }
                ]
            }
        },
        "postprocess_integrity": {
            "status": integrity,
            "parts": [
                {
                    "removed_by_mask": 10 if integrity == "FAIL" else 0,
                    "restored_to_background": 10 if integrity == "FAIL" else 0,
                }
            ],
        },
        "quality": {
            "status": "REJECTED",
            "hard_checks": {
                k: {"status": "PASS"}
                for k in ["background_lock", "original_canvas_size", "v3_saved_layers_contract"]
            },
        },
    }


class CausalIteration(unittest.TestCase):
    def test_failed_request_is_counted_and_journal_is_preserved(self):
        data = json.loads((ROOT / "config/controlled/v3_1/S20.json").read_text())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            image = root / "background.png"
            Image.new("RGB", (800, 450), "gray").save(image)
            data["background"].update({"path": str(image), "width": 800, "height": 450})
            data["background"].pop("sha256", None)
            data["objects"][0]["references"][0]["path"] = str(image)
            data["objects"][0]["references"][0].pop("sha256", None)
            data["harmonization"].update(
                {"mode": "none", "paired_anchors": [], "surface_adjustments": []}
            )
            data["fusion_v3"]["shadow_receivers"] = []
            data["output_dir"] = "output/unit"
            job = ControlledJob(data, root / "job.json", root)
            profile = SimpleNamespace(
                image_options={"max_images": 1}, public_view=lambda: {}, redact=lambda value: value
            )

            class FailingAdapter:
                capabilities = {"max_prompt_characters": 5000}

                def __init__(self, profile):
                    pass

                def preflight(self, prepared):
                    pass

                def validate_prompt(self, prepared, prompt):
                    if not prompt.startswith(prepared.prompt):
                        raise AssertionError("original prompt lost")

                def generate(self, *args):
                    raise AdapterError("synthetic transport failure")

            with (
                patch("fusion_engine.controlled.workflow.load_profile", return_value=profile),
                patch("fusion_engine.controlled.workflow.ConfiguredImageAdapter", FailingAdapter),
            ):
                with self.assertRaises(AdapterError):
                    run(job, generate=True, provider_config=root / "NOT_READ.yaml")
            manifests = list((root / "output").glob("**/manifest.json"))
            self.assertEqual(len(manifests), 1)
            report = json.loads(manifests[0].read_text())
            self.assertEqual(report["status"], "FAILED")
            self.assertEqual(report["execution"]["image_calls"], 1)
            self.assertEqual(report["execution"]["vlm_calls"], 0)
            self.assertEqual(report["iteration"]["stop_reason"], "REQUEST_OR_PIPELINE_FAILED")
            self.assertTrue((manifests[0].parent / "prompts/attempt-01.txt").is_file())

    def test_damaged_master_is_never_used_as_parent(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            r = record(root, "candidate-001", integrity="FAIL")
            chosen = select_parent(r, root, None)
            self.assertEqual(chosen["stage"], "aligned_raw")
            self.assertEqual(diagnose(r)["repair_target"], "local_reprocess")
            self.assertIn("segmentation", diagnose(r)["origins"])

    def test_new_failure_keeps_the_previous_parent(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a = record(root, "candidate-001")
            b = record(root, "candidate-002", identity="FAIL")
            old = select_parent(a, root, None)
            selected = select_parent(b, root, old)
            self.assertEqual(selected["candidate_id"], "candidate-001")
            self.assertEqual(selected["decision"], "KEEP_PREVIOUS_NO_CONFIRMED_IMPROVEMENT")

    def test_verified_repair_can_replace_parent(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a = record(root, "candidate-001", identity="FAIL")
            b = record(root, "candidate-002")
            selected = select_parent(b, root, select_parent(a, root, None))
            self.assertEqual(selected["candidate_id"], "candidate-002")
            self.assertIn("X/identity_status", selected["improved_checks"])
            self.assertFalse(selected["is_accepted_output"])

    def test_unknown_is_not_counted_as_verified_repair(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a = record(root, "candidate-001", identity="FAIL")
            b = record(root, "candidate-002", identity="UNKNOWN")
            self.assertEqual(
                select_parent(b, root, select_parent(a, root, None))["candidate_id"],
                "candidate-001",
            )

    def test_feedback_on_raw_does_not_inherit_a_postprocess_missing_part_claim(self):
        with tempfile.TemporaryDirectory() as d:
            r = record(Path(d), "candidate-001", integrity="FAIL")
            r["vlm_review"] = {
                "result": {
                    "recommendation": "refine",
                    "fix_prompt": "新增缺失的轮辐。",
                    "defects": [],
                    "issues": ["合成后轮辐消失"],
                    "uncertainties": [],
                }
            }
            f = feedback_from_record(r, source_stage="aligned_raw")
            self.assertEqual(f["generation_appendix"], "保留正确轮辐。")
            self.assertIn("新增缺失的轮辐。", "".join(f["messages"]))
            self.assertTrue(f["postprocess_only_defects_excluded_from_generation_fix"])

    def test_three_rounds_keep_exact_prefix_and_parent_identity(self):
        journal = PromptJournal("原场景与全部结构不变量。")
        journal.configure_envelope(5000, 3, 2, parent_description_budget=64)
        previous = ""
        for n in [1, 2, 3]:
            self.assertTrue(journal.current.startswith(previous))
            previous = journal.current
            journal.record_request(n)
            if n < 3:
                journal.append_feedback(
                    {
                        "generation_appendix": "只修复本轮目标。",
                        "selected_parent_description": "candidate-001的配准原始候选",
                    },
                    completed_round=n,
                    background_index=2,
                )
        self.assertEqual(len(journal.requests), 3)
        self.assertIn("candidate-001的配准原始候选", journal.current)
        self.assertLessEqual(len(journal.current), 5000)

    def test_budget_caps_rounds_and_invalid_history_is_blocked(self):
        data = {
            "candidate_geometry": {"method": "vlm_seeded_grabcut"},
            "budget": {"max_image_calls": 3, "max_vlm_calls": 4, "max_candidates": 3},
        }
        job = ControlledJob(copy.deepcopy(data), ROOT / "work/unit.json", ROOT)
        p = IterationPolicy.from_job(job).plan(job, generate=True)
        self.assertEqual(p["effective_max_rounds"], 2)
        self.assertTrue(p["budget_limited"])
        journal = PromptJournal("必须保留的原文。")
        journal.record_request(1)
        journal.current = "被替换"
        with self.assertRaises(ConfigError):
            journal.record_request(2)


if __name__ == "__main__":
    unittest.main()
