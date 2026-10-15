import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from scripts import test_pose_scores_three_objects as batch


class BatchChecks(unittest.TestCase):
    def test_selected_scope_has_three_classes_and_primary_files(self):
        self.assertEqual(len(batch.OBJECTS), 3)
        self.assertEqual(len({x[1] for x in batch.OBJECTS}), 3)
        for _, _, name in batch.OBJECTS:
            self.assertTrue((batch.ROOT / "input/objects" / name).is_file())

    def test_submitted_round_cannot_be_reposted_and_budget_cannot_reset(self):
        with tempfile.TemporaryDirectory() as t:
            run = Path(t)
            batch.save(run / "requests.json", {"limit": 18, "requests": []})
            batch.reserve(run, "wan", "job", "r1")
            with self.assertRaises(RuntimeError):
                batch.reserve(run, "wan", "job", "r1")
            for i in range(17):
                batch.reserve(run, "wan", "job", str(i))
            with self.assertRaises(RuntimeError):
                batch.reserve(run, "wan", "another", "r1")

    def test_missing_or_uncertain_audit_cannot_pass(self):
        self.assertFalse(batch.audit_pass({}))
        self.assertFalse(batch.audit_pass({"object_present": True, "new_instance_count": True}))
        raw = dict(size_judgment="acceptable", confidence=.5, quality_score=.5, fusion_quality="ok", reason="VLM 调用失败，使用默认评估")
        self.assertFalse(batch.strict_old_evaluation(raw, True))

    def test_completed_baseline_early_acceptance_still_runs_three_rounds(self):
        output = io.BytesIO()
        Image.new("RGB", (1200, 675), "gray").save(output, format="PNG")
        image_bytes = output.getvalue()
        opening = {"center": {"x": .5, "y": .5}, "corners": [{"x": .1, "y": .1}, {"x": .9, "y": .1}, {"x": .9, "y": .9}, {"x": .1, "y": .9}]}
        evaluation = dict(size_judgment="acceptable", confidence=.9, quality_score=.9, fusion_quality="good", should_retry=False, reason="mock only", tone_score=96,size_score=97,overall_score=95,visible_ratio=.035)
        audit = {"object_present": False, "reference_match": "mismatch", "uncertainties": ["mock only"]}
        class Response:
            content = image_bytes
            def __init__(self, value): self.value = value
            def raise_for_status(self): pass
            def json(self): return self.value
        class Session:
            verify = False
            def post(self, url, *, json, **kwargs):
                if json["model"] == "wan2.7-image":
                    return Response({"output": {"choices": [{"message": {"content": [{"image": "https://mock.invalid/output.png"}]}}]}})
                text = str(json["messages"][-1]["content"][0])
                value = opening if "corners" in text else (audit if "独立" in text else evaluation)
                return Response({"choices": [{"message": {"content": __import__("json").dumps(value)}}]})
            def get(self, *args, **kwargs): return Response({})
        with tempfile.TemporaryDirectory() as t:
            run = Path(t)
            batch.save(run / "requests.json", {"limit": 18, "requests": []})
            source_hash = batch.digest(batch.ROOT / "generate_aliyun.py")
            for label in ("44", "50"):
                job = dict(id="mock_" + label, object_class="密封胶罐", seed=42,
                    reference_paths=[str(batch.ROOT / "input/objects/密封胶罐.jpg")],object_path=str(batch.ROOT / "input/objects/密封胶罐.jpg"),
                    background_path=str(batch.ROOT / "input/backgrounds" / (batch.BG_PREFIX + label + ".jpg")), background_label=label)
                with patch.object(batch.requests, "Session", Session):
                    result = batch.worker(str(run), job)
                self.assertEqual(result["status"], "completed", result.get("error"))
                self.assertEqual(len(result["rounds"]), 3)
                self.assertTrue(all(r["baseline_would_stop"] for r in result["rounds"]))
                self.assertTrue(all(r["baseline_evaluation_valid"] for r in result["rounds"]))
                self.assertFalse(any(r["independent_passed"] for r in result["rounds"]))
                self.assertEqual(len(result["rounds"][0]["generation_image_inputs"]), 2)
                self.assertEqual(len(result["rounds"][1]["generation_image_inputs"]), 2)
            self.assertEqual(source_hash, batch.digest(batch.ROOT / "generate_aliyun.py"))
            ledger = json.loads((run / "requests.json").read_text())
            self.assertEqual(sum(r["kind"] == "wan" for r in ledger["requests"]), 6)
            self.assertEqual(sum(r["kind"] == "qwen" for r in ledger["requests"]), 20)
            self.assertTrue(all(r["previous_image"] is None for r in result["rounds"]))
            self.assertEqual(result["selected_round"],1)
            self.assertEqual(result["historical_stop_round"],1)
            self.assertEqual(result["models"]["vlm"],"doubao-seed-2.1-pro")


if __name__ == "__main__":
    unittest.main()
