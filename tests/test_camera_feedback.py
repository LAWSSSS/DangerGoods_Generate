"""Offline checks for explicit appearance feedback; never calls a provider."""

# ruff: noqa: E402
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusion_engine.controlled.appearance import prepare_foreground
from fusion_engine.controlled.camera_feedback import apply_user_appearance_feedback
from fusion_engine.controlled.job import ControlledJob, load_job
from fusion_engine.controlled.masks import MaskBundle, ObjectMasks
from fusion_engine.controlled.prompting import build_prompt_document
from fusion_engine.errors import ConfigError


def fixture():
    scene = np.full((140, 160, 3), [118, 115, 110], np.uint8)
    raw = scene.copy()
    cv2.circle(raw, (80, 50), 32, (25, 105, 210), 3)
    cv2.line(raw, (48, 50), (112, 50), (25, 105, 210), 1)
    cv2.line(raw, (80, 18), (80, 95), (25, 105, 210), 1)
    raw[95:125, 65:96] = [184, 60, 35]
    actual = np.any(raw != scene, axis=2)
    blank = np.zeros(actual.shape, bool)
    edit = np.ones_like(blank)
    arrays = {
        "obj_visible": actual,
        "obj_full": actual,
        "edit": edit,
        "protect": blank.copy(),
        "fg_occ": blank.copy(),
        "contact": blank.copy(),
    }
    masks = MaskBundle({"X": ObjectMasks("X", arrays)}, arrays, [], {}, [])
    settings = {
        "enabled": True,
        "reason": "用户要求降低物体鲜明度与高频细节",
        "object_ids": ["X"],
        "chroma_scale": 0.75,
        "detail_sigma_px": 1.2,
    }
    job = ControlledJob(
        {
            "harmonization": {"mode": "none", "feedback_correction": settings},
            "fusion_v3": {"surface_policy": "paired_only"},
        },
        ROOT / "work/feedback-fixture.json",
        ROOT,
    )
    return scene, raw, actual, masks, job


class CameraFeedback(unittest.TestCase):
    def test_disabled_is_exact_noop(self):
        scene, raw, actual, masks, job = fixture()
        job.data["harmonization"]["feedback_correction"]["enabled"] = False
        result, records = apply_user_appearance_feedback(raw, job, masks, actual, {"X": actual})
        self.assertTrue(np.array_equal(result, raw))
        self.assertEqual(records, [])

    def test_no_actual_mask_never_substitutes_plan(self):
        scene, raw, actual, masks, job = fixture()
        result, records = apply_user_appearance_feedback(raw, job, masks, actual, None)
        self.assertTrue(np.array_equal(result, raw))
        self.assertEqual(records[0]["status"], "SKIPPED_NO_CURRENT_VISIBLE_ESTIMATE")

    def test_colour_reduction_preserves_holes_and_single_pixel_spokes(self):
        scene, raw, actual, masks, job = fixture()
        original_mask = actual.copy()
        result, records = apply_user_appearance_feedback(raw, job, masks, actual, {"X": actual})
        self.assertTrue(np.array_equal(result[~actual], raw[~actual]))
        self.assertTrue(np.array_equal(actual, original_mask))
        self.assertTrue(np.all(np.any(result[actual] != scene[actual], axis=1)))
        self.assertLess(
            records[0]["output_chroma_p50_p90"][1],
            records[0]["input_chroma_p50_p90"][1] * 0.9,
        )
        self.assertEqual(result.shape, raw.shape)
        self.assertFalse(records[0]["alpha_or_geometry_modified"])

    def test_protect_and_original_foreground_are_unchanged(self):
        scene, raw, actual, masks, job = fixture()
        masks.arrays["protect"][95:105, 65:96] = True
        masks.arrays["fg_occ"][115:125, 65:96] = True
        excluded = masks.arrays["protect"] | masks.arrays["fg_occ"] | ~actual
        result, _ = apply_user_appearance_feedback(raw, job, masks, actual, {"X": actual})
        self.assertTrue(np.array_equal(result[excluded], raw[excluded]))

    def test_detail_softening_does_not_change_canvas_or_other_pixels(self):
        scene, raw, actual, masks, job = fixture()
        raw[95:125, 65:96:2] = [240, 240, 240]
        result, records = apply_user_appearance_feedback(raw, job, masks, actual, {"X": actual})
        before = cv2.Laplacian(cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY), cv2.CV_64F)
        after = cv2.Laplacian(cv2.cvtColor(result, cv2.COLOR_RGB2GRAY), cv2.CV_64F)
        self.assertLess(np.var(after[100:120, 70:90]), np.var(before[100:120, 70:90]))
        self.assertTrue(np.array_equal(result[~actual], raw[~actual]))
        self.assertFalse(records[0]["canvas_resized_or_cropped"])

    def test_foreground_pipeline_calls_feedback_even_with_paired_only_policy(self):
        scene, raw, actual, masks, job = fixture()
        result, target, records = prepare_foreground(scene, raw, job, masks, {"X": actual})
        self.assertTrue(np.array_equal(target, actual))
        self.assertTrue(any(r["operation"] == "user_requested_local_appearance" for r in records))
        self.assertTrue(np.any(result[actual] != raw[actual]))
        self.assertTrue(np.array_equal(result[~actual], raw[~actual]))

    def test_support_first_prompt_is_explicitly_nominal(self):
        job = load_job(ROOT / "config/controlled/v3_2/S01.json")
        document = build_prompt_document(job.data)
        placement = next(s for s in document["sections"] if s["id"] == "placement")
        text = "\n".join(placement["rules"])
        self.assertIn("初始名义框中心", text)
        self.assertIn("真实承托优先", text)
        self.assertIn("不扩大接触或编辑授权", text)
        self.assertNotIn("倾斜29度", text)

    def test_old_prompt_keeps_occlusion_and_fixed_placement(self):
        job = load_job(ROOT / "config/controlled/v3_1/S20.json")
        text = build_prompt_document(job.data)["text"]
        self.assertIn("完整框中心", text)
        for value in job.data["objects"][0]["prompt_directives"]["occlusion"]:
            self.assertIn(value.rstrip("。；;"), text)
        self.assertNotIn("用户已指出目标物体颜色鲜艳", text)

    def test_configuration_rejects_duplicate_softening_and_unknown_ids(self):
        original = load_job(ROOT / "config/controlled/v3_2/S01.json").data
        for change in ("double_blur", "unknown_id", "bad_scale_range", "production_mode"):
            data = copy.deepcopy(original)
            data["workspace"] = str(ROOT)
            if change == "double_blur":
                data["harmonization"]["blur_sigma_px"] = 0.5
            elif change == "unknown_id":
                data["harmonization"]["feedback_correction"]["object_ids"] = ["UNKNOWN"]
            elif change == "bad_scale_range":
                data["objects"][0]["placement"]["flexibility"]["scale_range"] = [1.1, 0.9]
            else:
                data["mode"] = "B"
            with self.subTest(change=change), tempfile.TemporaryDirectory(dir=ROOT / "work") as tmp:
                path = Path(tmp) / "case.json"
                path.write_text(json.dumps(data, ensure_ascii=False))
                with self.assertRaises(ConfigError):
                    load_job(path)


if __name__ == "__main__":
    unittest.main()
