"""Synthetic ground-truth checks for visible structures; no model calls."""

# ruff: noqa: E402
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusion_engine.controlled.candidate_geometry import analyze_candidate
from fusion_engine.controlled.fusion_v3.compositor import Layers, binary_partition, compose_layers
from fusion_engine.controlled.fusion_v3.interactions import contact_layer, extract_shadow
from fusion_engine.controlled.fusion_v3.shape_evidence import (
    candidate_evidence,
    structure_retention,
)
from fusion_engine.controlled.job import ControlledJob
from fusion_engine.controlled.masks import MaskBundle, ObjectMasks
from fusion_engine.controlled.photometry import adjust_surfaces
from fusion_engine.errors import ConfigError


def fixture():
    scene = np.full((200, 200, 3), [130, 120, 110], np.uint8)
    raw = scene.copy()
    colour = (35, 95, 165)
    cv2.circle(raw, (100, 70), 40, colour, 6)
    cv2.line(raw, (60, 70), (140, 70), colour, 5)
    cv2.line(raw, (100, 30), (100, 125), colour, 5)
    raw[125:181, 85:116] = [155, 65, 45]
    actual = np.any(raw != scene, axis=2)
    hole = [[0.34, 0.20], [0.66, 0.20], [0.66, 0.5], [0.34, 0.5]]
    guide = {
        "id": "X",
        "polygon": [
            [0.27, 0.12],
            [0.73, 0.12],
            [0.73, 0.58],
            [0.59, 0.58],
            [0.59, 0.92],
            [0.41, 0.92],
            [0.41, 0.58],
            [0.27, 0.58],
        ],
        "holes": [hole],
        "foreground_points": [[0.5, 0.15], [0.3, 0.35], [0.5, 0.35], [0.5, 0.78]],
        "parts": [
            {
                "id": "wheel",
                "polygon": [[0.27, 0.12], [0.73, 0.12], [0.73, 0.58], [0.27, 0.58]],
                "holes": [hole],
                "foreground_points": [[0.5, 0.15], [0.3, 0.35], [0.5, 0.35]],
            },
            {
                "id": "body",
                "polygon": [[0.4, 0.55], [0.6, 0.55], [0.6, 0.92], [0.4, 0.92]],
                "holes": [],
                "foreground_points": [[0.5, 0.78], [0.5, 0.87]],
            },
        ],
    }
    edit = np.zeros(actual.shape, bool)
    edit[15:190, 40:160] = True
    blank = np.zeros(actual.shape, bool)
    arrays = {
        "obj_full": actual,
        "obj_visible": actual,
        "edit": edit,
        "protect": ~edit,
        "contact": edit.copy(),
        "fg_occ": blank,
    }
    masks = MaskBundle({"X": ObjectMasks("X", arrays)}, arrays, [], {}, [])
    data = {
        "objects": [{"id": "X", "support_contacts": [], "spatial_constraints": {}}],
        "feature_flags": {"v3_structure_preservation": True, "v3_mask_refinement": True},
        "candidate_geometry": {
            "method": "vlm_seeded_grabcut",
            "seed": 42,
            "iterations": 3,
            "reject_on_risk": True,
        },
    }
    return scene, raw, actual, guide, masks, ControlledJob(data, ROOT / "work/fixture.json", ROOT)


class StructurePreservation(unittest.TestCase):
    def test_shadow_cannot_modify_protected_or_unauthorized_pixels(self):
        scene, raw, actual, guide, masks, job = fixture()
        core, edge = binary_partition(actual)
        blank = np.zeros(actual.shape, bool)
        cast = blank.copy()
        cast[0, 0] = True
        layers = Layers(
            actual, core, edge, blank, blank, cast, masks.arrays["edit"], masks.arrays["protect"]
        )
        with self.assertRaises(ConfigError):
            compose_layers(scene, raw, layers)

    def test_wrong_hole_is_soft_and_visible_spokes_survive(self):
        scene, raw, actual, guide, masks, job = fixture()
        evidence = candidate_evidence(scene, raw, guide)
        self.assertTrue(evidence.foreground[70, 100])
        self.assertFalse(evidence.background[70, 100])
        self.assertGreater(evidence.report["foreground_hole_conflicts"], 0)
        result = analyze_candidate(scene, raw, job, masks, {"result": {"objects": [guide]}})
        visible = result.visible["X"]
        self.assertTrue(visible[70, 100])
        self.assertTrue(visible[70, 85])
        # The true empty quadrant remains background even though a false broad
        # hole covers both this gap and the real crossing spokes.
        self.assertFalse(visible[54, 84])
        self.assertGreater(float(visible[actual].mean()), 0.98)

    def test_closed_rim_and_actual_holes_survive_compositing(self):
        scene, raw, actual, guide, masks, job = fixture()
        result = analyze_candidate(scene, raw, job, masks, {"result": {"objects": [guide]}})
        visible = result.visible["X"]
        core, edge = binary_partition(visible)
        blank = np.zeros(visible.shape, bool)
        layers = Layers(
            visible, core, edge, blank, blank, blank, masks.arrays["edit"], masks.arrays["protect"]
        )
        output, report = compose_layers(scene, raw, layers)
        check = structure_retention(scene, raw, output, result.visible, result.witnesses)
        self.assertEqual(check["status"], "PASS")
        self.assertTrue(np.array_equal(output[actual], raw[actual]))
        self.assertTrue(np.array_equal(output[54, 84], scene[54, 84]))
        self.assertEqual(report["changed_outside_authorized"], 0)
        self.assertEqual(output.shape, scene.shape)

    def test_deleting_a_candidate_spoke_is_detected(self):
        scene, raw, actual, guide, masks, job = fixture()
        evidence = candidate_evidence(scene, raw, guide)
        broken = actual.copy()
        broken[67:74, 80:97] = False
        output = scene.copy()
        output[broken] = raw[broken]
        report = structure_retention(scene, raw, output, {"X": broken}, {"X": evidence})
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any(p["removed_by_mask"] for p in report["parts"]))

    def test_new_surface_policy_does_not_repeat_fixed_grey_grading(self):
        scene, raw, actual, guide, masks, job = fixture()
        job.data["fusion_v3"] = {"surface_policy": "paired_only"}
        job.data["harmonization"] = {"surface_adjustments": [{"unusable_legacy_patch": True}]}
        output, report = adjust_surfaces(raw, job, masks, actual)
        self.assertTrue(np.array_equal(output, raw))
        self.assertEqual(report[0]["status"], "SKIPPED_FIXED_SURFACE_GRADING")

    def test_unknown_interaction_is_not_an_invented_shadow(self):
        scene, raw, actual, guide, masks, job = fixture()
        empty = np.zeros(actual.shape, bool)
        result = extract_shadow(
            scene, raw, empty, empty, actual, {}, registration_trusted=True, drift_trusted=False
        )
        self.assertEqual(result.report["status"], "UNKNOWN")
        self.assertFalse(result.receiver.any())
        self.assertTrue(np.all(result.transmission == 1))

    def test_one_unknown_contact_does_not_delete_independent_known_contact(self):
        scene, raw, actual, guide, masks, job = fixture()
        support = np.zeros(actual.shape, bool)
        support[181:190, 80:122] = True
        # A second receiver completely covered by the body cannot be measured.
        buried = np.zeros(actual.shape, bool)
        buried[145:155, 90:110] = True

        def desc(region):
            # Tiny synthetic rectangles have exact coordinates, not scene claims.
            yy, xx = np.where(region)
            return {
                "polygons": [
                    [
                        [float(xx.min() / 200), float(yy.min() / 200)],
                        [float(xx.max() / 200), float(yy.min() / 200)],
                        [float(xx.max() / 200), float(yy.max() / 200)],
                        [float(xx.min() / 200), float(yy.max() / 200)],
                    ]
                ],
                "evidence": {
                    "source": "synthetic_fixture",
                    "confirmed": False,
                    "notes": "unit fixture",
                },
            }

        job.data["scene"] = {
            "support_regions": [
                {"id": "floor", "mask": desc(support)},
                {"id": "buried", "mask": desc(buried)},
            ]
        }
        job.data["objects"][0].update(
            {
                "support_contacts": [
                    {"point": [0.5, 0.9], "support_id": "floor"},
                    {"point": [0.5, 0.75], "support_id": "buried"},
                ],
                "spatial_constraints": {"minimum_contact_count": 2},
            }
        )
        job.data["evaluation"] = {"contact_gap_max_px": 4}
        job.data["contact_rendering"] = {
            "enabled": True,
            "search_radius_px": 12,
            "strength": 0.12,
            "sigma_px": 2,
        }
        result = contact_layer(job, masks, {"X": actual})
        self.assertTrue(result.receiver.any())
        self.assertTrue(
            all(c["overall_support_status"] == "UNKNOWN" for c in result.report["contacts"])
        )
        self.assertFalse(np.any(result.receiver & masks.arrays["protect"]))


if __name__ == "__main__":
    unittest.main()
