"""Offline regression checks. No provider configuration or real API calls."""

# ruff: noqa: E402 -- Bootstrap the uninstalled src package for direct unittest discovery.

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusion_engine.controlled.appearance import harmonize
from fusion_engine.controlled.framing import request_frame
from fusion_engine.controlled.grounding import contact_diagnostics, render_contact
from fusion_engine.controlled.job import ControlledJob
from fusion_engine.controlled.masks import MaskBundle, ObjectMasks
from fusion_engine.controlled.normalization import normalize_candidate
from fusion_engine.controlled.observation import (
    bind_observation,
    map_observation_to_original,
    normalize_observation,
)
from fusion_engine.errors import AdapterError, ConfigError
from fusion_engine.providers.vlm import VLMClient


class GroundingRegression(unittest.TestCase):
    def setUp(self):
        self.shape = (120, 200)
        self.scene = np.full((*self.shape, 3), 128, np.uint8)
        obj = np.zeros(self.shape, bool)
        obj[30:60, 50:120] = True
        edit = np.zeros_like(obj)
        edit[20:105, 40:170] = True
        blank = np.zeros_like(obj)
        arrays = {
            "obj_full": obj,
            "obj_visible": obj,
            "edit": edit,
            "contact": edit.copy(),
            "protect": ~edit,
            "fg_occ": blank,
        }
        self.masks = MaskBundle({"X": ObjectMasks("X", arrays)}, arrays, [], {}, [])
        support = {
            "polygons": [[[0.2, 0.5], [0.9, 0.5], [0.9, 0.9], [0.2, 0.9]]],
            "evidence": {"source": "proposal", "confirmed": False, "notes": "fixture"},
        }
        self.data = {
            "objects": [
                {
                    "id": "X",
                    "support_contacts": [
                        {"point": [0.30, 0.50], "support_id": "surface"},
                        {"point": [0.75, 0.75], "support_id": "surface"},
                    ],
                    "spatial_constraints": {"minimum_contact_count": 2},
                }
            ],
            "scene": {"support_regions": [{"id": "surface", "mask": support}]},
            "generator": {"preserve_full_frame": True, "context_crop": False},
            "evaluation": {"contact_gap_max_px": 4},
            "contact_rendering": {
                "enabled": True,
                "search_radius_px": 18,
                "strength": 0.12,
                "sigma_px": 2,
            },
        }
        self.job = ControlledJob(self.data, ROOT / "work/fixture.json", ROOT)

    def test_original_frame_and_conflicting_crop(self):
        self.assertEqual(
            request_frame(self.job, self.masks, self.shape)["rect_pixels"], [0, 0, 200, 120]
        )
        self.data["generator"]["context_crop"] = True
        with self.assertRaises(ConfigError):
            request_frame(self.job, self.masks, self.shape)

    def test_distinct_contacts_on_same_support_are_checked_separately(self):
        visible = {"X": self.masks.arrays["obj_visible"]}
        checks = contact_diagnostics(self.job, self.masks, visible)
        contacts = checks["objects"]["X"]["contacts"]
        self.assertEqual([c["status"] for c in contacts], ["PASS", "FAIL"])
        self.assertEqual(checks["objects"]["X"]["status"], "FAIL")
        result, shading = render_contact(
            self.scene, self.scene.copy(), self.job, self.masks, visible
        )
        self.assertEqual(len(shading["contacts"]), 1)
        self.assertTrue(np.array_equal(result[85:100, 140:160], self.scene[85:100, 140:160]))
        self.assertTrue(
            np.array_equal(
                result[~self.masks.arrays["edit"]], self.scene[~self.masks.arrays["edit"]]
            )
        )

    def test_missing_visible_mask_is_unknown(self):
        check = contact_diagnostics(self.job, self.masks, {})
        self.assertEqual(check["hard_checks"]["receiver_contact_X"]["status"], "UNKNOWN")

    def test_actual_object_is_never_clipped_by_edit_scope(self):
        escaped = self.masks.arrays["obj_visible"].copy()
        escaped[10, 10] = True
        with self.assertRaises(ConfigError):
            harmonize(self.scene, self.scene, self.job, self.masks, {"X": escaped})

    def test_explicit_alpha_and_same_aspect_resize_keep_full_canvas(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rgba.png"
            Image.new("RGBA", (100, 60), (100, 110, 120, 252)).save(path)
            image, record = normalize_candidate(
                path,
                (200, 120),
                {
                    "output_registration": "resize_same_aspect",
                    "alpha_policy": "near_opaque_unassociated_rgb",
                    "near_opaque_min_alpha": 251,
                },
            )
            self.assertEqual(image.shape, (120, 200, 3))
            self.assertTrue(np.all(image == [100, 110, 120]))
            self.assertTrue(record)

    def test_declared_coordinates_and_single_instance_association_are_audited(self):
        raw = {
            "coordinate_space": "normalized_1000",
            "objects": [
                {
                    "id": "valve_1",
                    "polygon": [[400, 200], [600, 200], [600, 500]],
                    "holes": [],
                    "parts": [],
                    "foreground_points": [[500, 300], [550, 350]],
                }
            ],
        }
        normalized = normalize_observation(raw)
        self.assertEqual(normalized["objects"][0]["polygon"][0], [0.4, 0.2])
        associated, log = bind_observation(normalized, ["S20"])
        self.assertEqual(associated["objects"][0]["id"], "S20")
        self.assertFalse(log[0]["identity_verified"])
        with self.assertRaises(AdapterError):
            bind_observation(normalized, ["S01", "S20"])
        raw["coordinate_space"] = "pixels"
        with self.assertRaises(AdapterError):
            normalize_observation(raw)

    def test_unknown_json_fields_can_be_discarded_but_bad_values_cannot(self):
        class FakeTransport:
            def post_json(self, *args):
                return {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": json.dumps({"value": self.value, "extra": "ignored"})
                            },
                        }
                    ]
                }

        profile = SimpleNamespace(
            vlm_model="doubao-seed-2.1-pro",
            vlm_endpoint="mock",
            vlm_credential=None,
            execution_options={"timeout_seconds": 1},
            redact=lambda x: x,
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["value"],
            "properties": {"value": {"type": "number", "maximum": 1}},
        }
        transport = FakeTransport()
        transport.value = 0.5
        result = VLMClient(profile, transport).ask(
            "fixture", [], schema, discard_unknown_fields=True
        )
        self.assertEqual(result["discarded_unknown_fields"], ["root.extra"])
        transport.value = 2
        with self.assertRaises(AdapterError):
            VLMClient(profile, transport).ask("fixture", [], schema, discard_unknown_fields=True)

    def test_analysis_coordinates_map_back_without_changing_export_canvas(self):
        value = {
            "objects": [
                {
                    "id": "X",
                    "polygon": [[0, 0], [1, 0], [1, 1]],
                    "holes": [],
                    "parts": [],
                    "foreground_points": [[0.5, 0.5]],
                }
            ]
        }
        frame = {"rect_pixels": [40, 20, 170, 105], "size": [130, 85], "original_size": [200, 120]}
        mapped = map_observation_to_original(value, frame)
        np.testing.assert_allclose(mapped["objects"][0]["polygon"][0], [0.2, 1 / 6])
        np.testing.assert_allclose(
            mapped["objects"][0]["foreground_points"][0], [0.525, 62.5 / 120]
        )
        self.assertEqual(request_frame(self.job, self.masks, self.shape)["size"], [200, 120])


if __name__ == "__main__":
    unittest.main()
