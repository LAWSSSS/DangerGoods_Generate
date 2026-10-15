"""Strict public job/review contracts. These documents contain no API credentials."""

from __future__ import annotations

from typing import Any


def obj(properties: dict, required: tuple | list = ()) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(required),
    }


def array(items: dict, minimum: int = 0, maximum: int | None = None) -> dict:
    result = {"type": "array", "items": items, "minItems": minimum}
    if maximum is not None:
        result["maxItems"] = maximum
    return result


def number(low: float | None = None, high: float | None = None) -> dict:
    return {
        "type": "number",
        **({"minimum": low} if low is not None else {}),
        **({"maximum": high} if high is not None else {}),
    }


TEXT = {"type": "string", "minLength": 1}
BOOL = {"type": "boolean"}
ID = {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_-]{0,63}$"}
HASH = {"type": "string", "pattern": "^[a-f0-9]{64}$"}
POINT = array(number(0, 1), 2, 2)
VECTOR = array(number(), 2, 2)
EVIDENCE = obj(
    {
        "source": {
            "enum": [
                "human",
                "cad",
                "calibrated_geometry",
                "segmentation_model",
                "vlm",
                "catalog",
                "estimate",
                "proposal",
                "synthetic_fixture",
            ]
        },
        "confirmed": BOOL,
        "reviewer": TEXT,
        "notes": TEXT,
        "confidence": number(0, 1),
    },
    ("source", "confirmed", "notes"),
)
MASK = obj(
    {
        "path": TEXT,
        "sha256": HASH,
        "empty": {"const": True},
        "polygons": array(array(POINT, 3), 1),
        "coordinate_space": {"const": "normalized_original_background"},
        "evidence": EVIDENCE,
    },
    ("evidence",),
)
MASK["oneOf"] = [{"required": [key]} for key in ("path", "empty", "polygons")]
MASK["allOf"] = [
    {"if": {"required": ["polygons"]}, "then": {"required": ["coordinate_space"]}},
    {"if": {"required": ["sha256"]}, "then": {"required": ["path"]}},
]
MATERIALS = [
    "carbon_steel",
    "painted_steel",
    "cast_iron",
    "stainless_steel",
    "aluminium",
    "galvanized_steel",
    "copper",
    "brass",
    "plastic",
    "rubber",
    "wood",
    "paper",
    "glass",
    "fabric",
    "ceramic",
    "unknown",
]
REFERENCE_SCALE = obj(
    {
        "reference_pixels": number(0.001),
        "object_length": number(0.001),
        "reference_length": number(0.001),
        "units": {"enum": ["mm", "cm", "m"]},
        "depth_ratio_range": array(number(0.001), 2, 2),
        "foreshortening_ratio_range": array(number(0.001), 2, 2),
    },
    ("reference_pixels", "object_length", "reference_length", "units"),
)
SCALE = obj(
    {
        "level": {"enum": [0, 1, 2, 3]},
        "evidence": EVIDENCE,
        "reference_projection": REFERENCE_SCALE,
        "pixel_length_range": array(number(0.001), 2, 2),
        "assumptions": array(TEXT),
        "camera": obj(
            {
                "K": array(array(number(), 3, 3), 3, 3),
                "R": array(array(number(), 3, 3), 3, 3),
                "t": array(number(), 3, 3),
                "points_3d": array(array(number(), 3, 3), 4),
                "units": {"enum": ["mm", "cm", "m"]},
            },
            ("K", "R", "t", "points_3d", "units"),
        ),
    },
    ("level", "evidence"),
)
PLACEMENT = obj(
    {
        "box_type": {"const": "oriented"},
        "coordinate_space": {"const": "normalized_original_background"},
        "center": POINT,
        "width": number(0.000001, 1),
        "height": number(0.000001, 1),
        "rotation_deg": number(-180, 180),
        "allow_crop": BOOL,
        "status": {"enum": ["proposed", "confirmed"]},
        "evidence": EVIDENCE,
        "scale": SCALE,
        "pose_3d_notes": TEXT,
    },
    (
        "box_type",
        "coordinate_space",
        "center",
        "width",
        "height",
        "rotation_deg",
        "status",
        "evidence",
        "scale",
    ),
)
MATERIAL_PART = obj(
    {
        "id": ID,
        "base_material": {"enum": MATERIALS},
        "coating": TEXT,
        "evidence": EVIDENCE,
        "critical": BOOL,
        "weathering_locations": array(TEXT),
        "allowed_damage": array(TEXT),
    },
    ("id", "base_material", "evidence"),
)
CONTACT = obj(
    {
        "support_id": ID,
        "point": POINT,
        "normal_image": VECTOR,
        "evidence": EVIDENCE,
    },
    ("support_id", "point", "evidence"),
)
OBJECT = obj(
    {
        "id": ID,
        "category": TEXT,
        "references": array(
            obj(
                {
                    "path": TEXT,
                    "view_id": ID,
                    "sha256": HASH,
                    "orientation_hint": TEXT,
                    "complete": BOOL,
                    "source_kind": {"enum": ["photo", "atlas", "rusted_atlas", "cad"]},
                },
                ("path", "view_id", "source_kind"),
            ),
            1,
            8,
        ),
        "dimensions": obj(
            {
                "status": {"enum": ["measured", "catalog_verified", "estimated", "unknown"]},
                "units": {"enum": ["mm", "cm", "m"]},
                "values": {"type": "object", "additionalProperties": number(0.000001)},
                "ranges": {"type": "object", "additionalProperties": array(number(0.000001), 2, 2)},
                "evidence": EVIDENCE,
            },
            ("status", "units", "evidence"),
        ),
        "identity": obj(
            {
                "level": {"enum": ["category", "instance"]},
                "critical_parts": array(TEXT),
                "positive": array(TEXT),
                "negative": array(TEXT),
            },
            ("level", "critical_parts", "positive", "negative"),
        ),
        "material_parts": array(MATERIAL_PART, 1),
        "ageing": obj(
            {
                "level": {"type": "integer", "minimum": 0, "maximum": 4},
                "exposure_history": TEXT,
                "allow_structural_damage": BOOL,
            },
            ("level", "exposure_history", "allow_structural_damage"),
        ),
        "placement": PLACEMENT,
        "masks": obj({name: MASK for name in ("obj_full", "fg_occ", "contact", "edit", "protect")}),
        "support_contacts": array(CONTACT),
        "center_of_mass_projection": POINT,
        "stability_evidence": EVIDENCE,
        "render_class": {"enum": ["opaque_rigid", "transparent", "soft_body"]},
    },
    (
        "id",
        "category",
        "references",
        "dimensions",
        "identity",
        "material_parts",
        "ageing",
        "placement",
        "masks",
        "support_contacts",
        "render_class",
    ),
)
CHECK_NAMES = [
    "identity",
    "support",
    "occlusion",
    "material",
    "lighting",
    "style",
    "object_count",
    "realism",
]


def job_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:scene-fusion:job:2.0",
        "title": "图像融合任务 v2.0",
        **obj(
            {
                "schema_version": {"const": "2.0"},
                "task_id": ID,
                "workspace": TEXT,
                "mode": {"enum": ["A", "B", "C"]},
                "background": obj(
                    {
                        "path": TEXT,
                        "sha256": HASH,
                        "width": {"type": "integer", "minimum": 1},
                        "height": {"type": "integer", "minimum": 1},
                        "protect": MASK,
                        "description": TEXT,
                    },
                    ("path",),
                ),
                "objects": array(OBJECT, 1, 12),
                "scene": obj(
                    {
                        "lighting": TEXT,
                        "camera_style": TEXT,
                        "gravity_image_direction": VECTOR,
                        "evidence": EVIDENCE,
                        "support_regions": array(
                            obj(
                                {
                                    "id": ID,
                                    "mask": MASK,
                                    "load_bearing": {"enum": ["confirmed", "unknown", "unsafe"]},
                                    "description": TEXT,
                                },
                                ("id", "mask", "load_bearing", "description"),
                            )
                        ),
                        "occlusion_edges": array(
                            obj(
                                {
                                    "front_id": ID,
                                    "back_id": ID,
                                    "region": MASK,
                                    "evidence": EVIDENCE,
                                },
                                ("front_id", "back_id", "region", "evidence"),
                            )
                        ),
                    },
                    ("lighting", "camera_style", "support_regions", "occlusion_edges"),
                ),
                "constraints": obj(
                    {
                        "require_support": BOOL,
                        "require_foreground_occlusion": BOOL,
                        "require_confirmed_masks": BOOL,
                        "background_protection": {"enum": ["strict", "none"]},
                        "require_stability_evidence": BOOL,
                    },
                    (
                        "require_support",
                        "require_foreground_occlusion",
                        "require_confirmed_masks",
                        "background_protection",
                        "require_stability_evidence",
                    ),
                ),
                "generator": obj(
                    {
                        "adapter": {"enum": ["replay", "bailian"]},
                        "output_registration": {"enum": ["strict", "resize_same_aspect"]},
                        "route": {
                            "enum": ["concept", "guided_generation_then_restore", "native_mask"]
                        },
                    },
                    ("adapter", "route", "output_registration"),
                ),
                "budget": obj(
                    {
                        "max_image_calls": {"type": "integer", "minimum": 0, "maximum": 10},
                        "max_vlm_calls": {"type": "integer", "minimum": 0, "maximum": 10},
                        "max_candidates": {"type": "integer", "minimum": 1, "maximum": 80},
                    },
                    ("max_image_calls", "max_vlm_calls", "max_candidates"),
                ),
                "harmonization": obj(
                    {
                        "mode": {"enum": ["none", "bounded_lab"]},
                        "reference_roi": MASK,
                        "comparable_roi_confirmed": BOOL,
                        "max_lab_shift": array(number(0, 32), 3, 3),
                        "strength": number(0, 1),
                        "blur_sigma_px": number(0, 2),
                        "noise_std": number(0, 4),
                        "noise_seed": {"type": "integer", "minimum": 0},
                        "feather_radius_px": number(0, 16),
                    },
                    ("mode",),
                ),
                "evaluation": obj(
                    {
                        "center_error_max": number(0, 1),
                        "rotation_error_max_deg": number(0, 90),
                        "scale_log_error_max": number(0),
                        "mask_iou_min": number(0, 1),
                        "contact_gap_max_px": number(0),
                        "min_changed_visible_fraction": number(0, 1),
                        "required_visual_checks": array({"enum": CHECK_NAMES}, 1),
                        "require_observed_masks": BOOL,
                    },
                    (
                        "center_error_max",
                        "scale_log_error_max",
                        "mask_iou_min",
                        "contact_gap_max_px",
                        "min_changed_visible_fraction",
                        "required_visual_checks",
                        "require_observed_masks",
                    ),
                ),
                "output_dir": TEXT,
                "prompt_notes": TEXT,
            },
            (
                "schema_version",
                "task_id",
                "workspace",
                "mode",
                "background",
                "objects",
                "scene",
                "constraints",
                "generator",
                "budget",
                "harmonization",
                "evaluation",
                "output_dir",
            ),
        ),
    }


def review_schema() -> dict[str, Any]:
    check = obj(
        {"status": {"enum": ["PASS", "FAIL", "UNKNOWN"]}, "notes": TEXT}, ("status", "notes")
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **obj(
            {
                "schema_version": {"const": "2.0"},
                "master_sha256": HASH,
                "reviewer": TEXT,
                "notes": TEXT,
                "checks": obj({name: check for name in CHECK_NAMES}, CHECK_NAMES),
                "objects": array(
                    obj(
                        {
                            "id": ID,
                            "amodal_mask": MASK,
                            "visible_mask": MASK,
                            "mask_matches_master_confirmed": BOOL,
                        },
                        ("id", "mask_matches_master_confirmed"),
                    )
                ),
            },
            ("schema_version", "master_sha256", "reviewer", "notes", "checks", "objects"),
        ),
    }
