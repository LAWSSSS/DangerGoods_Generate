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
        "exclude_polygons": array(array(POINT, 3), 1),
        "coordinate_space": {"const": "normalized_original_background"},
        "evidence": EVIDENCE,
    },
    ("evidence",),
)
MASK["oneOf"] = [{"required": [key]} for key in ("path", "empty", "polygons")]
MASK["allOf"] = [
    {"if": {"required": ["polygons"]}, "then": {"required": ["coordinate_space"]}},
    {"if": {"required": ["sha256"]}, "then": {"required": ["path"]}},
    {"if": {"required": ["exclude_polygons"]}, "then": {"required": ["polygons"]}},
]
PAIRED_ANCHOR = obj({"id": ID, "mask": MASK, "description": TEXT}, ("id", "mask", "description"))
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
        "priority": {"enum": ["fixed", "support_first"]},
        "flexibility": obj(
            {
                "max_center_shift": number(0, 0.05),
                "scale_range": array(number(0.75, 1.25), 2, 2),
                "max_rotation_delta_deg": number(0, 45),
            },
            ("max_center_shift", "scale_range", "max_rotation_delta_deg"),
        ),
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
        "spatial_constraints": obj(
            {
                "forbidden": MASK,
                "edit_envelope": MASK,
                "occluded_fraction_range": array(number(0, 1), 2, 2),
                "minimum_contact_count": {"type": "integer", "minimum": 1, "maximum": 12},
            }
        ),
        "prompt_directives": obj(
            {name: array(TEXT) for name in ("appearance", "contact", "occlusion", "identity")}
        ),
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
                "prompt_version": {"enum": ["2.1", "2.2", "2.3", "3.0"]},
                "pipeline_version": {"enum": ["2.0", "3.0"]},
                "feature_flags": obj(
                    {
                        name: BOOL
                        for name in (
                            "v3_layer_compositor",
                            "v3_mask_refinement",
                            "v3_interaction_layers",
                            "v3_diagnostics",
                            "v3_repair_router",
                            "v3_structure_preservation",
                            "v3_causal_iteration",
                        )
                    }
                ),
                "fusion_v3": obj(
                    {
                        "allow_proposed_interactions": BOOL,
                        "surface_policy": {"enum": ["legacy_fixed", "paired_only"]},
                        "max_structure_loss_fraction": number(0, 0.1),
                        "shadow_receivers": array(
                            obj(
                                {
                                    "id": ID,
                                    "object_id": ID,
                                    "receiver": MASK,
                                    "allowed": MASK,
                                    "surface_kind": {
                                        "enum": ["locally_planar", "complex", "unknown"]
                                    },
                                },
                                ("id", "object_id", "receiver", "allowed", "surface_kind"),
                            ),
                            0,
                            24,
                        ),
                        "shadow": obj(
                            {
                                "max_darkening": number(0.01, 0.4),
                                "min_darkening": number(0.001, 0.1),
                                "max_chromatic_error": number(0.001, 0.1),
                                "max_texture_log_error": number(0.001, 0.3),
                                "min_valid_pixels": {
                                    "type": "integer",
                                    "minimum": 32,
                                    "maximum": 100000,
                                },
                                "min_valid_fraction": number(0.1, 1),
                                "min_component_pixels": {
                                    "type": "integer",
                                    "minimum": 4,
                                    "maximum": 1024,
                                },
                            }
                        ),
                    }
                ),
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
                        "enforce_spatial_plan": BOOL,
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
                        "alpha_policy": {"enum": ["reject", "near_opaque_unassociated_rgb"]},
                        "near_opaque_min_alpha": {
                            "type": "integer",
                            "minimum": 250,
                            "maximum": 255,
                        },
                        "near_opaque_outlier_min_alpha": {
                            "type": "integer",
                            "minimum": 240,
                            "maximum": 255,
                        },
                        "near_opaque_max_outlier_fraction": number(0, 0.0001),
                        "context_crop": BOOL,
                        "preserve_full_frame": BOOL,
                        "bbox_padding_px": number(0, 64),
                        "refinement_source": {"enum": ["original", "previous_candidate"]},
                        "layout_hint": {"enum": ["none", "procedural_proxy"]},
                        "context_padding_ratio": number(0.1, 1.5),
                        "route": {
                            "enum": ["concept", "guided_generation_then_restore", "native_mask"]
                        },
                    },
                    ("adapter", "route", "output_registration"),
                ),
                "iteration": obj(
                    {
                        "enabled": BOOL,
                        "max_rounds": {"type": "integer", "minimum": 1, "maximum": 5},
                        "prompt_mode": {"const": "append_only"},
                        "stop_on_visual_accept": BOOL,
                        "continue_on_unknown": BOOL,
                    }
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
                        "mode": {"enum": ["none", "bounded_lab", "paired_local"]},
                        "reference_roi": MASK,
                        "comparable_roi_confirmed": BOOL,
                        "max_lab_shift": array(number(0, 32), 3, 3),
                        "strength": number(0, 1),
                        "blur_sigma_px": number(0, 2),
                        "noise_std": number(0, 4),
                        "noise_seed": {"type": "integer", "minimum": 0},
                        "feedback_correction": obj(
                            {
                                "enabled": BOOL,
                                "reason": TEXT,
                                "object_ids": array(ID, 1),
                                "chroma_scale": number(0.65, 1),
                                "detail_sigma_px": number(0, 1.5),
                            },
                            ("enabled", "reason", "object_ids", "chroma_scale", "detail_sigma_px"),
                        ),
                        "feather_radius_px": number(0, 16),
                        "paired_anchors": array(PAIRED_ANCHOR, 1, 16),
                        "allow_proposed_anchors": BOOL,
                        "min_anchor_pixels": {"type": "integer", "minimum": 32, "maximum": 1000000},
                        "max_exposure_stops": number(0, 1),
                        "max_chromatic_stops": number(0, 0.5),
                        "max_pair_residual": number(0.001, 1),
                        "min_pair_inlier_fraction": number(0.1, 1),
                        "anchor_failure_policy": {"enum": ["stop", "skip_unreliable"]},
                        "min_reliable_anchors": {"type": "integer", "minimum": 1, "maximum": 16},
                        "surface_adjustments": array(
                            obj(
                                {
                                    "id": ID,
                                    "object_id": ID,
                                    "region": MASK,
                                    "chroma_scale": number(0.25, 1),
                                    "max_lab_chroma": number(0, 128),
                                    "highlight_knee_l": number(0, 100),
                                    "highlight_slope": number(0.1, 1),
                                    "strength": number(0, 1),
                                    "fill_unassigned_visible": BOOL,
                                },
                                ("id", "object_id", "region"),
                            ),
                            0,
                            64,
                        ),
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
                "candidate_geometry": obj(
                    {
                        "method": {"enum": ["grabcut_difference", "vlm_seeded_grabcut"]},
                        "seed": {"type": "integer", "minimum": 0, "maximum": 2147483647},
                        "iterations": {"type": "integer", "minimum": 1, "maximum": 8},
                        "max_side": {"type": "integer", "minimum": 256, "maximum": 1600},
                        "difference_floor": number(1, 64),
                        "max_background_difference": number(1, 100),
                        "min_visible_iou": number(0.01, 1),
                        "max_escape_fraction": number(0, 0.5),
                        "contact_tolerance_px": number(0, 100),
                        "min_seed_density": number(0, 1),
                        "min_area_ratio": number(0, 1),
                        "max_area_ratio": number(1, 10),
                        "max_component_count": {"type": "integer", "minimum": 1, "maximum": 64},
                        "reject_on_risk": BOOL,
                        "analysis_context": BOOL,
                    },
                    ("method", "seed", "reject_on_risk"),
                ),
                "registration": obj(
                    {
                        "max_side": {"type": "integer", "minimum": 512, "maximum": 2400},
                        "min_inliers": {"type": "integer", "minimum": 20, "maximum": 1000},
                        "max_rotation_deg": number(0, 3),
                        "max_scale_log": number(0, 0.05),
                        "max_translation_ratio": number(0, 0.05),
                        "max_reprojection_px": number(0.1, 8),
                    }
                ),
                "contact_rendering": obj(
                    {
                        "enabled": BOOL,
                        "search_radius_px": number(4, 120),
                        "sigma_px": number(0.5, 8),
                        "strength": number(0, 0.25),
                    },
                    ("enabled",),
                ),
                "output_dir": TEXT,
                "prompt_notes": TEXT,
                "prompt_directives": obj(
                    {name: array(TEXT) for name in ("scene_style", "negative")}
                ),
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
