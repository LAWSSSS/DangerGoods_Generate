"""Procedural 2D layout hint, never a final composite or a native geometry control."""

from __future__ import annotations

import base64
import hashlib
import io

import numpy as np
from PIL import Image

from fusion_engine.controlled.geometry import obb_vertices
from fusion_engine.controlled.masks import read_mask


def layout_hint(prepared) -> tuple[np.ndarray, dict]:
    scene, job, masks = prepared.scene, prepared.job, prepared.masks
    result = scene.copy()
    height, width = scene.shape[:2]
    records = []
    for item in job.data["objects"]:
        oid, p = item["id"], item["placement"]
        selected = masks.objects[oid].arrays["obj_visible"]
        yy, xx = np.where(selected)
        if not len(xx):
            continue
        angle = np.radians(p["rotation_deg"])
        local_y = (
            -(xx - p["center"][0] * width) * np.sin(angle)
            + (yy - p["center"][1] * height) * np.cos(angle)
        ) / (p["height"] * height / 2)
        # Merely a low-detail shading cue across a design proxy; no recovered normals or light.
        cue = 0.76 + 0.24 * np.sqrt(np.maximum(0, 1 - np.clip(local_y, -1, 1) ** 2))
        base = np.array([143, 116, 108] if "气瓶" in item["category"] else [137, 129, 119], float)
        result[selected] = np.rint(base * cue[:, None]).astype(np.uint8)
        for patch in job.data["harmonization"].get("surface_adjustments", []):
            if patch["object_id"] == oid and patch["id"] == "wheel":
                region = read_mask(job, patch["region"], (height, width))[0] & selected
                result[region] = [99, 114, 119]
        records.append(
            {
                "id": oid,
                "pixels": int(selected.sum()),
                "projection_vertices": obb_vertices(p, width, height).tolist(),
                "pose_source": "unconfirmed_design_proxy",
            }
        )
    return result, {
        "method": "procedural_2D_shape_and_low_detail_shading",
        "objects": records,
        "full_frame_size": [width, height],
        "cropped": False,
        "reference_pixels_pasted": False,
        "physical_renderer_used": False,
        "native_mask_control": False,
        "purpose": "extra_full_frame_visual_layout_reference_only_not_final_image",
    }


def encode_hint(pixels: np.ndarray, max_pixels: int = 4194304) -> tuple[str, dict]:
    image = Image.fromarray(pixels, "RGB")
    original = image.size
    ratio = min(1, (max_pixels / (image.width * image.height)) ** 0.5)
    if ratio < 1:
        image = image.resize(
            (int(image.width * ratio), int(image.height * ratio)), Image.Resampling.LANCZOS
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = buffer.getvalue()
    return "data:image/png;base64," + base64.b64encode(data).decode(), {
        "path": "<procedural_full_frame_layout_reference>",
        "original_size": list(original),
        "source_size_before_resize": list(original),
        "sent_size": list(image.size),
        "encoded_sha256": hashlib.sha256(data).hexdigest(),
        "crop": None,
        "source": "procedural_2D_layout_not_original_reference_or_final_candidate",
    }
