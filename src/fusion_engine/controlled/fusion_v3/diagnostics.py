"""Saved stage evidence and limited statistics; none is a naturalness score."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from fusion_engine.artifacts import sha256, write_json
from fusion_engine.controlled.fusion_v3.compositor import Layers, compose_layers
from fusion_engine.controlled.fusion_v3.interactions import Interaction
from fusion_engine.images import load_rgb, save_png


def saved_contract(scene: np.ndarray, path: Path, masks, processing: dict) -> dict:
    descriptor = processing.get("authorized_mask")
    if not descriptor or processing.get("status", "APPLIED") != "APPLIED":
        return {"status": "FAIL", "detail": "V3未完成合成或缺少实际授权Mask"}
    mask_path = Path(descriptor["path"])
    if not mask_path.is_file() or sha256(mask_path) != descriptor["sha256"]:
        return {"status": "FAIL", "detail": "V3授权Mask哈希不匹配"}
    with Image.open(mask_path) as image:
        auth = np.asarray(image)
    result = load_rgb(path)
    if (
        auth.shape != scene.shape[:2]
        or not np.isin(auth, [0, 255]).all()
        or result.shape != scene.shape
    ):
        return {"status": "FAIL", "detail": "V3保存结果或授权Mask尺寸非法"}
    changed = np.any(result != scene, axis=2)
    protected = int((changed & masks.arrays["protect"]).sum())
    front = int((changed & masks.arrays["fg_occ"]).sum())
    escaped = int((changed & (auth != 255)).sum())
    return {
        "status": "PASS" if protected == front == escaped == 0 else "FAIL",
        "detail": "最终文件重新解码后的RGB8/前景/实际图层授权契约，不代表视觉验收",
        "changed_protected_pixels": protected,
        "changed_foreground_pixels": front,
        "changed_outside_actual_layers": escaped,
        "output_size": list(result.shape[1::-1]),
    }


def pixel_audit(scene: np.ndarray, image: np.ndarray, layers: Layers) -> dict:
    if image.shape != scene.shape or image.dtype != np.uint8:
        return {"status": "FAIL", "reason": "输出尺寸/类型与原图不同"}
    delta = np.abs(image.astype(np.int16) - scene.astype(np.int16))
    changed = np.any(delta, axis=2)
    return {
        "canvas": list(image.shape[1::-1]),
        "changed_pixels": int(changed.sum()),
        "changed_protected_pixels": int((changed & layers.protect).sum()),
        "protected_max_channel_delta": int(delta[layers.protect].max())
        if layers.protect.any()
        else None,
        "changed_front_pixels": int((changed & layers.front).sum()),
        "changed_outside_authorized": int((changed & ~layers.authorized).sum()),
        "claim": "仅RGB8像素契约，不能证明自然度、身份、三维几何或标注正确",
    }


def surface_stats(image: np.ndarray, region: np.ndarray) -> dict:
    if not region.any():
        return {"status": "UNKNOWN", "pixels": 0}
    values = image[region].astype(np.float32).reshape(-1, 1, 3) / 255
    lab = cv2.cvtColor(values, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    return {
        "pixels": int(region.sum()),
        "L_star_q10_q50_q90": np.quantile(lab[:, 0], [0.1, 0.5, 0.9]).tolist(),
        "C_ab_q10_q50_q90": np.quantile(
            np.linalg.norm(lab[:, 1:], axis=1), [0.1, 0.5, 0.9]
        ).tolist(),
    }


def save_diagnostics(
    directory: Path,
    scene: np.ndarray,
    aligned: np.ndarray,
    foreground: np.ndarray,
    final: np.ndarray,
    layers: Layers,
    cast: Interaction,
    contact: Interaction,
) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    blank = np.zeros(scene.shape[:2], bool)
    plain_layers = replace(layers, cast=blank, contact=blank)
    plain, _ = compose_layers(scene, foreground, plain_layers)
    no_ao_layers = replace(layers, contact=blank)
    no_ao, _ = compose_layers(scene, foreground, no_ao_layers, cast_transmission=cast.transmission)
    stages = {
        "V0_original": scene,
        "V2_aligned_candidate": aligned,
        "V3_adjusted_object": foreground,
        "V4_object_only": plain,
        "A2_no_contact_AO": no_ao,
        "V5_final_layers": final,
    }
    paths, audits, stats = {}, {}, {}
    for name, pixels in stages.items():
        path = directory / f"{name}.png"
        save_png(pixels, path)
        paths[name] = {"path": str(path), "sha256": sha256(path)}
        audits[name] = pixel_audit(scene, load_rgb(path), layers)
        stats[name] = surface_stats(pixels, layers.object)
    for field in fields(layers):
        save_png(
            getattr(layers, field.name).astype(np.uint8) * 255, directory / f"mask_{field.name}.png"
        )
    # Enclosed negative spaces are topology diagnostics, not verified anatomical holes.
    count, labels = cv2.connectedComponents((~layers.object).astype(np.uint8), connectivity=8)
    border_ids = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    holes = (labels > 0) & ~np.isin(labels, border_ids)
    save_png(holes.astype(np.uint8) * 255, directory / "mask_enclosed_negative_space.png")
    np.save(directory / "cast_transmission.npy", cast.transmission, allow_pickle=False)
    np.save(directory / "contact_transmission.npy", contact.transmission, allow_pickle=False)
    dark = np.clip((1 - np.minimum(cast.transmission, contact.transmission)) * 255, 0, 255).astype(
        np.uint8
    )
    save_png(np.repeat(dark[..., None], 3, axis=2), directory / "interaction_attenuation.png")
    delta = np.max(np.abs(final.astype(np.int16) - scene.astype(np.int16)), axis=2).astype(np.uint8)
    save_png(
        cv2.cvtColor(cv2.applyColorMap(delta, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB),
        directory / "difference_heatmap.png",
    )
    ys, xs = np.where(layers.object)
    pad = 45
    x0, x1 = max(0, int(xs.min()) - pad), min(scene.shape[1], int(xs.max()) + pad + 1)
    y0, y1 = max(0, int(ys.min()) - pad), min(scene.shape[0], int(ys.max()) + pad + 1)
    for name in ("V2_aligned_candidate", "V4_object_only", "V5_final_layers"):
        crop = Image.fromarray(stages[name][y0:y1, x0:x1])
        crop.resize((crop.width * 2, crop.height * 2), Image.Resampling.NEAREST).save(
            directory / f"analysis_only_200pct_{name}.png"
        )
    report = {
        "stages": paths,
        "saved_pixel_audits": audits,
        "object_statistics": stats,
        "enclosed_negative_space_components": max(0, count - 1 - len(set(border_ids) - {0})),
        "hole_evidence": "binary_topology_only_not_verified_holes",
        "analysis_crop_rect": [x0, y0, x1, y1],
        "final_output_cropped": False,
        "cast": cast.report,
        "contact": contact.report,
        "human_visual_acceptance": "NOT_PERFORMED",
    }
    write_json(directory / "diagnostics.json", report)
    return {
        "directory": str(directory),
        "report_sha256": sha256(directory / "diagnostics.json"),
        "final_png_readback": audits["V5_final_layers"],
        "analysis_crops_are_not_deliverables": True,
    }
