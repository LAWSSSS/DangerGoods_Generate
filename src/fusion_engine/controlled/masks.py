"""Six masks derived only from explicit silhouettes, foreground and contact evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from fusion_engine.controlled.geometry import local_depth_order, obb_vertices
from fusion_engine.controlled.job import ControlledJob, asset_record, evidence_confirmed
from fusion_engine.errors import ConfigError
from fusion_engine.images import save_png

NAMES = ("obj_full", "fg_occ", "obj_visible", "contact", "edit", "protect")


@dataclass
class ObjectMasks:
    id: str
    arrays: dict[str, np.ndarray]
    sources: dict = field(default_factory=dict)


@dataclass
class MaskBundle:
    objects: dict[str, ObjectMasks]
    arrays: dict[str, np.ndarray]
    diagnostics: list[dict]
    depth_order: dict
    sources: list[dict]


def issue(code: str, status: str, detail: str, object_id: str | None = None) -> dict:
    return {
        "code": code,
        "status": status,
        "detail": detail,
        **({"object_id": object_id} if object_id else {}),
    }


def read_mask(
    job: ControlledJob, descriptor: dict, shape: tuple[int, int]
) -> tuple[np.ndarray, dict]:
    height, width = shape
    if "path" in descriptor:
        record = asset_record(job, descriptor)
        with Image.open(job.asset(descriptor["path"])) as image:
            if image.mode not in ("1", "L") or image.size != (width, height):
                raise ConfigError("Mask 必须为原背景尺寸的单通道 1/L 图，禁止隐式缩放")
            mask = np.asarray(image.convert("L"))
        if not np.isin(mask, [0, 255]).all():
            raise ConfigError("二值 Mask 只能含 0/255，软 alpha 必须走单独实现")
        return mask == 255, {
            **record,
            "method": "supplied_binary_mask",
            "evidence": descriptor["evidence"],
        }
    if descriptor.get("empty"):
        return np.zeros(shape, bool), {
            "method": "explicit_empty_mask",
            "evidence": descriptor["evidence"],
        }
    canvas = Image.new("L", (width, height), 0)
    drawing = ImageDraw.Draw(canvas)
    for polygon in descriptor["polygons"]:
        points = np.asarray(polygon, float) * [width, height]
        if abs(cv2.contourArea(points.astype(np.float32))) < 0.5:
            raise ConfigError("Mask 多边形退化，不能表示可验证轮廓")
        drawing.polygon(
            [(min(width - 1, round(x)), min(height - 1, round(y))) for x, y in points], fill=255
        )
    return np.array(canvas) == 255, {
        "method": "supplied_polygons_rasterized",
        "polygons": descriptor["polygons"],
        "evidence": descriptor["evidence"],
        "pixel_convention": "round(x*width), round(y*height), clamped_to_last_pixel",
    }


def build_masks(job: ControlledJob, shape: tuple[int, int]) -> MaskBundle:
    diagnostics, sources = [], []
    blank = np.zeros(shape, bool)
    arrays: dict[str, dict[str, np.ndarray]] = {}
    source_map: dict[str, dict] = {}
    require_confirmed = (
        job.data["constraints"]["require_confirmed_masks"] or job.data["mode"] == "C"
    )
    for item in job.data["objects"]:
        oid = item["id"]
        arrays[oid], source_map[oid] = {}, {}
        for name in ("obj_full", "fg_occ", "contact", "edit", "protect"):
            descriptor = item["masks"].get(name)
            if descriptor is None:
                if name in ("obj_full", "fg_occ", "contact"):
                    diagnostics.append(
                        issue("MISSING_MASK", "UNKNOWN", f"缺少 {name}；零占位不是真值", oid)
                    )
                arrays[oid][name] = blank.copy()
                source_map[oid][name] = {"method": "missing_not_ground_truth"}
                continue
            arrays[oid][name], record = read_mask(job, descriptor, shape)
            source_map[oid][name] = record
            sources.append({"object_id": oid, "mask": name, **record})
            if require_confirmed and not evidence_confirmed(descriptor["evidence"]):
                diagnostics.append(
                    issue("UNCONFIRMED_MASK", "UNKNOWN", f"{name} 尚无人/CAD审核证据", oid)
                )
        if not arrays[oid]["obj_full"].any():
            diagnostics.append(issue("EMPTY_FULL_MASK", "UNKNOWN", "完整投影轮廓为空", oid))
        if job.data["constraints"]["require_support"] and not arrays[oid]["contact"].any():
            diagnostics.append(
                issue("EMPTY_CONTACT_REGION", "UNKNOWN", "任务要求支撑，但接触交互区域为空", oid)
            )

    original_foreground = np.logical_or.reduce([value["fg_occ"] for value in arrays.values()])
    hidden = {oid: blank.copy() for oid in arrays}
    edges = []
    coverage: dict[tuple[str, str], np.ndarray] = {}
    for edge in job.data["scene"]["occlusion_edges"]:
        front, back = edge["front_id"], edge["back_id"]
        region, record = read_mask(job, edge["region"], shape)
        sources.append({"interaction": f"{front}>{back}", **record})
        overlap = arrays[front]["obj_full"] & arrays[back]["obj_full"]
        if not region.any() or np.any(region & ~overlap):
            diagnostics.append(
                issue(
                    "INVALID_OCCLUSION_REGION", "FAIL", "局部遮挡区域必须非空且属于双方完整投影交集"
                )
            )
        if require_confirmed and (
            not evidence_confirmed(edge["evidence"])
            or not evidence_confirmed(edge["region"]["evidence"])
        ):
            diagnostics.append(
                issue("UNCONFIRMED_OCCLUSION", "UNKNOWN", "多物品局部前后关系尚未审核")
            )
        hidden[back] |= region & overlap
        key = tuple(sorted((front, back)))
        coverage[key] = coverage.get(key, blank.copy()) | region
        edges.append((front, back, region))
    order = local_depth_order(edges, list(arrays), shape)
    if order["status"] == "FAIL":
        diagnostics.append(
            issue("LOCAL_OCCLUSION_CYCLE", "FAIL", "同一局部像素存在循环遮挡，需拆区域/修遮挡图")
        )
    ids = list(arrays)
    for i, first in enumerate(ids):
        for second in ids[i + 1 :]:
            overlap = arrays[first]["obj_full"] & arrays[second]["obj_full"] & ~original_foreground
            unresolved = overlap & ~coverage.get(tuple(sorted((first, second))), blank)
            if unresolved.any():
                diagnostics.append(
                    issue(
                        "UNRESOLVED_OVERLAP",
                        "UNKNOWN",
                        f"{first}/{second} 有 {int(unresolved.sum())} 像素未定义局部遮挡",
                    )
                )

    background_protect = blank.copy()
    if job.data["background"].get("protect"):
        descriptor = job.data["background"]["protect"]
        background_protect, record = read_mask(job, descriptor, shape)
        sources.append({"mask": "background_protect", **record})
        if require_confirmed and not evidence_confirmed(descriptor["evidence"]):
            diagnostics.append(
                issue("UNCONFIRMED_BACKGROUND_PROTECTION", "UNKNOWN", "背景保护区尚未审核")
            )
    explicit_protect = background_protect | original_foreground
    for value in arrays.values():
        explicit_protect |= value["protect"]
    edit_union = blank.copy()
    radius = int(np.ceil(job.data["harmonization"].get("feather_radius_px", 0)))
    for item in job.data["objects"]:
        oid, value = item["id"], arrays[item["id"]]
        visible = value["obj_full"] & ~original_foreground & ~hidden[oid]
        value["obj_visible"] = visible
        if np.any(visible & explicit_protect):
            diagnostics.append(
                issue("OBJECT_PROTECT_CONFLICT", "FAIL", "物体可见轮廓与显式保护区冲突", oid)
            )
        if not visible.any():
            diagnostics.append(
                issue("NO_VISIBLE_OBJECT", "UNKNOWN", "没有可见轮廓，不能验收物体存在", oid)
            )
        occluded = value["obj_full"] & original_foreground
        if job.data["constraints"]["require_foreground_occlusion"] and not occluded.any():
            diagnostics.append(
                issue("MISSING_FOREGROUND_OCCLUSION", "FAIL", "任务要求原前景遮挡，但交集为空", oid)
            )
        required_edit = visible | value["contact"]
        allowed_edit = required_edit.copy()
        if radius:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
            allowed_edit |= cv2.dilate(visible.astype(np.uint8), kernel).astype(bool)
        if "edit" not in item["masks"]:
            value["edit"] = allowed_edit & ~explicit_protect
            source_map[oid]["edit"] = {
                "method": "visible_union_contact_plus_boundary_band",
                "radius_px": radius,
                "ground_truth_claim": False,
            }
        elif np.any((required_edit & ~explicit_protect) & ~value["edit"]):
            diagnostics.append(
                issue("EDIT_MASK_INCOMPLETE", "FAIL", "提供的编辑区没有覆盖物体/接触区", oid)
            )
        if np.any(value["edit"] & ~allowed_edit & ~explicit_protect):
            diagnostics.append(
                issue(
                    "EDIT_MASK_EXCESSIVE", "FAIL", "编辑区超出可见目标、已声明交互区及边界带", oid
                )
            )
        value["edit"] &= ~explicit_protect
        edit_union |= value["edit"]
        value["fg_occ"] = original_foreground.copy()
    if job.data["mode"] == "A" and not edit_union.any():
        # A is explicitly a concept canvas; these are not segmentation/annotation masks.
        edit_union = ~explicit_protect
    protect = ~edit_union | explicit_protect
    if not protect.any() and job.data["mode"] in ("B", "C"):
        diagnostics.append(
            issue("EMPTY_PROTECT_SCOPE", "UNKNOWN", "保护范围为空，背景像素验收不可适用")
        )
    if not edit_union.any():
        diagnostics.append(issue("EMPTY_EDIT_SCOPE", "FAIL", "有效编辑区为空"))
    for value in arrays.values():
        value["protect"] = protect.copy()
    combined = {
        name: np.logical_or.reduce([value[name] for value in arrays.values()]) for name in NAMES
    }
    combined["edit"], combined["protect"] = edit_union, protect
    return MaskBundle(
        {oid: ObjectMasks(oid, value, source_map[oid]) for oid, value in arrays.items()},
        combined,
        diagnostics,
        order,
        sources,
    )


def save_masks(bundle: MaskBundle, destination: Path) -> dict:
    result = {"global": {}, "objects": {}}
    for name, mask in bundle.arrays.items():
        path = destination / f"M_{name}.png"
        save_png(mask.astype(np.uint8) * 255, path)
        result["global"][name] = str(path)
    for oid, item in bundle.objects.items():
        result["objects"][oid] = {}
        for name, mask in item.arrays.items():
            path = destination / oid / f"M_{name}.png"
            save_png(mask.astype(np.uint8) * 255, path)
            result["objects"][oid][name] = str(path)
    return result


def placement_preview(scene: np.ndarray, job: ControlledJob, bundle: MaskBundle) -> np.ndarray:
    canvas = scene.copy()
    color = np.array([180, 80, 20])
    selected = bundle.arrays["edit"]
    canvas[selected] = (0.72 * canvas[selected] + 0.28 * color).astype(np.uint8)
    canvas[bundle.arrays["fg_occ"]] = (
        0.55 * canvas[bundle.arrays["fg_occ"]] + 0.45 * np.array([40, 100, 240])
    ).astype(np.uint8)
    height, width = scene.shape[:2]
    for item in job.data["objects"]:
        points = np.rint(obb_vertices(item["placement"], width, height)).astype(np.int32)
        cv2.polylines(canvas, [points], True, (40, 220, 70), 2)
        cv2.putText(
            canvas, item["id"], tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 220, 70), 2
        )
    return canvas
