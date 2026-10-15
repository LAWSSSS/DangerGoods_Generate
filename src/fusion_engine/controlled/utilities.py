"""The manual's executable offline utilities. Baselines never imply realistic fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from fusion_engine.artifacts import sha256
from fusion_engine.controlled.geometry import project_points, reference_projection
from fusion_engine.controlled.job import read_json
from fusion_engine.errors import FusionError
from fusion_engine.evaluation import protected_delta
from fusion_engine.images import image_info, load_rgb, save_png


def binary(path: Path, shape: tuple) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode not in ("1", "L") or image.size != (shape[1], shape[0]):
            raise FusionError("Mask 必须是原画布尺寸的单通道图像")
        array = np.asarray(image.convert("L"))
    if not np.isin(array, [0, 255]).all():
        raise FusionError("Mask 只能有 0/255")
    return array == 255


def destination(path: Path, suffixes: tuple[str, ...]) -> Path:
    value = path.expanduser().resolve()
    if value.exists() or value.suffix.lower() not in suffixes:
        raise FusionError("输出不得覆盖已有文件，且扩展名必须符合命令格式")
    value.parent.mkdir(parents=True, exist_ok=True)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="手册离线图像/尺度/背景保护工具，无模型请求")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--image", "--input", dest="image", type=Path, required=True)
    scale = commands.add_parser("scale")
    object_size = scale.add_mutually_exclusive_group(required=True)
    object_size.add_argument("--object-length", type=float)
    object_size.add_argument("--object-meters", type=float)
    reference_size = scale.add_mutually_exclusive_group(required=True)
    reference_size.add_argument("--reference-length", type=float)
    reference_size.add_argument("--reference-meters", type=float)
    scale.add_argument("--reference-pixels", type=float, required=True)
    scale.add_argument("--depth-ratio-range", nargs=2, type=float)
    scale.add_argument("--foreshortening-ratio-range", nargs=2, type=float)
    scale.add_argument(
        "--relative-depth", type=float, help="手册兼容：Z_obj/Z_ref；显式值转换为逆比"
    )
    scale.add_argument("--direction-factor", type=float, help="手册兼容：gamma_obj/gamma_ref")
    scale.add_argument("--units", choices=("mm", "cm", "m"))
    project = commands.add_parser("project")
    project.add_argument("--camera-json", type=Path, required=True)
    for name in ("compare-background", "restore-protected", "composite"):
        command = commands.add_parser(name)
        command.add_argument(
            "--original", "--background", dest="original", type=Path, required=True
        )
        command.add_argument(
            "--candidate",
            "--composite",
            "--object-layer",
            dest="candidate",
            type=Path,
            required=True,
        )
        command.add_argument(
            "--mask",
            "--protect-mask",
            "--visible-mask",
            dest="mask",
            type=Path,
            required=True,
            help="compare/restore:保护区白；composite:编辑区白",
        )
        if name != "compare-background":
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument("--tolerance", type=int, default=0)
    grade = commands.add_parser("grade-case")
    grade.add_argument("--image", "--input", dest="image", type=Path, required=True)
    grade.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = {}
        if args.command == "inspect":
            with Image.open(args.image) as image:
                rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
            result = {
                **image_info(args.image),
                "sha256": sha256(args.image),
                "rgb_mean": rgb.mean(axis=(0, 1)).tolist(),
                "mean_scope": "decoded_RGB8_alpha_ignored",
            }
        elif args.command == "scale":
            using_meters = args.object_meters is not None or args.reference_meters is not None
            if using_meters:
                if (
                    args.object_meters is None
                    or args.reference_meters is None
                    or args.units not in (None, "m")
                ):
                    raise FusionError("meters 参数需成对给出，且 units 只能为 m")
                args.object_length, args.reference_length = (
                    args.object_meters,
                    args.reference_meters,
                )
            if args.relative_depth is not None:
                if (
                    args.depth_ratio_range is not None
                    or not np.isfinite(args.relative_depth)
                    or args.relative_depth <= 0
                ):
                    raise FusionError("relative-depth 必须为正且不能与 depth-ratio-range 同时填写")
                args.depth_ratio_range = [1 / args.relative_depth, 1 / args.relative_depth]
            if args.direction_factor is not None:
                if args.foreshortening_ratio_range is not None:
                    raise FusionError("direction-factor 不能与 foreshortening-ratio-range 同时填写")
                args.foreshortening_ratio_range = [args.direction_factor, args.direction_factor]
            values = [
                args.object_length,
                args.reference_length,
                args.reference_pixels,
                *(args.depth_ratio_range or []),
                *(args.foreshortening_ratio_range or []),
            ]
            if not all(np.isfinite(value) and value > 0 for value in values):
                raise FusionError("尺度输入必须是正的有限数值")
            data = {
                "object_length": args.object_length,
                "reference_length": args.reference_length,
                "reference_pixels": args.reference_pixels,
                "units": "m" if using_meters else (args.units or "mm"),
            }
            if args.depth_ratio_range:
                data["depth_ratio_range"] = args.depth_ratio_range
            if args.foreshortening_ratio_range:
                data["foreshortening_ratio_range"] = args.foreshortening_ratio_range
            result = reference_projection(data)
        elif args.command == "project":
            result = project_points(read_json(args.camera_json))
        elif args.command in ("compare-background", "restore-protected", "composite"):
            original = load_rgb(args.original)
            alpha = None
            if args.command == "composite":
                with Image.open(args.candidate) as image:
                    rgba = np.asarray(ImageOps.exif_transpose(image).convert("RGBA"))
                    candidate, alpha = rgba[..., :3].copy(), rgba[..., 3] / 255.0
            else:
                candidate = load_rgb(args.candidate)
            if original.shape != candidate.shape:
                raise FusionError("图片尺寸不同，禁止隐式缩放/配准")
            selected = binary(args.mask, original.shape[:2])
            if args.command == "compare-background":
                if args.tolerance < 0 or args.tolerance > 255:
                    raise FusionError("tolerance 必须介于0和255；严格像素保护用0")
                result = protected_delta(original, candidate, selected)
                result["status"] = (
                    "UNKNOWN"
                    if not selected.any()
                    else "PASS"
                    if result["max_channel_delta"] <= args.tolerance
                    else "FAIL"
                )
                result.update(
                    tolerance=args.tolerance,
                    exact_pixel_lock_claim=args.tolerance == 0 and result["status"] == "PASS",
                )
            else:
                output = destination(args.output, (".png",))
                if args.command == "restore-protected":
                    candidate[selected] = original[selected]
                    result["method"] = "restore_original_protected_pixels"
                else:
                    alpha = alpha * selected
                    candidate = np.clip(
                        np.rint(original * (1 - alpha[..., None]) + candidate * alpha[..., None]),
                        0,
                        255,
                    ).astype(np.uint8)
                    result.update(
                        method="masked_alpha_paste_baseline",
                        realism_claim=False,
                        annotations_created=False,
                        blend_space="encoded_RGB8_baseline",
                    )
                save_png(candidate, output)
                result.update(output=str(output), sha256=sha256(output))
        elif args.command == "grade-case":
            output = destination(args.output, (".jpg", ".jpeg"))
            image = load_rgb(args.image)
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
            lab[..., 0] = 1.20 * lab[..., 0] + 9
            lab[..., 1] = 128 + 0.92 * (lab[..., 1] - 128) - 2.5
            lab[..., 2] = 128 + 0.89 * (lab[..., 2] - 128) - 4.0
            image = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
            image = cv2.GaussianBlur(image, (0, 0), 0.55)
            Image.fromarray(image).save(output, format="JPEG", quality=87)
            result = {
                "output": str(output),
                "sha256": sha256(output),
                "historical_parameters_only": True,
                "lab8_formula": ["L'=1.20L+9", "a'=128+.92(a-128)-2.5", "b'=128+.89(b-128)-4"],
                "gaussian_sigma": 0.55,
                "jpeg_quality": 87,
                "strict_background_claim": False,
                "identical_to_historical_bytes_claim": False,
            }
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return {"FAIL": 3, "UNKNOWN": 4}.get(result.get("status"), 0)
    except (FusionError, OSError, ValueError, KeyError, TypeError) as exc:
        print(
            json.dumps(
                {
                    "error": type(exc).__name__,
                    "message": str(exc)
                    if isinstance(exc, FusionError)
                    else "输入无效；未输出文件原始内容",
                },
                ensure_ascii=False,
            )
        )
        return 2
