"""Offline behavior checks for per-reference sizes; no real API calls."""

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import generate_aliyun as engine
from object_catalog import (
    catalog_items,
    evaluation_user_prompt,
    load_catalog,
    render_prompt,
    select_profile,
    size_spec,
)

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "profiles/object_profiles.json"


def catalog():
    return load_catalog(CATALOG_PATH)


def config_with(profile):
    return SimpleNamespace(
        active_object_profile=profile,
        object_resize_scale=0.06,
        object_resize_scales={},
        object_size_briefs={},
    )


def test_every_reference_has_unique_complete_size_and_prompt():
    data = catalog()
    assert len(data["objects"]) == 38
    assert len({p["id"] for p in data["objects"]}) == 38
    assert len({p["reference_filename"] for p in data["objects"]}) == 38
    for profile in data["objects"]:
        assert profile["dimension_evidence"]["instance_measured"] is False
        assert all(value > 0 for value in profile["dimensions_mm"].values())
        assert profile["identity_features"] and profile["parts_mm"]
        assert profile["id"] in profile["prompt"]
        assert "六个格子不是六个实例" in profile["prompt"]
        assert (ROOT / f"profiles/prompts/{profile['id']}.txt").is_file()


def test_similar_broad_classes_do_not_share_sizes():
    data = catalog()
    small, large = select_profile(data, "S15"), select_profile(data, "S21")
    assert small["category"] == large["category"] == "阀门"
    assert small["parts_mm"]["标称通径"] == 25
    assert large["parts_mm"]["标称通径"] == 350
    assert (
        large["size_policy"]["reference_length_mm"]
        > small["size_policy"]["reference_length_mm"] * 5
    )
    air, motorcycle = select_profile(data, "S36"), select_profile(data, "S38")
    assert air["dimensions_mm"]["端到端长"] == 190
    assert motorcycle["dimensions_mm"]["端到端长"] == 350


def test_material_prompts_exclude_uniform_rust_on_plastic_and_aluminium():
    data = catalog()
    plastic = render_prompt(select_profile(data, "S25"))
    aluminium = render_prompt(select_profile(data, "S22"))
    stainless = render_prompt(select_profile(data, "S13"))
    assert "没有红褐铁锈" in plastic
    assert "禁止把铝壳和杆统一涂成红锈" in aluminium
    assert "禁止均匀厚红锈" in stainless


def test_engine_uses_active_item_instead_of_legacy_class_size():
    data = catalog()
    profile = select_profile(data, "S15")
    cfg = config_with(profile)
    cfg.object_size_briefs["阀门"] = {"height_cm": 55, "diameter_cm": 50, "parts": []}
    spec = engine.round_size_spec(cfg, "阀门")
    assert spec["profile_id"] == "S15"
    assert spec["real_height_cm"] == 12.5
    assert spec["real_diameter_cm"] == 5
    assert spec["background_calibration"] == "assumed_not_measured"
    assert engine.object_scale_range(cfg, "阀门") == tuple(
        profile["size_policy"]["visible_ratio_range"]
    )


def test_engine_prompt_avoids_generic_identity_and_material_overrides():
    profile = select_profile(catalog(), "S25")
    cfg = config_with(profile)
    text = engine.build_prompt(
        "密封胶罐", "只修接触位置", "所有部件都必须长铁锈", cfg=cfg, target_ratio=0.02
    )
    assert "白色 PP 广口带盖圆罐" in text
    assert "所有部件都必须长铁锈" not in text
    assert "只修接触位置" in text


def test_reference_collection_never_adds_another_same_class_item():
    items = catalog_items(catalog())
    selected = next(item for item in items if item["profile_id"] == "S15")
    views = engine.collect_object_views(
        selected["path"],
        selected["class"],
        items,
        3,
        Path("/Users/liuzhaocheng/Desktop/六视图"),
    )
    assert views == [selected["path"]]


def test_size_feedback_is_clamped_to_current_item_policy():
    profile = select_profile(catalog(), "S16")
    low, high = profile["size_policy"]["visible_ratio_range"]
    assert size_spec(profile, -1)["target_ratio"] == low
    assert size_spec(profile, 5)["target_ratio"] == high


def test_background_catalog_covers_all_73_assets_without_claiming_calibration():
    data = json.loads(
        (ROOT / "profiles/background_profiles.json").read_text(encoding="utf-8")
    )
    assert len(data["backgrounds"]) == 73
    assert sum(data["groups"].values()) == 73
    assert len(data["groups"]) == 3
    assert all(
        item["calibration_status"] == "unmeasured_assumption"
        for item in data["backgrounds"]
    )
    assert all(
        (ROOT / "input/backgrounds" / item["filename"]).is_file()
        for item in data["backgrounds"]
    )


def test_vlm_evaluation_does_not_assert_pixels_equal_measured_centimeters():
    profile = select_profile(catalog(), "S21")
    text = evaluation_user_prompt(profile, size_spec(profile), "无既往轮次")
    assert "不得宣称从单图测得真实长度" in text
    assert "不要求可见片段长于完整横截面" in text


def test_config_load_enables_catalog_without_printing_credentials():
    fake = {
        "model": "wan2.7-image",
        "vlm_model": "doubao-seed-2.1-pro",
        "api_key_ali": "fake-ali-credential-not-to-print",
        "api_key_huoshan": "fake-huoshan-credential-not-to-print",
        "photo_generate_api_url_ali": "https://api.example.com/compatible-mode/v1",
        "vlm_api_huoshan_url": "https://vlm.example.com/v3",
    }
    output = io.StringIO()
    with (
        patch.object(engine, "load_bailian_raw", return_value=fake),
        redirect_stdout(output),
    ):
        cfg = engine.load_app_config()
    assert len(cfg.object_catalog["objects"]) == 38
    assert cfg.six_views_dir == Path("/Users/liuzhaocheng/Desktop/六视图")
    assert "fake-ali-credential" not in output.getvalue()
    assert "fake-huoshan-credential" not in output.getvalue()
    assert "fake-ali-credential" not in repr(cfg)
