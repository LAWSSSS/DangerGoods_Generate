"""Scene-conditioned fusion requirements and conservative visual review gates."""

from __future__ import annotations

import math
import re
from typing import Any

FUSION_SCORE_AXES = (
    "weathering_score",
    "lighting_score",
    "occlusion_score",
    "style_score",
    "background_preservation_score",
    "identity_score",
)
QUALITY_THRESHOLD = 85.0


def geometry_identity(profile: dict[str, Any]) -> list[str]:
    colors = r"青绿色|深绿色|深蓝色|紫粉色|银白色|红色|绿色|蓝色|黑色|银色|白色|灰色|黄色|紫色|红黄|红黑|黑蓝"
    return [
        re.sub(colors, "", feature).strip()
        for feature in profile["identity_features"]
        if not any(token in feature for token in ("字样", "标签文字", "商标"))
    ]


def appearance_neutral_name(profile: dict[str, Any]) -> str:
    return re.sub(
        r"^(?:青绿色|深绿色|深蓝色|紫粉色|红色|绿色|蓝色|黑色|银色|白色|灰色|黄色|紫色|红黄|红黑|黑蓝)\s*",
        "",
        profile["name"],
    )


WEATHERING = {
    "painted_steel": "明确呈现与当前废钢堆相称的报废钢件：可见钢主体主要为深褐锈、灰褐尘土和旧油泥，原漆仅作约10–20%的暗淡碎片残留；此为合成外观目标，不是实测腐蚀面积。裸钢有不规则锈层、点蚀和接缝积锈，大片剥漆，不保留完整红/蓝/绿大色块或亮漆高光。不能只加几个锈点。锈纹沿材质、磕碰和水路分布，不均匀喷成橙色。塑料旋钮/标牌不生铁锈。",
    "painted_castiron": "报废铸铁件的可见外壳明显氧化、漆膜剥落，深褐锈层与铸造粗糙度相互叠加，凹槽积尘油泥，底脚、孔边及螺栓有锈蚀；原色只留下暗淡残漆。受力面可局部擦亮但不能形成干净的新品反光，禁止仅在角落添几处轻微磨痕。",
    "stainless_steel": "呈现长期弃置的不锈钢旧件：镜面高光被灰尘、油膜、水垢和不均匀灰褐污层压低，表面有密集细擦伤、暗斑，焊缝/钢制紧固件可有局部锈迹或铁污染。主体仍为不锈钢，不把全部表面变成碳钢红锈；禁止均匀厚红锈。绝不保留干净银白新品和摄影棚灯条反射。塑料手柄只污旧。",
    "cast_steel": "报废铸钢阀门主体和法兰明显呈深褐/红褐氧化、油泥与积尘，螺栓孔和接缝积锈；银白新金属面改为失去光泽的旧钢材，蓝手轮褪色、剥漆，保留暗淡蓝色身份线索。保留厚法兰、曲面及真实接缝，不套用同一锈贴图，不仅给螺栓点几处轻锈。",
    "aerosol_can": "报废薄壁气雾罐应明显污旧：印刷褪色磨花、标签残损，表面附着灰褐粉尘和油污，钢制罐在掉漆、划伤、卷边处有清楚的不规则氧化/锈斑。保留弱化的主色与结构线索，禁止亮丽完整广告罐和可读新品印刷。塑料罩盖/喷头只积灰黄化擦伤、不生铁锈，薄壁结构不能被画成厚铸铁。",
    "fire_extinguisher": "报废灭火器筒体明显褪色、大片擦伤掉漆，裸钢有清楚可辨的褐色锈蚀、接缝流痕和灰尘油泥；红/绿原漆仅保留暗淡旧色块，标签污损模糊，不能保留亮丽洁净的新筒体。金属握把和连接件可锈蚀，橡胶软管只沾灰油并自然弯曲，不生铁锈，不新增喷射烟雾。",
    "painted_steel_chrome": "报废液压钢缸外壳明显剥漆、锈蚀、油泥和积尘，钢耳环/油口凹处有深褐锈与污垢，原漆暗淡残留。镀铬杆有旧油膜、细擦痕和局部暗斑，只保留弱金属反光，不整根涂成橙锈；禁止干净亮漆缸筒、亮银镜面活塞杆和新品黄标。保持缸筒厚度及机械连接。",
    "mixed_shock": "报废减震器的钢弹簧和钢端件有明显掉漆、褐锈和油灰，弹簧缝积尘；铝件/镀铬杆有灰暗污膜、氧化擦伤，只有微弱反射。原有彩色残漆降低饱和度；橡胶护套和软管积灰油污、轻微老化但不生铁锈。不能保留崭新的彩色弹簧、亮黑新品外壳；保持圈距和连接。",
    "air_shock": "废旧空气避震以明显灰褐积尘、油泥、涂层擦伤、铝表面灰白氧化表现旧化，钢耳件和螺栓有锈斑。黑色气室呈灰暗哑光而非新品亮黑；调节钮仅保留暗淡颜色线索，禁止厚铁锈覆盖铝气室，也不增加螺旋弹簧。",
    "aluminium_steel": "报废气缸铝壳明显灰暗、灰白氧化、密集擦伤并覆盖不均匀油泥积尘；钢拉杆、螺栓、安装端有清楚锈斑。活塞杆有旧油膜和弱反光，禁止干净的银亮铝型材和鲜明新品贴标，禁止把铝壳和杆统一涂成红锈。保持铝/钢/镀铬部件的材质差异。",
    "plastic_pp": "废弃白色PP罐明显积灰、灰褐油污、黄化和擦伤，盖纹及接缝积污，反射变为旧塑料的暗哑漫反射，不能只点几处轻污后仍像白色新品。没有红褐铁锈，不变成金属罐，不无故烧熔或大面积开裂；污损色与附近废料相称。",
}


def scene_context_prompt(context: dict[str, Any] | None) -> str:
    labels = {
        "lighting": "当前光照与方向",
        "tone": "当前局部色调",
        "contrast": "当前明暗和反射强度",
        "capture_style": "当前成像风格与噪声",
        "surface_condition": "附近废料的旧化状态",
        "support_and_occlusion": "支撑面和前景遮挡",
    }
    context = context or {}
    parts = [
        f"{label}：{str(context[key])[:260]}"
        for key, label in labels.items()
        if isinstance(context.get(key), str) and context[key].strip()
    ]
    if not parts:
        return "【背景适配】直接观察本张原监控图，按目标附近钢件的明暗、色温、旧化、反射、噪声和锐度重画物品；无法确定光源时不要添加摄影棚高光或新投影方向。"
    return "【背景适配，来源为VLM观察而非物理测量】" + "；".join(parts)


def review_requirements() -> str:
    return (
        "分别给weathering_score、lighting_score、occlusion_score、style_score、"
        "background_preservation_score、identity_score（0–100）。"
        "旧化检查可见材质是否明确报废且与附近废料相称；铁/碳钢须有清楚可辨的锈蚀，"
        "塑料、铝、不锈钢主要检查积污、氧化、擦伤和弱反射，不能要求全部铁锈。"
        "若仍是新漆、亮银干净新品、摄影棚高光或只有装饰性几个锈点，new_appearance_detected=true且weathering_score不得高于60。"
        "检查物品是否有真实承重支点、至少部分轮廓被原有前景钢件截断、贴合接触阴影；"
        "完整展品摆在堆顶、悬浮、边缘光晕、轮廓全露或独立黑描边判pasted_appearance_detected=true且occlusion_score不得高于60。"
        "照明检查局部亮度、高光方向/软硬、阴影和邻近反射；风格检查饱和度、对比、锐度和噪声是否相称，不能用柔化全图掩盖贴图感。"
        "背景保持需对照原图检查目标区域外结构和文字；身份检查该件的主体和连接，不以生造铭文代替结构。"
        "另给object_present（布尔）、new_appearance_detected（布尔）、pasted_appearance_detected（布尔）、"
        "material_violation（布尔）、weathering_fix、lighting_fix、occlusion_fix、style_fix、background_fix、identity_fix。"
        "全部六项必须高于85、存在目标且三个缺陷标志均为false才达到模型复核标准。"
        "分数必须根据看见的图像给出，不能因提示词写了要求就给高分；模型复核通过仍需人工看图。"
    )


def quality_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    scores: dict[str, float] = {}
    reasons = []
    for key in FUSION_SCORE_AXES:
        try:
            score = float(evaluation.get(key, 0))
            if not math.isfinite(score):
                score = 0.0
        except (TypeError, ValueError):
            score = 0.0
        score = min(100.0, max(0.0, score))
        scores[key] = score
        if score <= QUALITY_THRESHOLD:
            reasons.append(f"{key}_not_above_85")
    if evaluation.get("object_present") is not True:
        reasons.append("object_presence_not_confirmed")
    for key in (
        "new_appearance_detected",
        "pasted_appearance_detected",
        "material_violation",
    ):
        if evaluation.get(key) is not False:
            reasons.append(key)
    return {
        "passed": not reasons,
        "scores": scores,
        "minimum_score": min(scores.values()),
        "reasons": reasons,
        "evidence_type": "vlm_visual_opinion_not_human_acceptance",
    }


def edit_region(
    xy: list[float], ratio: float, image_size: tuple[int, int], cross_ratio: float
) -> list[float]:
    """Return a normalized interactive edit hint, including support and shadow context."""
    width, height = image_size
    box_w = min(0.72, max(0.14, ratio * 1.55 + 0.03))
    box_h = min(
        0.60, max(0.16, ratio * (0.60 + min(1.0, cross_ratio)) * width / height)
    )
    cx = min(0.96 - box_w / 2, max(0.04 + box_w / 2, float(xy[0])))
    cy = min(0.84 - box_h / 2, max(0.10 + box_h / 2, float(xy[1])))
    return [
        round(cx - box_w / 2, 5),
        round(cy - box_h / 2, 5),
        round(cx + box_w / 2, 5),
        round(cy + box_h / 2, 5),
    ]


def region_prompt(region: list[float] | None) -> str:
    if not region:
        return ""
    x1, y1, x2, y2 = region
    return (
        f"【局部重建区域】归一化坐标x={x1:.3f}–{x2:.3f}、y={y1:.3f}–{y2:.3f}，"
        f"目标位于区域中心附近（{(x1 + x2) / 2:.1%}，{(y1 + y2) / 2:.1%}）。"
        "此框包含物品、接触阴影和少量已有前景钢件，不是物品的外接框；物品可见长度仍按本轮设定比例。"
        "允许在框内重建接触处和遮挡边缘，使原有钢条/钢板自然盖住物品部分轮廓；保持框外场景、时间戳和机位字样。"
    )
