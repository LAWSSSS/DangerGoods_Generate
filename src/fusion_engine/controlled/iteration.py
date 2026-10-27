"""Bounded image/VLM rounds with strictly append-only, stateless request prompts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from fusion_engine.artifacts import write_json
from fusion_engine.controlled.job import ControlledJob
from fusion_engine.errors import ConfigError


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf8")).hexdigest()


@dataclass(frozen=True)
class IterationPolicy:
    enabled: bool = True
    max_rounds: int = 3
    stop_on_visual_accept: bool = False
    continue_on_unknown: bool = True

    @classmethod
    def from_job(cls, job: ControlledJob, override: int | None = None) -> IterationPolicy:
        values = job.data.get("iteration", {})
        rounds = values.get("max_rounds", 3) if override is None else override
        if isinstance(rounds, bool) or not isinstance(rounds, int) or not 1 <= rounds <= 5:
            raise ConfigError("迭代轮数须为1至5的整数；默认3轮，包含首次生成")
        return cls(
            values.get("enabled", True),
            rounds,
            values.get("stop_on_visual_accept", False),
            values.get("continue_on_unknown", True),
        )

    def plan(self, job: ControlledJob, *, generate: bool) -> dict:
        active = generate and self.enabled
        vlm_per_round = (
            2 if job.data.get("candidate_geometry", {}).get("method") == "vlm_seeded_grabcut" else 1
        )
        budget = job.data["budget"]
        effective = (
            min(
                self.max_rounds,
                budget["max_image_calls"],
                budget["max_candidates"],
                budget["max_vlm_calls"] // vlm_per_round,
            )
            if active
            else 1
        )
        if active and effective < 1:
            raise ConfigError("迭代至少需要一轮生图和审阅预算；未绕过预算生成无审阅图片")
        return {
            "enabled_for_this_run": active,
            "requested_max_rounds": self.max_rounds,
            "effective_max_rounds": effective,
            "rounds_include_initial_generation": True,
            "vlm_calls_reserved_per_round": vlm_per_round,
            "prompt_mode": "append_only_full_history",
            "previous_candidate_included": active,
            "original_references_and_full_background_resent_each_round": active,
            "stop_on_visual_accept": self.stop_on_visual_accept,
            "continue_on_unknown": self.continue_on_unknown,
            "budget_limited": active and effective < self.max_rounds,
        }


def feedback_from_record(
    record: dict, max_patch_chars: int = 420, *, source_stage: str = "master"
) -> dict:
    """Keep all located FAIL/UNKNOWN evidence; do not invent a defect from failed JSON."""
    review = record.get("vlm_review", {}).get("result")
    observation = (record.get("candidate_observation") or {}).get("result")
    processing = record["processing"]
    messages, defects = [], []
    if review:
        for item in review.get("defects", []):
            defects.append(item)
            status = "模型认为存在" if item["status"] == "FAIL" else "尚待核实，不可当成已知缺陷"
            messages.append(
                f"{item['object_id']} / {item['kind']} / {status}，原图相对ROI {item['region']}："
                f"观察依据：{item['evidence']}。建议：{item['instruction']}"
            )
        messages += ["本轮视觉问题：" + value for value in review.get("issues", [])]
        if review.get("fix_prompt"):
            messages.append("本轮整体修正建议：" + review["fix_prompt"])
        messages += ["不确定性：" + value for value in review.get("uncertainties", [])]
    if observation:
        for obj in observation.get("objects", []):
            for status_key, evidence_key, label in (
                ("support_status", "support_evidence", "承托"),
                ("identity_status", "identity_evidence", "身份/连接"),
                ("occlusion_status", "occlusion_evidence", "遮挡"),
                ("color_status", "color_evidence", "光色"),
            ):
                if obj.get(status_key) == "FAIL":
                    messages.append(
                        f"候选观察 {obj['id']} / {label}："
                        + obj.get(evidence_key, "需对照参考复核")
                    )
            if obj.get("fix_prompt"):
                messages.append(f"候选观察 {obj['id']} 的建议：" + obj["fix_prompt"])
    geometry = processing.get("candidate_geometry", {})
    hard_fail = [
        name for name, value in geometry.get("hard_checks", {}).items() if value["status"] == "FAIL"
    ]
    if hard_fail:
        messages.append(
            "程序测得的二维风险："
            + "、".join(hard_fail)
            + "。核对落位、可见完整轮廓和真实接触，不以黑影补缝或裁断物体。"
        )
    if processing.get("status") != "APPLIED":
        messages.append(
            "本轮局部算法处理未完成；这不证明物体必定缺件。请依据原参考和真实承载面重生成，保留原背景结构/纹理，避免黑底、白底和钢材融入主体；不要把分析失败当作已修复。"
        )
    usable = bool(review or observation)
    if not usable:
        messages.append(
            "本轮未获得可解析的模型视觉意见，没有新增已知视觉事实。下一轮严格重新执行原需求和历史具体修正，不编造缺陷，不把UNKNOWN当成通过。"
        )
    elif review and review.get("recommendation") == "accept" and not review.get("issues"):
        messages.append(
            "本轮模型未指出新的明确问题。下一轮保持已经正确的结构、位置和旧化，仅按原背景复核照明与成像，不为制造变化破坏正确部分；模型意见不代表人工验收。"
        )
    raw_only = (
        source_stage == "aligned_raw"
        or record.get("postprocess_integrity", {}).get("status") == "FAIL"
    )
    fixes = [
        v.get("fix_prompt", "").strip()
        for v in (({} if raw_only else review or {}), observation or {})
    ]
    if raw_only:
        fixes += [o.get("fix_prompt", "").strip() for o in (observation or {}).get("objects", [])]
    appendix = next((v for v in fixes if v and len(v) <= max_patch_chars), "")
    appendix_source = "complete_model_fix_prompt" if appendix else "bounded_contract_fallback"
    if not appendix:
        recipes = {
            "identity_missing_part": "补全参考关键部件和连接，保留真实开口。",
            "wrong_pose_scale": "按指定原区域校正刚体姿态与尺度。",
            "no_contact": "承重实体直接接触原承载面，不能用黑影托起。",
            "occlusion_inversion": "修正交叉处前后关系，保留原钢件连续边缘。",
            "mask_halo": "保留细连接和孔内背景，去白边黑边，不做宽羽化。",
            "lighting_specular_mismatch": "按现场重表现受光和粗糙反射，去独立棚灯高光。",
            "color_cast": "统一现场曝光白平衡，保留低饱和残漆与锈蚀本色。",
            "cast_shadow_missing": "只在核实的可见接收面补合理投影，别画整圈黑底。",
            "camera_blur_mismatch": "匹配同深度清晰度、噪声与压缩，保留可辨结构。",
            "background_changed": "恢复原背景纹理与叠字，只改允许的新物体及交互。",
        }
        selected = []
        for defect in [] if raw_only else defects:
            short = recipes.get(defect["kind"]) if defect["status"] == "FAIL" else None
            if (
                short
                and short not in selected
                and len("".join(selected) + short) <= max_patch_chars
            ):
                selected.append(short)
        appendix = "".join(selected) or (
            "选定的是原始候选；后处理误删不作为原物体缺件。保留已有正确结构、连接和真实孔洞，继续执行原需求，不能凭空补件。"
            if raw_only
            else "继续执行原需求和历史修正；本轮没有可用的新短指令，不编造缺陷或宣称通过。"
        )
    # Full observations stay in the audit. Only a complete short instruction is sent;
    # no old prompt or instruction is ever sliced to fit the provider's limit.
    return {
        "source_candidate_id": record["id"],
        "source_master_sha256": record["master_sha256"],
        "source_raw_sha256": record["raw_sha256"],
        "usable_model_feedback": usable,
        "review_recommendation": (review or {}).get("recommendation", "unknown"),
        "messages": list(dict.fromkeys(v.strip() for v in messages if v.strip())),
        "defects": defects,
        "unusable_localizations": record.get("vlm_review", {})
        .get("validation_audit", {})
        .get("rejected_defects", []),
        "generation_appendix": appendix,
        "generation_appendix_source": appendix_source,
        "full_review_messages_preserved_in_audit_not_duplicated_in_prompt": True,
        "feedback_is_model_proposal_not_ground_truth": True,
        "generation_feedback_input_stage": source_stage,
        "postprocess_only_defects_excluded_from_generation_fix": raw_only,
    }


def may_stop_accepted(record: dict) -> bool:
    if record["quality"]["status"] == "REJECTED":
        return False
    if record.get("vlm_review", {}).get("result", {}).get("recommendation") != "accept":
        return False
    if record["processing"].get("status") != "APPLIED":
        return False
    relevant = {
        k: v
        for k, v in record["quality"]["hard_checks"].items()
        if k in ("background_lock", "original_canvas_size", "v3_saved_layers_contract")
        or k.startswith(("candidate_", "receiver_contact_"))
    }
    return bool(relevant) and all(v["status"] == "PASS" for v in relevant.values())


@dataclass
class PromptJournal:
    base: str
    current: str = field(init=False)
    requests: list[dict] = field(default_factory=list)
    feedback: list[dict] = field(default_factory=list)
    _last_request_prompt: str = field(default="", init=False, repr=False)
    max_characters: int | None = field(default=None, init=False)
    patch_budget: int = field(default=420, init=False)

    def __post_init__(self):
        self.current = self.base

    @staticmethod
    def block(
        appendix: str, completed_round: int, background_index: int, parent_description: str = ""
    ) -> str:
        role = parent_description or "上轮候选"
        return (
            f"\n\n【第{completed_round}轮反馈→第{completed_round + 1}轮追加】\n"
            "前面全部原需求和历史修正继续有效，已修好部分保持。"
            f"本轮图号更新：图1至图{background_index - 1}是物品参考，图{background_index}是完整原背景，"
            f"图{background_index + 1}是{role}并替代初轮引导。只纠正既有物体，不新增实例，不继承错误背景漂移。\n"
            + appendix
            + "\n保留完整画幅、真实承托、连接及孔洞；正常透视不要强改。反馈仍需重新审阅，不代表已修复。\n"
        )

    def configure_envelope(
        self,
        limit: int | None,
        rounds: int,
        background_index: int,
        *,
        parent_description_budget: int = 0,
    ) -> None:
        if self.requests:
            raise ConfigError("不能在已发送请求后重写提示词历史预算")
        self.max_characters = limit
        if limit is not None and len(self.base) > limit:
            raise ConfigError("原始提示词已超过模型字符上限；未截断或发送")
        if limit is not None and rounds > 1:
            overhead = max(
                len(self.block("", n, background_index, "x" * parent_description_budget))
                for n in range(1, rounds)
            )
            available = (limit - len(self.base)) // (rounds - 1) - overhead
            if available < 80:
                raise ConfigError(
                    "原始提示词未给全部迭代预留足够空间；请显式改短首轮模板，不能截断历史"
                )
            self.patch_budget = min(420, available)

    def append_feedback(
        self, feedback: dict, *, completed_round: int, background_index: int
    ) -> None:
        previous = self.current
        appendix = feedback["generation_appendix"]
        if len(appendix) > self.patch_budget:
            raise ConfigError("新增修正超出预留预算；未截断旧提示词或新指令")
        block = self.block(
            appendix,
            completed_round,
            background_index,
            feedback.get("selected_parent_description", ""),
        )
        self.current = previous + block
        if self.max_characters is not None and len(self.current) > self.max_characters:
            raise ConfigError("累积提示词超过模型上限，已拦截，未发送给服务端截断")
        if not self.current.startswith(previous) or not self.current.startswith(self.base):
            raise ConfigError("累积提示词前缀丢失；禁止发送删减历史要求的请求")
        self.feedback.append(
            {
                "completed_round": completed_round,
                **feedback,
                "previous_prompt_sha256": text_sha256(previous),
                "next_prompt_sha256": text_sha256(self.current),
                "appended_text": block,
                "previous_prompt_is_exact_prefix": True,
            }
        )

    def record_request(self, round_index: int) -> dict:
        if (
            round_index != len(self.requests) + 1
            or not self.current.startswith(self.base)
            or not self.current.startswith(self._last_request_prompt)
        ):
            raise ConfigError("提示词历史不连续或原始要求丢失，停止发送")
        record = {
            "round": round_index,
            "prompt_sha256": text_sha256(self.current),
            "base_prompt_sha256": text_sha256(self.base),
            "characters": len(self.current),
            "cumulative_feedback_rounds": len(self.feedback),
            "previous_prompt_is_exact_prefix": True,
        }
        self.requests.append(record)
        self._last_request_prompt = self.current
        return record

    def save(self, root: Path) -> None:
        write_json(
            root / "prompts/accumulation.json",
            {
                "policy": "append_only_no_truncation_no_implicit_model_memory",
                "maximum_prompt_characters": self.max_characters,
                "reserved_feedback_characters_per_round": self.patch_budget,
                "base_prompt_sha256": text_sha256(self.base),
                "requests": self.requests,
                "feedback": self.feedback,
            },
        )
