#!/usr/bin/env python3
"""Replay both recovered 2026-10-12 morning pipelines for three full rounds."""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import inspect
import importlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import time

from PIL import Image, ImageDraw, ImageFilter, ImageFont
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
app = None
RECOVERED = ROOT / "output/version_lookup_20261012/editor_recovery/snapshots"
MODES = ("no_reference", "with_reference")
MODE_LABELS = {"no_reference": "上午前段：仅背景", "with_reference": "上午后段：背景＋主参考"}

def load_pipeline(mode, snapshot_root=RECOVERED):
    global app
    source = snapshot_root / mode / "generate_aliyun.py"
    name = "morning_pipeline_" + mode
    spec = importlib.util.spec_from_file_location(name, source)
    app = importlib.util.module_from_spec(spec)
    sys.modules[name] = app
    spec.loader.exec_module(app)
    app.load_bailian_raw = lambda: app.load_yaml_config(ROOT / "config_bailian.yaml")
    return app

def configured(mode, snapshot_root=RECOVERED):
    load_pipeline(mode, snapshot_root)
    with redirect_stdout(io.StringIO()):
        cfg = app.load_app_config()
    cfg.backgrounds_dir = ROOT / "input/backgrounds"
    cfg.objects_dir = ROOT / "input/objects"
    if (cfg.model, cfg.vlm_model) != EXPECTED_MODELS:
        raise RuntimeError("Recovered models do not match historical metadata")
    return cfg

OBJECTS = [
    ("sealant", "密封胶罐", "密封胶罐.jpg"),
    ("oil", "油缸", "油缸.jpg"),
    ("shock", "减震器", "液压减震器.jpg"),
]
SEED_OFFSETS = {"sealant": 0, "oil": 10, "shock": 5}
EXPECTED_COMMIT = "ec23f39c3851f1dd7f61517020d10c5129701b73"
EXPECTED_MODELS = ("wan2.7-image", "doubao-seed-2.1-pro")

BG_PREFIX = "2026_08_06_36_6_jianliao1_50_houjian_18#33_zhongfei3_15_zhongfei2_8#33_jianliao2_8#33_"
SOURCE_FILES = ["generate_aliyun.py", "config.yaml", "prompts/prompt_templates.json", "prompts/negative_prompt.txt"]
EXPECTED_HASHES = {"no_reference": "c15b1f7495174e00b6c156a9f1575a1e5766b8a07cd74c6b0ee0d910f46bac29", "with_reference": "f7e0260445880661ee7492cc6f6734f3c3f8e615f96a92888ca730b5b27fdf5e"}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


@contextmanager
def ledger_lock(run):
    with (run / "requests.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def reserve(run, kind, job, stage):
    with ledger_lock(run):
        path = run / "requests.json"
        ledger = json.loads(path.read_text())
        if kind == "wan" and sum(r["kind"] == "wan" for r in ledger["requests"]) >= ledger["limit"]:
            raise RuntimeError("wan_request_limit_exhausted")
        if kind == "wan" and any(r["kind"] == kind and r["job"] == job and r["stage"] == stage for r in ledger["requests"]):
            raise RuntimeError("submitted_round_must_not_be_resubmitted")
        entry = dict(id=len(ledger["requests"]) + 1, kind=kind, job=job, stage=stage,
            status="submitted", submitted_at=now())
        ledger["requests"].append(entry)
        save(path, ledger)
        return entry["id"]


def finish(run, request_id, **fields):
    with ledger_lock(run):
        path = run / "requests.json"
        ledger = json.loads(path.read_text())
        next(r for r in ledger["requests"] if r["id"] == request_id).update(fields, finished_at=now())
        save(path, ledger)


def clean_messages(messages):
    cleaned = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            content = [dict(type="image_url", encoded_sha256=hashlib.sha256(p["image_url"]["url"].encode()).hexdigest())
                if p.get("type") == "image_url" else p for p in content]
        cleaned.append(dict(role=message["role"], content=content))
    return cleaned


def force_three_rounds():
    """Suppress only the baseline's early exit, without changing its evaluation."""
    source = textwrap.dedent(inspect.getsource(app.VLMAnalyzer.run_optimization_loop))
    tree = ast.parse(source)
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.If)
        and ast.unparse(n.test) == "not need_retry"]
    if len(matches) != 1:
        raise RuntimeError("baseline early-exit condition changed")
    matches[0].test = ast.Constant(False)
    ast.fix_missing_locations(tree)
    scope = app.__dict__
    exec(compile(tree, app.__file__, "exec"), scope)
    app.VLMAnalyzer.run_optimization_loop = scope["run_optimization_loop"]
    return ast.unparse(tree) + "\n"


def session():
    # Explicitly submit each billed request once; no adapter POST retries.
    s = requests.Session()
    s.verify = False
    return s


def compressed_original(path, longest=1600):
    target = Path(path)
    with Image.open(target) as image:
        image = image.convert("RGB")
        image.thumbnail((longest, longest), Image.Resampling.LANCZOS)
        b = io.BytesIO()
        image.save(b, format="JPEG", quality=92)
    import base64
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def strict_old_evaluation(raw, request_ok):
    return bool(request_ok and raw.get("size_judgment") in {"too_large", "acceptable", "too_small"}
        and isinstance(raw.get("confidence"), (int, float)) and not isinstance(raw.get("confidence"), bool)
        and 0 <= raw["confidence"] <= 1 and isinstance(raw.get("quality_score"), (int, float))
        and not isinstance(raw.get("quality_score"), bool) and 0 <= raw["quality_score"] <= 1
        and raw.get("fusion_quality") in {"very bad", "bad", "ok", "good", "very good"}
        and "默认评估" not in str(raw.get("reason", ""))
        and isinstance(raw.get("visible_ratio"),(int,float)) and not isinstance(raw.get("visible_ratio"),bool) and 0 <= raw["visible_ratio"] <= 1)


def audit_pass(raw):
    return bool(raw.get("object_present") is True and type(raw.get("new_instance_count")) is int and raw["new_instance_count"] == 1
        and raw.get("reference_match") == "match" and raw.get("placement_match") is True
        and raw.get("support_judgment") == "stable" and raw.get("occlusion_match") is True
        and raw.get("integration_judgment") == "natural" and raw.get("size_judgment") == "acceptable"
        and isinstance(raw.get("confidence"), (int, float)) and not isinstance(raw.get("confidence"), bool)
        and .65 <= raw["confidence"] <= 1 and isinstance(raw.get("evidence"), dict)
        and all(isinstance(raw["evidence"].get(k), str) and raw["evidence"][k].strip()
            for k in ("identity", "placement", "support", "occlusion", "size"))
        and raw.get("uncertainties") == [])


def worker(run_string, job):
    # A pool process can handle several cases; start each with pristine baseline
    # functions before installing case-local logging and the single loop change.
    run = Path(run_string)
    load_pipeline(job["mode"], run / "source_snapshot")
    folder = run / "cases" / job["id"]
    folder.mkdir(parents=True, exist_ok=True)
    result = dict(job, status="running", rounds=[], started_at=now(), forced_rounds=3,
        baseline_original_reference_is_not_model_input=job["mode"] == "no_reference", baseline_vlm_uses_object_reference=False)
    save(folder / "result.json", result)
    with (folder / "run.log").open("a", encoding="utf-8") as log, redirect_stdout(log):
        try:
            with redirect_stdout(io.StringIO()):
                cfg = app.load_app_config()
            if (cfg.model, cfg.vlm_model) != EXPECTED_MODELS:
                raise ValueError("model selection changed")
            cfg.backgrounds_dir = ROOT / "input/backgrounds"
            cfg.objects_dir = ROOT / "input/objects"
            cfg.seed = job["seed"]
            cfg.max_images = 1
            cfg.max_optimization_rounds = 3
            app.random.seed(job["seed"])
            force_three_rounds()
            templates = app.load_prompt_templates(cfg.prompt_templates_path)
            negative = app.load_negative_prompt(cfg.negative_prompt_path)
            # Consume the same one-item selection draw as the historical CLI.
            catalog = app.get_all_object_images_with_classes(cfg.objects_dir, templates)
            item = next(i for i in catalog if i["path"].resolve() == Path(job["object_path"]).resolve())
            if item["class"] != job["object_class"]:
                raise RuntimeError("Historical item class mismatch")
            app.choose_weighted_object([item])
            result["historical_item_weight"] = item["weight"]
            size_data = app.calculate_optimal_size(Path(job["background_path"]), job["object_class"], cfg)
            size_state = {"ratio": size_data["target_ratio"]}
            base_prompt = app.random.choice(templates[job["object_class"]])
            result.update(configured_base_ratio=cfg.object_resize_scale, initial_size_calculation=size_data,
                models={"generation": cfg.model, "vlm": cfg.vlm_model},
                reference_paths=job["reference_paths"], reference_scope="selected_primary_only" if job["reference_paths"] else "background_only")
            state = {"stage": "opening", "vlm_ok": False}

            class LoggedAnalyzer(app.VLMAnalyzer):
                def _post_vlm(self, messages, max_tokens):
                    rid = reserve(run, "vlm", job["id"], state["stage"])
                    state["vlm_ok"] = False
                    save(folder / f"vlm_{rid:04d}_request.json", dict(model=cfg.vlm_model,
                        messages=clean_messages(messages), max_tokens=max_tokens, temperature=.3))
                    try:
                        response = session().post(cfg.vlm_api_url,
                            headers={"Authorization": "Bearer " + cfg.vlm_api_key, "Content-Type": "application/json"},
                            json=dict(model=cfg.vlm_model, messages=messages, max_tokens=max_tokens, temperature=.3),
                            timeout=cfg.timeout_seconds)
                        response.raise_for_status()
                        raw = response.json()
                        save(folder / f"vlm_{rid:04d}_response.json", raw)
                        content = raw["choices"][0]["message"]["content"]
                        if isinstance(content, list):
                            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
                        state["vlm_ok"] = True
                        finish(run, rid, status="returned", response_path=str(folder / f"vlm_{rid:04d}_response.json"))
                        return content
                    except Exception as exc:
                        finish(run, rid, status="failed", error=str(exc))
                        # Avoid the baseline's unrecorded SSL fallback POST.
                        raise RuntimeError(f"VLM transport failed: {exc}") from exc

                def compose_generation_prompt(self, *args, **kwargs):
                    state["stage"] = f"prompt_round_{len(result['rounds'])+1}"
                    return super().compose_generation_prompt(*args, **kwargs)

                def evaluate_generated_image(self, image_path, cls, prompt, target_ratio=None, size_spec=None, size_rounds=None):
                    state["stage"] = f"evaluation_round_{len(result['rounds'])}"
                    old = super().evaluate_generated_image(image_path, cls, prompt, target_ratio, size_spec, size_rounds)
                    row = result["rounds"][-1]
                    row["baseline_evaluation"] = dict(old)
                    row["baseline_evaluation_valid"] = strict_old_evaluation(old, state["vlm_ok"])
                    row["baseline_would_stop"] = old.get("size_judgment", "acceptable") not in {"too_large", "too_small"}
                    row["sharpness"] = sharpness(image_path)
                    state["stage"] = f"independent_audit_round_{row['round']}"
                    visible_roi(image_path, row["sampled_xy"], row["target_ratio"], folder / f"round{row['round']}_target.png")
                    content = [{"type": "text", "text": audit_prompt(job, row, app.object_scale_range(cfg, cls))}]
                    for p in [Path(job["object_path"]), Path(job["background_path"]), image_path, folder / f"round{row['round']}_target.png"]:
                        content.append({"type": "image_url", "image_url": {"url": compressed_original(p)}})
                    try:
                        raw_text = self._post_vlm([{"role": "user", "content": content}], 3200)
                        audited = self._extract_json(raw_text)
                        row["independent_audit"] = audited
                        row["independent_passed"] = audit_pass(audited)
                    except Exception as exc:
                        row.update(independent_audit={"error": str(exc)}, independent_passed=False)
                    save(folder / f"round{row['round']}_evaluation.json", row)
                    save(folder / "result.json", result)
                    return old

            def generated(prompt, round_num, previous_path, position_hint, orientation_hint):
                state["stage"] = f"generation_round_{round_num}"
                path = folder / f"round{round_num}.png"
                row = dict(round=round_num, status="submitted", image=str(path), prompt_core=prompt,
                    target_ratio=app.round_size_spec(cfg, job["object_class"], size_state["ratio"])["target_ratio"], previous_image=str(previous_path) if previous_path else None,
                    original_background=job["background_path"], position_hint=position_hint,
                    orientation_hint=orientation_hint)
                # The baseline's callback receives a textual position; its numeric sample
                # is saved by its sampling function below, without changing its result.
                row["sampled_xy"] = state["position"]["sampled_xy"]
                result["rounds"].append(row)
                save(folder / "result.json", result)
                full = app.build_prompt(job["object_class"], prompt, negative,
                    background_path=Path(job["background_path"]), position_hint=position_hint,
                    orientation_hint=orientation_hint, cfg=cfg, target_ratio=size_state["ratio"])
                row["size_spec"] = app.round_size_spec(cfg, job["object_class"], size_state["ratio"])
                kwargs = {"reference_paths": [Path(p) for p in job["reference_paths"]]} if job["mode"] == "with_reference" else {}
                api, labeled = app.generate_single_call(cfg, Path(job["background_path"]), full, previous_path, **kwargs)
                (folder / f"round{round_num}_prompt.txt").write_text(labeled, encoding="utf-8")
                row["full_prompt"] = labeled
                save(folder / f"round{round_num}_api.json", api)
                urls = app.extract_image_urls(api, cfg.image_protocol)
                if not urls:
                    raise RuntimeError("Wan returned no image URL")
                path.write_bytes(app.download_image(urls[0]))
                with Image.open(path) as im:
                    row["output_dimensions"] = list(im.size)
                row.update(status="returned", raw_image_sha256=digest(path))
                save(folder / "result.json", result)
                return path, api, labeled

            def safe_image_api(payload, active_cfg):
                if "seed" in payload["parameters"]:
                    raise RuntimeError("Historical morning Wan requests did not include a seed")
                rid = reserve(run, "wan", job["id"], state["stage"])
                row = result["rounds"][-1]
                row["request_id"] = rid
                row["request_parameters"] = dict(payload["parameters"])
                row["generation_image_inputs"] = [job["background_path"], *job["reference_paths"]]
                row["generation_input_hashes"] = [digest(p) for p in row["generation_image_inputs"]]
                save(folder / "result.json", result)
                try:
                    response = session().post(active_cfg.image_api_url,
                        headers={"Authorization": "Bearer " + active_cfg.image_api_key, "Content-Type": "application/json"},
                        json=payload, timeout=active_cfg.timeout_seconds)
                    response.raise_for_status()
                    raw = response.json()
                    save(folder / f"wan_{rid:04d}_response.json", raw)
                    finish(run, rid, status="returned", response_path=str(folder / f"wan_{rid:04d}_response.json"))
                    return raw
                except Exception as exc:
                    finish(run, rid, status="failed", error=str(exc))
                    raise

            app.call_image_api = safe_image_api
            app.get_session_with_ssl_bypass = session
            original_sample = app.sample_foreign_object_position
            def sampled(geometry):
                value = original_sample(geometry)
                state["position"] = value
                return value
            app.sample_foreign_object_position = sampled
            analyzer = LoggedAnalyzer(cfg)
            outcome = analyzer.run_optimization_loop(Path(job["object_path"]), Path(job["background_path"]),
                job["object_class"], base_prompt, generated, job["id"], size_state)
            save(folder / "history.json", outcome["history"])
            save(folder / "native_outcome.json", {k:v for k,v in outcome.items() if k not in ["history", "api_result"]}
                | {"last_image_path":str(outcome.get("last_image_path"))})
            result.update(status="completed", completed_at=now(), position=outcome["position_sample"],
                orientation=outcome["orientation_sample"], baseline_early_exit_suppressed=True,
                independent_audit_does_not_feed_back=True,
                passed_rounds=[r["round"] for r in result["rounds"] if r.get("independent_passed")],
                selected_round=len(result["rounds"]), selected_reason="morning pipeline keeps last generated round; forced three rounds",
                historical_stop_round=next((r["round"] for r in result["rounds"] if r.get("baseline_would_stop")),None),
                size_rounds=outcome["size_rounds"], judgment_history=outcome["judgment_history"], round_sharpness=outcome["round_sharpness"])
            selected=next(r for r in result["rounds"] if r["round"]==result["selected_round"])
            chosen=run/"selected"/(job["id"]+".png"); chosen.parent.mkdir(exist_ok=True)
            shutil.copyfile(selected["image"],chosen)
            result.update(selected_image=str(chosen),selected_image_sha256=digest(chosen))
        except Exception as exc:
            result.update(status="failed", error=str(exc), completed_at=now())
            if result["rounds"] and result["rounds"][-1]["status"] == "submitted":
                result["rounds"][-1].update(status="failed", error=str(exc))
        save(folder / "result.json", result)
    return result


def sharpness(path):
    with Image.open(path) as im:
        im = im.convert("L")
        im.thumbnail((960, 960))
        edge = im.filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1))
        data = list(edge.tobytes())
        mean = sum(data) / len(data)
        return sum((x - mean) ** 2 for x in data) / len(data)


def visible_roi(path, point, ratio, target):
    with Image.open(path) as im:
        half = max(.06, ratio * 2.5)
        x, y = point[0] * im.width, point[1] * im.height
        span = half * im.width
        im.crop((max(0, int(x-span)), max(0, int(y-span)), min(im.width, int(x+span)), min(im.height, int(y+span)))).save(target)


def audit_prompt(job, row, allowed):
    special = "密封胶罐参考中的左侧白罐已由用户确认是封闭盖，不是开口桶。" if job["object_class"] == "密封胶罐" else ""
    return ("你是独立的实图对照评估员。本次判断不反馈给下一轮生成。第1张原物参考，第2张固定原背景，第3张完整生成图，第4张计划位置局部；局部可能没有覆盖偏移的目标，整图仍是依据。"
        "只统计相比原背景新增的目标，原有废钢管路不能冒充。参考若有多件，应匹配其中一件完整实物，不能混合零件。允许环境光、灰尘、合理遮挡，不允许更换主体、主色、口部状态或连接关系。"
        "不能仅凭暗色认为有底，不能把画面下方当真实低处。被遮住的结构不得编造；无法辨认或无法确认支撑、深度判uncertain，不自动通过。" + special
        + f"目标{job['object_class']}，计划最终位置{row['sampled_xy']}，目标可见最长边/画面宽{row['target_ratio']:.4f}，旧方案允许范围{allowed}。"
        + '输出JSON：{"object_present":true|false|null,"new_instance_count":0,"bbox":null或归一化[x1,y1,x2,y2],"reference_match":"match|mismatch|uncertain",'
          '"placement_match":true|false|null,"support_judgment":"stable|unstable|uncertain","occlusion_match":true|false|null,'
          '"integration_judgment":"natural|pasted|uncertain","size_judgment":"acceptable|too_large|too_small|not_applicable|uncertain",'
          '"confidence":0.0,"evidence":{"identity":"原图依据","placement":"落点依据","support":"承托/限位依据","occlusion":"前景依据","size":"可见尺度依据"},"uncertainties":[],"reason":"具体问题"}')


def summarize(run):
    manifest = json.loads((run / "manifest.json").read_text())
    jobs = []
    for job in manifest["jobs"]:
        path = run / "cases" / job["id"] / "result.json"
        jobs.append(json.loads(path.read_text()) if path.exists() else dict(job, status="pending", rounds=[]))
    ledger = json.loads((run / "requests.json").read_text())
    report = dict(total_cases=len(jobs), completed=sum(j["status"] == "completed" for j in jobs),
        failed=sum(j["status"] == "failed" for j in jobs), pending=sum(j["status"] == "pending" for j in jobs),
        running=sum(j["status"] == "running" for j in jobs),
        wan_requests=sum(r["kind"] in {"wan", "wan_recovery"} for r in ledger["requests"]),
        wan_base_attempts=sum(r["kind"] == "wan" for r in ledger["requests"]),
        wan_recovery_attempts=sum(r["kind"] == "wan_recovery" for r in ledger["requests"]),
        wan_returned=sum(r["kind"] in {"wan", "wan_recovery"} and r["status"]=="returned" for r in ledger["requests"]),
        wan_failed=sum(r["kind"] in {"wan", "wan_recovery"} and r["status"]=="failed" for r in ledger["requests"]),
        vlm_requests=sum(r["kind"] == "vlm" for r in ledger["requests"]),
        images=sum(r.get("status") == "returned" for j in jobs for r in j["rounds"]),
        independently_passed=sum(bool(r.get("independent_passed")) for j in jobs for r in j["rounds"]),
        selected_branch=manifest["branch"],source_commit=manifest["head"],jobs=jobs, updated_at=now())
    save(run / "report.json", report)
    lines = ["# 10月12日上午两版本：3物品×2场景×3轮×2版本", "", "早段仅输入背景，晚段输入原背景＋选定主参考。两组均不输入上一轮生成图。使用历史记录的wan2.7-image与doubao-seed-2.1-pro；Python种子20261012，Wan未传seed。",
        "每组强制三轮，原评估的提前停止条件单独保存；独立评估不反馈到生图链路。", "",
        f"完整组 {report['completed']}/{len(jobs)}；失败组 {report['failed']}；完整图 {report['images']}/{len(jobs)*3}；Wan尝试 {report['wan_requests']}（返回{report['wan_returned']}、连接失败{report['wan_failed']}、恢复尝试{report['wan_recovery_attempts']}）；VLM请求 {report['vlm_requests']}。", "",
        "|版本|物品|场景|状态|第一轮|第二轮|第三轮|独立通过轮|", "|---|---|---|---|---|---|---|---|"]
    cards = []
    for j in jobs:
        values = []
        images = []
        for n in (1, 2, 3):
            r = next((r for r in j["rounds"] if r["round"] == n), {})
            ev = r.get("baseline_evaluation", {})
            value = ev.get("size_judgment", r.get("status", "待运行"))
            if ev and not r.get("baseline_evaluation_valid"):
                value = "评估失败"
            values.append(value)
            if r.get("status") == "returned":
                relative = Path(r["image"]).relative_to(run).as_posix()
                audit = r.get("independent_audit", {})
                images.append(f'<figure><a href="{relative}"><img src="{relative}"></a><figcaption>R{n}: {html.escape(value)} / 原貌 {html.escape(str(audit.get("reference_match", "未评估")))} / 支撑 {html.escape(str(audit.get("support_judgment", "未评估")))}</figcaption><p>{html.escape(str(audit.get("reason", "")))}</p></figure>')
        passed = j.get("passed_rounds", [])
        lines.append(f"|{MODE_LABELS[j['mode']]}|{j['object_class']}|{j['background_label']}|{j['status']}|{'|'.join(values)}|{passed or '无'}|")
        cards.append(f'<section><h2>{MODE_LABELS[j["mode"]]} · {html.escape(j["object_class"])} · {j["background_label"]} · {j["status"]}</h2><a href="cases/{j["id"]}/result.json">逐轮元数据</a><div class="rounds">'+''.join(images)+'</div></section>')
    (run / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三轮迭代对照</title><style>body{font:16px/1.6 system-ui;margin:24px;background:#f1f3f5;color:#192330}section{background:white;padding:20px;margin:20px 0}.rounds{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}figure{margin:0}img{width:100%;height:auto}p{font-size:14px}@media(max-width:900px){.rounds{grid-template-columns:1fr}}</style><h1>10月12日上午两版本对照</h1><p>完整原图均可点击。旧尺寸评估与独立身份/物理评估分开保存，不确定不自动通过。</p>'
    (run / "index.html").write_text(page + ''.join(cards) + '</html>', encoding="utf-8")
    return report


def historical_crosscheck():
    results=[]
    for mode in MODES:
        cfg=configured(mode)
        with redirect_stdout(io.StringIO()):
            templates=app.load_prompt_templates(cfg.prompt_templates_path)
            catalog=app.get_all_object_images_with_classes(cfg.objects_dir,templates)
        for cls,stamp in [("密封胶罐","20261012T015029Z" if mode=="no_reference" else "20261012T030849Z"),("油缸","20261012T015808Z" if mode=="no_reference" else "20261012T032339Z")]:
            md=next(p for p in (ROOT/"output/metadata").glob("*"+stamp+".json") if cls in p.name)
            old=json.loads(md.read_text());obj=Path(old["object_image"])
            item=next(i for i in catalog if i["path"].name==obj.name)
            app.random.seed(20261012)
            with redirect_stdout(io.StringIO()):
                app.choose_weighted_object([item])
                size=app.calculate_optimal_size(Path(old["background_image"]),cls,cfg)
                prompt=app.random.choice(templates[cls])
                opening=next(r["data"] for r in old["optimization_history"] if r.get("stage")=="opening")
                position=app.sample_foreign_object_position(app.geometry_from_vlm_reply(opening))
                orientation=app.sample_horizontal_orientation()
            results.append(dict(mode=mode,object_class=cls,metadata=str(md),weight=item["weight"],
                first_target_ratio_original=old["size_rounds"][0]["target_ratio"],first_target_ratio_recovered=size["target_ratio"],
                exact_ratio_match=old["size_rounds"][0]["target_ratio"]==size["target_ratio"],
                exact_prompt_match=old["base_prompt"]==prompt,exact_position_match=old["scrap_position"]==json.loads(json.dumps(position)),
                exact_orientation_match=old["object_orientation"]==orientation))
    return results

def prepare(run, seed):
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT).decode().strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    if head != EXPECTED_COMMIT:
        raise RuntimeError("Working checkout unexpectedly changed")
    hashes = {mode: {name: digest(RECOVERED / mode / name) for name in SOURCE_FILES} for mode in MODES}
    for mode in MODES:
        if hashes[mode]["generate_aliyun.py"] != EXPECTED_HASHES[mode]:
            raise RuntimeError("Recovered historical source changed")
    existing = run / "manifest.json"
    if existing.exists():
        saved = json.loads(existing.read_text())
        if saved["source_hashes"] != hashes or saved["seed"] != seed:
            raise ValueError("Existing experiment source/seed changed")
        return saved
    replay_check = historical_crosscheck()
    if not all(r["exact_ratio_match"] and r["exact_prompt_match"] and r["exact_position_match"] and r["exact_orientation_match"] for r in replay_check):
        raise RuntimeError("Recovered pipeline does not reproduce historical deterministic samples")
    jobs = []
    for mode in MODES:
        cfg = configured(mode)
        for index, (tag, cls, name) in enumerate(OBJECTS):
            obj = ROOT / "input/objects" / name
            for bg_label in ("44", "50"):
                bg = ROOT / "input/backgrounds" / f"{BG_PREFIX}{bg_label}.jpg"
                refs = app.collect_object_views(obj, cls, [{"class":cls,"path":obj}], cfg.max_object_views, None) if mode == "with_reference" else []
                jobs.append(dict(mode=mode, reference_paths=[str(p) for p in refs], reference_hashes=[digest(p) for p in refs],
                    id=f"{mode}_{index+1:02d}_{tag}_{bg_label}", object_class=cls, object_path=str(obj), object_hash=digest(obj),
                    background_path=str(bg), background_hash=digest(bg), background_label=bg_label, seed=seed))
    manifest = dict(branch=branch, head=head, seed=seed, created_at=now(), rounds=3, wan_limit=len(jobs)*3,
        source_hashes=hashes, jobs=jobs, pipeline="two recovered original morning pipelines",
        models={"generation":EXPECTED_MODELS[0],"vlm":EXPECTED_MODELS[1]},
        recorded_model_evidence="21 original morning photo metadata records plus Cursor source edit history",
        reference_scope="no reference in first mode; selected primary only in second mode; red oil cylinder excluded per earlier human instruction",
        historical_python_seed=20261012, wan_seed=None, historical_crosscheck=replay_check,
        adaptations=["suppress size-based early exit to run three rounds", "safe transport with no secret logging/POST retry",
            "independent audit outside feedback chain", "restrict later references to selected primary to preserve item identity", "add hydraulic shock absorber absent from historical morning batch"],
        historical_difference_note="Original later oil-cylinder batch included both blue and red oil photographs. This run uses blue only. Three rounds continue after original size-only stop. No new three-axis scores or best-round selection.")
    save(existing, manifest)
    save(run / "requests.json", dict(limit=len(jobs)*3, requests=[]))
    for mode in MODES:
        for name in SOURCE_FILES:
            target = run / "source_snapshot" / mode / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(RECOVERED / mode / name, target)
        load_pipeline(mode, run / "source_snapshot")
        (run / "source_snapshot" / mode / "forced_loop_source.py").write_text(force_three_rounds(), encoding="utf-8")
    shutil.copy2(ROOT / "scripts/test_morning_three_objects.py", run / "source_snapshot/test_morning_three_objects.py")
    shutil.copy2(ROOT / "output/version_lookup_20261012/editor_recovery/exact_content_edit_chain.json", run / "source_snapshot/editor_provenance.json")
    summarize(run)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        report = summarize(run)
        print(json.dumps({k:v for k,v in report.items() if k != "jobs"}, ensure_ascii=False))
        return
    manifest = prepare(run, args.seed)
    if args.prepare_only:
        print(json.dumps(dict(run=str(run), cases=len(manifest["jobs"]), wan_limit=manifest["wan_limit"], prepared=True), ensure_ascii=False))
        return
    jobs = []
    for job in manifest["jobs"]:
        if args.cases and job["id"] not in args.cases:
            continue
        record = run / "cases" / job["id"] / "result.json"
        if record.exists():
            # An interrupted, submitted round is intentionally not re-posted.
            continue
        if digest(job["object_path"]) != job["object_hash"] or digest(job["background_path"]) != job["background_hash"]:
            raise ValueError("Input content changed")
        jobs.append(job)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(worker, str(run), job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            report = summarize(run)
            print(json.dumps(dict(case=result["id"], status=result["status"], completed=report["completed"],
                images=report["images"], wan_requests=report["wan_requests"], error=result.get("error")), ensure_ascii=False), flush=True)
    summarize(run)


if __name__ == "__main__":
    main()
