#!/usr/bin/env python3
"""Run feat/realistic-object-scale's feedback chain for exactly three rounds per case."""
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
import generate_aliyun as app

OBJECTS = [
    ("sealant", "密封胶罐", "密封胶罐.jpg"),
    ("collector", "集油器", "集油器.jpg"),
    ("bearing", "轴承座", "a轴承座.jpg"),
    ("paint", "自喷漆罐", "自喷漆罐.jpg"),
    ("pesticide", "杀虫剂罐", "杀虫剂罐.jpg"),
    ("shock", "减震器", "液压减震器.jpg"),
    ("jack", "千斤顶", "千斤顶.jpg"),
    ("pneumatic", "气缸", "气缸.png"),
    ("extinguisher", "灭火器", "灭火器.jpg"),
    ("filter", "分离过滤器", "分离过滤器.jpg"),
    ("oil", "油缸", "油缸.jpg"),
    ("hydraulic", "液压缸", "液压缸.jpg"),
    ("valve", "球阀", "aDN350 球阀.jpg"),
    ("gas", "气瓶", "气瓶.jpg"),
    ("tank", "储罐", "储罐.jpg"),
]
BG_PREFIX = "2026_08_06_36_6_jianliao1_50_houjian_18#33_zhongfei3_15_zhongfei2_8#33_jianliao2_8#33_"
SOURCE_FILES = ["generate_aliyun.py", "config.yaml", "prompts/prompt_templates.json",
    "prompts/negative_prompt.txt", "requirements.txt", "scripts/test_prev_branch_iterations.py"]


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
        if kind == "wan" and sum(r["kind"] == "wan" for r in ledger["requests"]) >= 90:
            raise RuntimeError("ninety_wan_request_limit_exhausted")
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
        and ast.unparse(n.test) == "size_judgment == 'acceptable' or not should_retry"]
    if len(matches) != 1:
        raise RuntimeError("baseline early-exit condition changed")
    matches[0].test = ast.Constant(False)
    ast.fix_missing_locations(tree)
    scope = app.__dict__
    exec(compile(tree, str(ROOT / "generate_aliyun.py"), "exec"), scope)
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
        and "VLM 调用失败" not in str(raw.get("reason", "")))


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
    importlib.reload(app)
    run = Path(run_string)
    folder = run / "cases" / job["id"]
    folder.mkdir(parents=True, exist_ok=True)
    result = dict(job, status="running", rounds=[], started_at=now(), forced_rounds=3,
        baseline_original_reference_is_not_model_input=True)
    save(folder / "result.json", result)
    with (folder / "run.log").open("a", encoding="utf-8") as log, redirect_stdout(log):
        try:
            with redirect_stdout(io.StringIO()):
                cfg = app.load_app_config()
            if (cfg.model, cfg.vlm_model) != ("wan2.7-image", "qwen3.8-flash"):
                raise ValueError("model selection changed")
            cfg.seed = job["seed"]
            cfg.max_images = 1
            cfg.max_optimization_rounds = 3
            app.random.seed(job["seed"])
            force_three_rounds()
            templates = app.load_prompt_templates(cfg.prompt_templates_path)
            negative = app.load_negative_prompt(cfg.negative_prompt_path)
            base_prompt = app.random.choice(templates[job["object_class"]])
            size_state = {"ratio": app.sample_object_resize_scale(cfg, job["object_class"])}
            state = {"stage": "opening", "qwen_ok": False}

            class LoggedAnalyzer(app.VLMAnalyzer):
                def _post_vlm(self, messages, max_tokens):
                    rid = reserve(run, "qwen", job["id"], state["stage"])
                    state["qwen_ok"] = False
                    save(folder / f"qwen_{rid:04d}_request.json", dict(model=cfg.vlm_model,
                        messages=clean_messages(messages), max_tokens=max_tokens, temperature=.3))
                    try:
                        response = session().post(cfg.vlm_api_url,
                            headers={"Authorization": "Bearer " + cfg.vlm_api_key, "Content-Type": "application/json"},
                            json=dict(model=cfg.vlm_model, messages=messages, max_tokens=max_tokens, temperature=.3),
                            timeout=cfg.timeout_seconds)
                        response.raise_for_status()
                        raw = response.json()
                        save(folder / f"qwen_{rid:04d}_response.json", raw)
                        content = raw["choices"][0]["message"]["content"]
                        if isinstance(content, list):
                            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
                        state["qwen_ok"] = True
                        finish(run, rid, status="returned", response_path=str(folder / f"qwen_{rid:04d}_response.json"))
                        return content
                    except Exception as exc:
                        finish(run, rid, status="failed", error=str(exc))
                        # Avoid the baseline's unrecorded SSL fallback POST.
                        raise RuntimeError(f"VLM transport failed: {exc}") from exc

                def evaluate_generated_image(self, image_path, cls, prompt, target_ratio=None):
                    state["stage"] = f"evaluation_round_{len(result['rounds'])}"
                    old = super().evaluate_generated_image(image_path, cls, prompt, target_ratio)
                    row = result["rounds"][-1]
                    row["baseline_evaluation"] = dict(old)
                    row["baseline_evaluation_valid"] = strict_old_evaluation(old, state["qwen_ok"])
                    row["baseline_would_stop"] = old.get("size_judgment") == "acceptable" or not old.get("should_retry", False)
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
                    target_ratio=size_state["ratio"], previous_image=str(previous_path) if previous_path else None,
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
                api, labeled = app.generate_single_call(cfg, Path(job["background_path"]), full, previous_path)
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
                payload["parameters"]["seed"] = job["seed"]
                rid = reserve(run, "wan", job["id"], state["stage"])
                row = result["rounds"][-1]
                row["request_id"] = rid
                row["request_parameters"] = dict(payload["parameters"])
                row["generation_image_inputs"] = [job["background_path"]] + ([row["previous_image"]] if row["previous_image"] else [])
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
            result.update(status="completed", completed_at=now(), position=outcome["position_sample"],
                orientation=outcome["orientation_sample"], baseline_early_exit_suppressed=True,
                independent_audit_does_not_feed_back=True,
                passed_rounds=[r["round"] for r in result["rounds"] if r.get("independent_passed")])
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
        qwen_requests=sum(r["kind"] == "qwen" for r in ledger["requests"]),
        images=sum(r.get("status") == "returned" for j in jobs for r in j["rounds"]),
        independently_passed=sum(bool(r.get("independent_passed")) for j in jobs for r in j["rounds"]),
        selected_branch=manifest["branch"],source_commit=manifest["head"],jobs=jobs, updated_at=now())
    save(run / "report.json", report)
    lines = [f"# {manifest['branch']}：15类×2场景×3轮", "", "旧链路首轮仅背景，后续输入固定原背景＋上一轮生成图；原物照片只用于独立对照评估。",
        "每组强制三轮，原评估的提前停止条件单独保存；独立评估不反馈到生图链路。", "",
        f"完整组 {report['completed']}/30；失败组 {report['failed']}；完整图 {report['images']}/90；Wan尝试 {report['wan_requests']}（返回{report['wan_returned']}、连接失败{report['wan_failed']}、恢复尝试{report['wan_recovery_attempts']}）；Qwen请求 {report['qwen_requests']}。", "",
        "|物品|场景|状态|第一轮|第二轮|第三轮|独立通过轮|", "|---|---|---|---|---|---|---|"]
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
        lines.append(f"|{j['object_class']}|{j['background_label']}|{j['status']}|{'|'.join(values)}|{passed or '无'}|")
        cards.append(f'<section><h2>{html.escape(j["object_class"])} · {j["background_label"]} · {j["status"]}</h2><a href="cases/{j["id"]}/result.json">逐轮元数据</a><div class="rounds">'+''.join(images)+'</div></section>')
    (run / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三轮迭代对照</title><style>body{font:16px/1.6 system-ui;margin:24px;background:#f1f3f5;color:#192330}section{background:white;padding:20px;margin:20px 0}.rounds{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}figure{margin:0}img{width:100%;height:auto}p{font-size:14px}@media(max-width:900px){.rounds{grid-template-columns:1fr}}</style><h1>旧分支三轮迭代对照</h1><p>完整原图均可点击。旧尺寸评估与独立身份/物理评估分开保存，不确定不自动通过。</p>'
    (run / "index.html").write_text(page + ''.join(cards) + '</html>', encoding="utf-8")
    return report


def prepare(run, seed):
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT).decode().strip()
    if branch != "feat/realistic-object-scale":
        raise RuntimeError("Expected feat/realistic-object-scale")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    source_hashes = {p: digest(ROOT / p) for p in SOURCE_FILES}
    existing = run / "manifest.json"
    if existing.exists():
        saved = json.loads(existing.read_text())
        if saved["source_hashes"] != source_hashes or saved["seed"] != seed or saved["head"] != head:
            raise ValueError("Existing experiment source/seed changed")
        return saved
    jobs = []
    for index, (tag, cls, name) in enumerate(OBJECTS):
        obj = ROOT / "input/objects" / name
        if not obj.is_file():
            raise FileNotFoundError(obj)
        for bg_label in ("44", "50"):
            bg = ROOT / "input/backgrounds" / f"{BG_PREFIX}{bg_label}.jpg"
            if not bg.is_file():
                raise FileNotFoundError(bg)
            jobs.append(dict(id=f"{index+1:02d}_{tag}_{bg_label}", object_class=cls,
                object_path=str(obj), object_hash=digest(obj), background_path=str(bg),
                background_hash=digest(bg), background_label=bg_label, seed=seed+index))
    manifest = dict(branch=branch, head=head, seed=seed, created_at=now(), rounds=3, wan_limit=90,
        source_hashes=source_hashes, jobs=jobs, pipeline="unchanged_old_background_and_previous_output",
        adaptations=["suppress early exit to run three rounds", "safe transport with no secret logging/POST retry", "explicit Wan seed", "independent audit outside feedback chain"])
    save(existing, manifest)
    save(run / "requests.json", dict(limit=90, requests=[]))
    for name in SOURCE_FILES:
        target = run / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    (run / "forced_loop_source.py").write_text(force_three_rounds(), encoding="utf-8")
    summarize(run)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261014)
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
        print(json.dumps(dict(run=str(run), cases=len(manifest["jobs"]), wan_limit=90, prepared=True), ensure_ascii=False))
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
