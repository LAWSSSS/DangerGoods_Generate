#!/usr/bin/env python3
"""Create source-photo and three-round comparison boards without model calls."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import statistics

from PIL import Image, ImageDraw, ImageFont


def font(size):
    for path in ["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Light.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"]:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def place(canvas, path, box):
    with Image.open(path) as original:
        image = original.convert("RGB")
        image.thumbnail((box[2]-box[0], box[3]-box[1]), Image.Resampling.LANCZOS)
    canvas.paste(image, (int(box[0] + (box[2]-box[0]-image.width)/2),
        int(box[1] + (box[3]-box[1]-image.height)/2)))


def board(run, index, jobs):
    width, height = 1552, 756
    canvas = Image.new("RGB", (width, height), "#15212c")
    draw, title, label, small = ImageDraw.Draw(canvas), font(26), font(19), font(16)
    cls = jobs[0]["object_class"]
    draw.text((16, 10), f"{cls} · 上一个版本 0fded2f · 三轮迭代", font=title, fill="white")
    draw.text((16, 48), "旧链路不输入原物照片；此照片用于独立核对。每张生成图可在HTML中查看完整原字节。", font=small, fill="#c5d4e0")
    draw.text((16, 88), "原物主参考", font=label, fill="#c5d4e0")
    place(canvas, jobs[0]["object_path"], (16, 130, 222, 650))
    for yindex, job in enumerate(jobs):
        y = 88 + yindex * 320
        draw.text((244, y), "固定背景 " + job["background_label"], font=label, fill="white")
        for n in (1, 2, 3):
            x = 244 + (n-1) * 436
            row = next((r for r in job.get("rounds", []) if r["round"] == n), {})
            draw.text((x, y+28), f"R{n}", font=label, fill="white")
            if row.get("status") == "returned" and Path(row["image"]).is_file():
                place(canvas, row["image"], (x, y+56, x+418, y+291))
                ev, audit = row.get("baseline_evaluation", {}), row.get("independent_audit", {})
                size = ev.get("size_judgment", "未评估") if row.get("baseline_evaluation_valid") else "评估无效/待评估"
                text = f"尺寸 {size}  原貌 {audit.get('reference_match', '待评估')}"
                draw.text((x, y+295), text, font=small, fill="#6ae2ac" if row.get("independent_passed") else "#ffc77e")
            else:
                draw.rectangle((x, y+56, x+418, y+291), fill="#233240")
                draw.text((x+20, y+90), row.get("status", "尚未生成"), font=label, fill="#c5d4e0")
    output = run / "comparison_boards" / f"{index:02d}_{jobs[0]['id'].split('_')[1]}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    report = json.loads((run / "report.json").read_text())
    jobs = report["jobs"]
    completed = [j for j in jobs if j["status"] == "completed" and len(j["rounds"]) == 3]
    quality_deltas, sharpness_ratios = [], []
    trajectories = []
    for job in completed:
        a, b = job["rounds"][0], job["rounds"][-1]
        if a.get("baseline_evaluation_valid") and b.get("baseline_evaluation_valid"):
            quality_deltas.append(b["baseline_evaluation"]["quality_score"]-a["baseline_evaluation"]["quality_score"])
        if a.get("sharpness", 0) > 0:
            sharpness_ratios.append(b["sharpness"]/a["sharpness"])
        trajectories.append(dict(case=job["id"], object_class=job["object_class"], background=job["background_label"],
            size_judgments=[r.get("baseline_evaluation", {}).get("size_judgment") if r.get("baseline_evaluation_valid") else None for r in job["rounds"]],
            reference_matches=[r.get("independent_audit", {}).get("reference_match") for r in job["rounds"]],
            independent_passed=[r.get("independent_passed", False) for r in job["rounds"]]))
    by_round = []
    for n in (1, 2, 3):
        rows = [next(r for r in j["rounds"] if r["round"] == n) for j in completed]
        valid = [r for r in rows if r.get("baseline_evaluation_valid")]
        audits = [r.get("independent_audit", {}) for r in rows]
        by_round.append(dict(round=n, complete_cases=len(rows), valid_old_evaluations=len(valid),
            old_size_acceptable=sum(r["baseline_evaluation"].get("size_judgment") == "acceptable" for r in valid),
            old_mean_quality=statistics.mean(r["baseline_evaluation"]["quality_score"] for r in valid) if valid else None,
            reference_match=sum(a.get("reference_match") == "match" for a in audits),
            reference_mismatch=sum(a.get("reference_match") == "mismatch" for a in audits),
            reference_uncertain=sum(a.get("reference_match") == "uncertain" for a in audits),
            stable_support=sum(a.get("support_judgment") == "stable" for a in audits),
            natural_integration=sum(a.get("integration_judgment") == "natural" for a in audits),
            independently_passed=sum(r.get("independent_passed", False) for r in rows)))
    analysis = dict(completed_pairs=len(completed), by_round=by_round, trajectories=trajectories,
        mean_r3_minus_r1_old_quality=statistics.mean(quality_deltas) if quality_deltas else None,
        quality_pairs=len(quality_deltas), quality_improved=sum(v > 0 for v in quality_deltas),
        quality_worsened=sum(v < 0 for v in quality_deltas),
        mean_r3_to_r1_laplacian_ratio=statistics.mean(sharpness_ratios) if sharpness_ratios else None,
        caveats=["old quality and size are model judgments, not physical truth",
            "sharpness varies with texture and contrast and is not an image-quality guarantee",
            "comparisons use only completed three-round cases", "independent audit is not fed to the baseline"],
        reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (run / "iteration_analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2)+"\n")
    sections = []
    for index in range(len(jobs)//2):
        pair = jobs[index*2:index*2+2]
        image = board(run, index+1, pair).relative_to(run).as_posix()
        links = []
        for j in pair:
            for row in j.get("rounds", []):
                if row.get("status") == "returned":
                    target = Path(row["image"]).relative_to(run).as_posix()
                    links.append(f'<a href="{target}">背景{j["background_label"]} R{row["round"]}完整图</a>')
        sections.append(f'<section><h2>{html.escape(pair[0]["object_class"])}</h2><a href="{image}"><img src="{image}"></a><p>'+" · ".join(links)+"</p></section>")
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>上一个版本三轮对照</title><style>body{font:16px/1.6 system-ui;margin:24px;background:#eff2f5;color:#192330}main{max-width:1600px;margin:auto}section{margin:24px 0;background:white;padding:18px}img{width:100%;height:auto}a{color:#205fb0}</style><main><h1>上一个版本 0fded2f · 3类 · 背景44/50 · 三轮迭代</h1><p>原物参考、完整生成图及旧尺寸判断 / 独立原貌判断并排。三轮完成不等于质量验收通过。</p><p><a href="index.html">完整逐轮评估画廊</a> · <a href="summary.md">状态汇总</a> · <a href="iteration_analysis.json">各轮统计与轨迹</a></p>'
    (run / "comparison_index.html").write_text(page+''.join(sections)+"</main></html>", encoding="utf-8")
    print(json.dumps({k:v for k,v in analysis.items() if k not in {"trajectories", "caveats", "reporter_sha256"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
