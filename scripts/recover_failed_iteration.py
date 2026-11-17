#!/usr/bin/env python3
"""Recover one failed final round from its recorded feedback and seeded baseline."""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import test_prev_branch_iterations as batch
import generate_aliyun as app


def replay_final_payload(run, job, original):
    receipts = json.loads((run / "requests.json").read_text())["requests"]
    q = [r for r in receipts if r["job"] == job["id"] and r["kind"] == "qwen"]
    if len(q) != 5 or any(r["status"] != "returned" for r in q):
        raise ValueError("Final-round replay requires opening and two complete recorded evaluations/audits")
    responses = [json.loads(Path(r["response_path"]).read_text()) for r in q]
    captured, state = {}, {"q": 0, "wan": 0}

    class Response:
        def __init__(self, raw=None, data=None):
            self.raw, self.content = raw, data
        def json(self): return self.raw
        def raise_for_status(self): pass

    class ReplaySession:
        verify = False
        def post(self, url, *, json, **unused):
            if json["model"] == "qwen3.8-flash":
                raw = responses[state["q"]]
                state["q"] += 1
                return Response(raw)
            state["wan"] += 1
            n = state["wan"]
            prompt = json["input"]["messages"][0]["content"][0]["text"]
            if n < 3:
                if prompt != original["rounds"][n-1]["full_prompt"]:
                    raise ValueError("Seeded replay does not match the original submitted prompt")
                return Response({"output": {"choices": [{"message": {"content": [{"image": "https://replay.invalid/image"}]}}]}})
            captured["payload"] = json
            raise RuntimeError("offline_replay_complete_no_network_request")
        def get(self, *unused, **kwargs):
            return Response(data=Path(original["rounds"][state["wan"]-1]["image"]).read_bytes())

    with tempfile.TemporaryDirectory(prefix="iteration-replay-") as temp:
        scratch = Path(temp)
        batch.save(scratch / "requests.json", {"limit": 90, "requests": []})
        with patch.object(batch.requests, "Session", ReplaySession):
            replay = batch.worker(str(scratch), job)
        if state != {"q": 5, "wan": 3} or "payload" not in captured:
            raise ValueError("Incomplete offline replay: " + str(replay.get("error")))
    importlib.reload(app)
    return captured["payload"]


def restore_dialogue(folder, receipts):
    # The baseline retains the opening plus the first two generated-image turns.
    candidates = [r for r in receipts if r["stage"] == "evaluation_round_2"]
    if len(candidates) != 1:
        raise ValueError("Missing recorded second evaluation")
    entry = candidates[0]
    clean = json.loads((folder / f"qwen_{entry['id']:04d}_request.json").read_text())["messages"]
    source = json.loads((folder / "result.json").read_text())
    images = [app.image_to_data_url_vlm(Path(source["background_path"]))]
    import base64
    for row in source["rounds"][:2]:
        with Image.open(row["image"]) as im:
            im.thumbnail((768, 768), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            im.convert("RGB").save(buffer, format="JPEG", quality=75)
            images.append("data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode())
    mapping = {hashlib.sha256(s.encode()).hexdigest(): s for s in images}
    for message in clean:
        if isinstance(message["content"], list):
            message["content"] = [{"type": "image_url", "image_url": {"url": mapping[p["encoded_sha256"]]}}
                if p.get("type") == "image_url" else p for p in message["content"]]
    raw = json.loads(Path(entry["response_path"]).read_text())
    content = raw["choices"][0]["message"]["content"]
    if isinstance(content, list): content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    clean.append({"role": "assistant", "content": content})
    return clean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--replay-only", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    if subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT).decode().strip() != "feat/realistic-object-scale":
        raise ValueError("Expected selected feat/realistic-object-scale branch")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip() != manifest["head"]:
        raise ValueError("Selected commit differs from the original experiment")
    for name, h in manifest["source_hashes"].items():
        if batch.digest(ROOT/name) != h: raise ValueError("Original experiment source changed")
    job = next(j for j in manifest["jobs"] if j["id"] == args.case)
    folder = run / "cases" / job["id"]
    original = json.loads((folder / "result.json").read_text())
    if original["status"] != "failed" or len(original["rounds"]) != 3 or original["rounds"][-1]["status"] != "failed":
        raise ValueError("Only a failed final generation can be recovered")
    receipts = [r for r in json.loads((run / "requests.json").read_text())["requests"] if r["job"] == job["id"]]
    if any(r["kind"] == "wan_recovery" for r in receipts): raise ValueError("Recovery already submitted; do not repeat it")
    existing_hashes = {r["image"]:batch.digest(r["image"]) for r in original["rounds"][:2]}
    payload = replay_final_payload(run, job, original)
    dialogue = restore_dialogue(folder, receipts)
    preview = {"offline_replay_verified": True, "matched_original_round_prompts": [1,2],
        "prompt":payload["input"]["messages"][0]["content"][0]["text"], "parameters":payload["parameters"],
        "image_hashes":[hashlib.sha256(p["image"].encode()).hexdigest() for p in payload["input"]["messages"][0]["content"][1:]],
        "first_two_images_unchanged":existing_hashes, "source_script_hash":batch.digest(Path(__file__))}
    batch.save(folder / "recovery_replay.json", preview)
    if args.replay_only:
        print("Offline final-round payload and original dialogue reconstructed; zero model calls.")
        return
    batch.save(folder / "result_before_recovery.json", original)
    with redirect_stdout(io.StringIO()): cfg = app.load_app_config()
    state = {"stage":"generation_round_3_recovery", "qwen_ok":False}
    rid = batch.reserve(run, "wan_recovery", job["id"], state["stage"])
    try:
        response = batch.session().post(cfg.image_api_url, headers={"Authorization":"Bearer "+cfg.image_api_key,"Content-Type":"application/json"}, json=payload, timeout=cfg.timeout_seconds)
        response.raise_for_status(); raw=response.json()
        batch.save(folder / f"wan_recovery_{rid:04d}_response.json", raw)
        batch.finish(run,rid,status="returned",response_path=str(folder/f"wan_recovery_{rid:04d}_response.json"))
        path=folder/"round3.png"
        app.get_session_with_ssl_bypass=batch.session
        path.write_bytes(app.download_image(app.extract_image_urls(raw,cfg.image_protocol)[0]))
    except Exception as exc:
        batch.finish(run,rid,status="failed",error=str(exc));raise
    row=dict(original["rounds"][-1]);row.pop("error",None)
    row.update(status="returned",request_id=rid,raw_image_sha256=batch.digest(path),
        full_prompt=preview["prompt"],recovered_after_pause_transport_failure=True,
        previous_failed_request_id=original["rounds"][-1]["request_id"])
    (folder/"round3_prompt.txt").write_text(preview["prompt"])
    with Image.open(path) as im:row["output_dimensions"]=list(im.size)

    class Analyzer(app.VLMAnalyzer):
        def _post_vlm(self,messages,max_tokens):
            qid=batch.reserve(run,"qwen",job["id"],state["stage"]);state["qwen_ok"]=False
            batch.save(folder/f"qwen_{qid:04d}_request.json",dict(model=cfg.vlm_model,messages=batch.clean_messages(messages),max_tokens=max_tokens,temperature=.3))
            try:
                response=batch.session().post(cfg.vlm_api_url,headers={"Authorization":"Bearer "+cfg.vlm_api_key,"Content-Type":"application/json"},json=dict(model=cfg.vlm_model,messages=messages,max_tokens=max_tokens,temperature=.3),timeout=cfg.timeout_seconds)
                response.raise_for_status();data=response.json();batch.save(folder/f"qwen_{qid:04d}_response.json",data)
                content=data["choices"][0]["message"]["content"]
                if isinstance(content,list):content="".join(p.get("text","") for p in content if isinstance(p,dict))
                state["qwen_ok"]=True;batch.finish(run,qid,status="returned",response_path=str(folder/f"qwen_{qid:04d}_response.json"));return content
            except Exception as exc:
                batch.finish(run,qid,status="failed",error=str(exc));raise RuntimeError(str(exc)) from exc
    analyzer=Analyzer(cfg);analyzer.vlm_messages=dialogue
    state["stage"]="evaluation_round_3_recovery"
    old=analyzer.evaluate_generated_image(path,job["object_class"],row["prompt_core"],row["target_ratio"])
    row.update(baseline_evaluation=old,baseline_evaluation_valid=batch.strict_old_evaluation(old,state["qwen_ok"]),baseline_would_stop=old.get("size_judgment")=="acceptable" or not old.get("should_retry",False),sharpness=batch.sharpness(path))
    batch.visible_roi(path,row["sampled_xy"],row["target_ratio"],folder/"round3_target.png")
    content=[{"type":"text","text":batch.audit_prompt(job,row,app.object_scale_range(cfg,job["object_class"]))}]
    for p in [Path(job["object_path"]),Path(job["background_path"]),path,folder/"round3_target.png"]:
        content.append({"type":"image_url","image_url":{"url":batch.compressed_original(p)}})
    state["stage"]="independent_audit_round_3_recovery"
    try:
        audited=analyzer._extract_json(analyzer._post_vlm([{"role":"user","content":content}],3200))
        row.update(independent_audit=audited,independent_passed=batch.audit_pass(audited))
    except Exception as exc:row.update(independent_audit={"error":str(exc)},independent_passed=False)
    for p,h in existing_hashes.items():assert batch.digest(p)==h
    batch.save(folder/"round3_evaluation.json",row)
    original["rounds"][-1]=row;original.update(status="completed",completed_at=datetime.now(timezone.utc).isoformat(),recovery={"request_id":rid,"failed_request_retained":True,"first_two_images_unchanged":True},selected_branch="feat/realistic-object-scale",passed_rounds=[r["round"] for r in original["rounds"] if r.get("independent_passed")])
    original.pop("error",None);batch.save(folder/"result.json",original)
    print(json.dumps(dict(case=job["id"],recovered=True,rounds=3,recovery_wan_requests=1),ensure_ascii=False))


if __name__=="__main__":main()
