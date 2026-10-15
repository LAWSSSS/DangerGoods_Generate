#!/usr/bin/env python3
"""Build morning A/B iteration comparisons, verify provenance, and export raw images; no model calls."""
import argparse,hashlib,html,json,shutil,statistics,subprocess
from datetime import datetime
from pathlib import Path
from PIL import Image,ImageDraw
from scripts.report_pose_scores_three_objects import font,place
from scripts import test_morning_three_objects as batch

def read(path):return json.loads(Path(path).read_text())
def save(path,value):batch.save(path,value)

def board(run,number,jobs):
    width,height=1552,790
    canvas=Image.new('RGB',(width,height),'#15212c');d=ImageDraw.Draw(canvas)
    title,label,small=font(26),font(19),font(16)
    d.text((16,10),f"{jobs[0]['object_class']} · 固定场景 {jobs[0]['background_label']} · 10月12日上午两版本",font=title,fill='white')
    d.text((16,48),'原始代码＋历史模型；每组强制三轮；Python seed 20261012，Wan未传seed。独立核对不反馈。',font=small,fill='#c5d4e0')
    d.text((16,90),'原物主参考',font=label,fill='#c5d4e0');place(canvas,jobs[0]['object_path'],(16,134,225,642))
    d.text((16,680),'下方主参考供对照',font=small,fill='#c5d4e0');d.text((16,706),'上排生图不输入原物',font=small,fill='#c5d4e0')
    for i,j in enumerate(jobs):
        y=88+i*337
        d.text((244,y),batch.MODE_LABELS[j['mode']],font=label,fill='white')
        for n in (1,2,3):
            x=244+(n-1)*436;r=next((r for r in j.get('rounds',[]) if r['round']==n),{})
            d.text((x,y+29),f'R{n}'+(' · 原算法会停止' if j.get('historical_stop_round')==n else ''),font=label,fill='white')
            if r.get('status')=='returned':
                place(canvas,r['image'],(x,y+58,x+418,y+293))
                ev=r.get('baseline_evaluation',{});a=r.get('independent_audit',{})
                size={'too_small':'偏小','acceptable':'合适','too_large':'偏大'}.get(ev.get('size_judgment'),'待评估') if r.get('baseline_evaluation_valid') else '评估无效/待评估'
                identity={'match':'匹配','mismatch':'不符','uncertain':'不确定'}.get(a.get('reference_match'),'待评估')
                d.text((x,y+298),f"原尺寸：{size} · 目标占宽 {r['target_ratio']:.1%}",font=small,fill='#c5d4e0')
                d.text((x,y+319),f"原貌 {identity} · 独立核对"+('通过' if r.get('independent_passed') else '未通过'),font=small,fill='#6ae2ac' if r.get('independent_passed') else '#ffc77e')
            else:
                d.rectangle((x,y+58,x+418,y+293),fill='#233240');d.text((x+20,y+100),r.get('status','尚未生成'),font=label,fill='#c5d4e0')
    path=run/'comparison_boards'/f"{number:02d}_{jobs[0]['object_class']}_{jobs[0]['background_label']}.png";path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path);return path

def analysis(jobs):
    groups={}
    for mode in batch.MODES:
        subset=[j for j in jobs if j['mode']==mode];rows=[r for j in subset for r in j.get('rounds',[]) if r['status']=='returned']
        by_round=[]
        for n in (1,2,3):
            current=[r for r in rows if r['round']==n];valid=[r for r in current if r.get('baseline_evaluation_valid')];audits=[r.get('independent_audit',{}) for r in current]
            by_round.append(dict(round=n,images=len(current),valid_native_evaluations=len(valid),
                size_acceptable=sum(r['baseline_evaluation']['size_judgment']=='acceptable' for r in valid),
                mean_native_quality=statistics.mean(r['baseline_evaluation']['quality_score'] for r in valid) if valid else None,
                identity_match=sum(a.get('reference_match')=='match' for a in audits),
                identity_mismatch=sum(a.get('reference_match')=='mismatch' for a in audits),
                identity_uncertain=sum(a.get('reference_match')=='uncertain' for a in audits),
                stable_support=sum(a.get('support_judgment')=='stable' for a in audits),
                natural_integration=sum(a.get('integration_judgment')=='natural' for a in audits),
                independent_pass=sum(r.get('independent_passed',False) for r in current)))
        groups[mode]=dict(label=batch.MODE_LABELS[mode],cases=len(subset),images=len(rows),by_round=by_round,
            identity_match=sum(r.get('independent_audit',{}).get('reference_match')=='match' for r in rows),
            independent_pass=sum(r.get('independent_passed',False) for r in rows))
    return dict(modes=groups,trajectories=[dict(case=j['id'],historical_stop_round=j.get('historical_stop_round'),last_round=j.get('selected_round'),
        target_ratios=[r.get('target_ratio') for r in j.get('rounds',[])],
        native_size=[r.get('baseline_evaluation',{}).get('size_judgment') for r in j.get('rounds',[])],
        identity=[r.get('independent_audit',{}).get('reference_match') for r in j.get('rounds',[])],
        independent_pass=[r.get('independent_passed',False) for r in j.get('rounds',[])]) for j in jobs],
        caveats=['Only 3 objects, 2 backgrounds and one sample per round; not a controlled statistical quality estimate.',
            'Both native VLM and independent audits are model judgments; uncertainty cannot pass.',
            'Exact recovered source/config bytes; seed matches recorded Python CLI; Wan had no seed and VLM remains stochastic.',
            'Later historical oil inputs included a red variant; this test uses the selected blue main reference only per human instruction.',
            'Three rounds are forced after native size-only early stop; morning output is last round without afternoon scoring/selection.'])

def validate_export(run,manifest,report):
    root=batch.ROOT;ledger=read(run/'requests.json')['requests'];jobs=report['jobs'];rows=[r for j in jobs for r in j['rounds']]
    assert len(jobs)==report['completed']==12 and len(rows)==report['images']==36
    assert not any(r['status']=='submitted' for r in ledger)
    assert sum(r['kind']=='wan' for r in ledger)==36
    assert all(r['status']=='returned' for r in ledger)
    assert manifest['models']==dict(generation='wan2.7-image',vlm='doubao-seed-2.1-pro')
    for mode,hashes in manifest['source_hashes'].items():
        for name,value in hashes.items():
            assert batch.digest(run/'source_snapshot'/mode/name)==value
            assert batch.digest(batch.RECOVERED/mode/name)==value
    for name in batch.SOURCE_FILES+['requirements.txt']:
        assert (root/name).read_bytes()==subprocess.check_output(['git','show',manifest['head']+':'+name],cwd=root),name
    exports=[];stamp=datetime.fromisoformat(manifest['created_at']).strftime('%Y%m%dT%H%M%SZ')
    for j in jobs:
        assert j['status']=='completed' and [r['round'] for r in j['rounds']]==[1,2,3]
        assert batch.digest(j['object_path'])==j['object_hash'] and batch.digest(j['background_path'])==j['background_hash']
        assert j['reference_paths']==([j['object_path']] if j['mode']=='with_reference' else [])
        stem=f"retest_morning_{j['id']}_wan2.7-image_{stamp}";round_paths=[]
        with Image.open(j['background_path']) as im:bg_ratio=im.width/im.height
        for r in j['rounds']:
            assert r['status']=='returned' and r['previous_image'] is None
            assert r['generation_image_inputs']==[j['background_path'],*j['reference_paths']]
            assert r['generation_input_hashes']==[batch.digest(p) for p in r['generation_image_inputs']]
            assert 'seed' not in r['request_parameters'] and r['request_parameters']['n']==1
            w,h=map(int,r['request_parameters']['size'].split('*'));assert abs(w/h-bg_ratio)<.01
            assert abs(r['output_dimensions'][0]/r['output_dimensions'][1]-bg_ratio)<.01
            assert r['size_spec']['target_ratio']==r['target_ratio']
            assert batch.digest(r['image'])==r['raw_image_sha256']
            assert (run/'cases'/j['id']/f"round{r['round']}_prompt.txt").read_text()==r['full_prompt']
            path=root/'output/images'/f"{stem}_round{r['round']}.png";path.parent.mkdir(exist_ok=True)
            if path.exists():assert batch.digest(path)==r['raw_image_sha256']
            else:shutil.copyfile(r['image'],path)
            round_paths.append(str(path))
        assert j['selected_round']==3 and batch.digest(j['selected_image'])==j['rounds'][-1]['raw_image_sha256']
        final=root/'output/final_images'/f'{stem}.png'
        if final.exists():assert batch.digest(final)==j['selected_image_sha256']
        else:shutil.copyfile(j['selected_image'],final)
        chosen=j['rounds'][-1];md=root/'output/metadata'/f'{stem}.json'
        save(md,dict(image_id=stem,test_run=str(run),historical_source='Cursor original file-content checkpoints',
            historical_mode=j['mode'],source_sha256=manifest['source_hashes'][j['mode']]['generate_aliyun.py'],
            working_branch=manifest['branch'],working_commit=manifest['head'],background_image=j['background_path'],object_image=j['object_path'],
            object_views=j['reference_paths'],object_class=j['object_class'],model='wan2.7-image',vlm_model='doubao-seed-2.1-pro',
            python_seed=j['seed'],wan_seed=None,generated_images=[str(final)],round_images=round_paths,
            forced_rounds=3,optimization_rounds=3,selected_round=3,selected_reason=j['selected_reason'],historical_stop_round=j['historical_stop_round'],
            scrap_position=j['position'],object_orientation=j['orientation'],size_rounds=j['size_rounds'],judgment_history=j['judgment_history'],
            optimization_history=read(run/'cases'/j['id']/'history.json'),full_prompt=chosen['full_prompt'],
            api_result=read(run/'cases'/j['id']/'round3_api.json'),independent_evaluations=[r.get('independent_audit',{}) for r in j['rounds']],
            quality_status='independent_pass' if chosen.get('independent_passed') else 'not_passed',
            notes='Generated with wan2.7-image (HTTP), VLM: doubao-seed-2.1-pro; forced three rounds; no references in early version, primary-only in later version.'))
        exports.append(dict(case=j['id'],rounds=round_paths,final=str(final),metadata=str(md)))
    actual=[]
    for r in ledger:
        req=read(run/'cases'/r['job']/f"vlm_{r['id']:04d}_request.json") if r['kind']=='vlm' else {'model':'wan2.7-image'}
        assert req['model']==('doubao-seed-2.1-pro' if r['kind']=='vlm' else 'wan2.7-image')
        actual.append(dict(r,model=req['model']))
    save(run/'requests_by_model.json',dict(requests=actual));save(run/'exports.json',exports)
    final=dict(cases=12,raw_images=36,last_round_exports=12,wan_requests=36,vlm_requests=sum(r['kind']=='vlm' for r in ledger),
        models=manifest['models'],source_config_and_prompt_hashes_verified=True,core_checkout_unchanged=True,
        native_loop_only_early_stop_disabled=True,no_previous_generated_inputs=True,background_aspect_ratios_verified=True,
        raw_outputs_and_exports_byte_identical=True,reference_counts_verified=True,wan_seed_absent_verified=True,
        native_evaluations_valid=sum(r.get('baseline_evaluation_valid',False) for r in rows),independent_passes=sum(r.get('independent_passed',False) for r in rows))
    discarded=root/'output/validation/morning_both_three_objects_20261014/discarded_run.json'
    if discarded.is_file():
        invalid=read(discarded)
        final.update(discarded_entry_mismatch_run=str(discarded.parent),discarded_wan_requests=invalid['wan_requests'],
            total_wan_requests_including_discarded=36+invalid['wan_requests'],discarded_returned_images=invalid['wan_returned'],
            discarded_vlm_requests=invalid['vlm_requests'],discarded_vlm_interrupted_unknown=invalid['vlm_interrupted'])
    save(run/'final_validation.json',final)
    for name in ['scripts/test_morning_three_objects.py','scripts/report_morning_three_objects.py','scripts/report_pose_scores_three_objects.py','tests/test_morning_three_objects.py']:
        target=run/'report_source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,target)
    return final

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--finalize',action='store_true');args=p.parse_args()
    run=args.run_dir.resolve();report=batch.summarize(run);manifest=read(run/'manifest.json');jobs=report['jobs'];a=analysis(jobs);save(run/'iteration_analysis.json',a)
    sections=[];n=0
    for _,cls,_ in batch.OBJECTS:
        for bg in ('44','50'):
            pair=[next(j for j in jobs if j['mode']==mode and j['object_class']==cls and j['background_label']==bg) for mode in batch.MODES]
            n+=1;image=board(run,n,pair).relative_to(run).as_posix()
            links=[f'<a href="{Path(r["image"]).relative_to(run).as_posix()}">{batch.MODE_LABELS[j["mode"]]} R{r["round"]}</a>' for j in pair for r in j.get('rounds',[]) if r['status']=='returned']
            sections.append(f'<section><h2>{html.escape(cls)} · {bg}</h2><a href="{image}"><img src="{image}"></a><p>'+ ' · '.join(links)+'</p></section>')
    tables=[]
    for mode,v in a['modes'].items():
        trs=''.join(f'<tr><td>R{r["round"]}</td><td>{r["images"]}</td><td>{r["size_acceptable"]}</td><td>{r["identity_match"]}</td><td>{r["independent_pass"]}</td></tr>' for r in v['by_round'])
        tables.append(f'<h2>{v["label"]}</h2><table><tr><th>轮次</th><th>图数</th><th>旧尺寸合适</th><th>原貌匹配</th><th>独立综合通过</th></tr>{trs}</table>')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>上午两版本三轮对照</title><style>body{font:16px/1.6 system-ui;margin:24px;background:#eff2f5;color:#192330}main{max-width:1600px;margin:auto}section{margin:24px 0;background:white;padding:18px}img{width:100%}a{color:#205fb0}td,th{padding:6px 20px;text-align:left}table{background:white}</style><main><h1>10月12日上午两版本 · 三物品 · 44/50 · 各三轮</h1><p>原始源码从编辑器完整内容快照恢复。Wan 2.7 Image / Doubao Seed 2.1 Pro；Python种子20261012；Wan未传seed。上午前段只输入背景，后段加入指定主参考。两组都使用完整原背景，不回输生成图。</p><p>按用户要求跑满三轮，单独记录原算法按尺寸提前停止的轮次。后段油缸只用蓝色主图，与历史蓝/红混合参考有输入差异。模型输出仍有随机性；此小样本结果不代表稳定质量提升。另有6张入口权重失配试跑已剔除并存档，正式36张之外共计42次Wan请求。</p><p><a href="index.html">逐轮提示词与评估画廊</a> · <a href="manifest.json">源码/配置/输入与调整记录</a> · <a href="iteration_analysis.json">完整统计</a></p>'
    (run/'comparison_index.html').write_text(page+''.join(tables)+''.join(sections)+'</main></html>')
    if args.finalize:print(json.dumps(validate_export(run,manifest,report),ensure_ascii=False))
    else:print(json.dumps({mode:{k:v for k,v in val.items() if k!='by_round'} for mode,val in a['modes'].items()},ensure_ascii=False))
if __name__=='__main__':main()
