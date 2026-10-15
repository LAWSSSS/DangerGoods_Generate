#!/usr/bin/env python3
"""Replay recorded morning dialogue locally and retry only a missing generation; keep earlier images."""
import argparse,json,shutil,hashlib
from pathlib import Path
from unittest.mock import patch
from scripts import test_morning_three_objects as batch

class CachedResponse:
    def __init__(self,value=None,content=None):self.value,self.content=value,content
    def raise_for_status(self):pass
    def json(self):return self.value

def recover(run,case):
    ledger=json.loads((run/'requests.json').read_text())['requests'];folder=run/'cases'/case
    old=json.loads((folder/'result.json').read_text())
    assert old['status']=='failed'
    failed=[r for r in old['rounds'] if r['status']=='failed'];assert len(failed)==1
    number=failed[0]['round'];assert number==3,'This recovery is limited to the final round.'
    wan=[r for r in ledger if r['job']==case and r['kind']=='wan' and r['stage']==f'generation_round_{number}'];assert len(wan)==1 and wan[0]['status']=='failed'
    assert '504' in wan[0]['error'],'Only confirmed HTTP 504 failures may be recovered by this command.'
    assert not any(r['job']==case and r['kind']=='wan_recovery' for r in ledger),'Do not automatically repeat a recovery.'
    backup=folder/'result_before_recovery.json';assert not backup.exists();shutil.copyfile(folder/'result.json',backup)
    manifest=json.loads((run/'manifest.json').read_text());job=next(j for j in manifest['jobs'] if j['id']==case)
    cache={(r['kind'],r['stage']):r for r in ledger if r['job']==case and r['status']=='returned'}
    images={f"generation_round_{r['round']}":Path(r['image']).read_bytes() for r in old['rounds'] if r['status']=='returned'}
    raw={key:json.loads(Path(r['response_path']).read_text()) for key,r in cache.items()}
    request_data={key:json.loads((folder/f"vlm_{r['id']:04d}_request.json").read_text()) for key,r in cache.items() if key[0]=='vlm'}
    original_reserve,original_finish,real_session=batch.reserve,batch.finish,batch.requests.Session
    state={};replayed=[]
    def reserve(active_run,kind,jobid,stage):
        assert jobid==case;state.update(kind=kind,stage=stage,key=(kind,stage))
        if state['key'] in cache:
            replayed.append(dict(kind=kind,stage=stage,request_id=cache[state['key']]['id']))
            return cache[state['key']]['id']
        if kind=='wan':
            assert stage==f'generation_round_{number}'
            return original_reserve(active_run,'wan_recovery',jobid,stage)
        return original_reserve(active_run,kind,jobid,stage)
    cached_ids={r['id'] for r in cache.values()}
    def finish(active_run,rid,**fields):
        if rid not in cached_ids:original_finish(active_run,rid,**fields)
    class ReplaySession:
        def __init__(self):self.actual=real_session();self.actual.verify=False;self.verify=False
        def post(self,url,*,json,**kwargs):
            key=state['key']
            if key in cache:
                if key[0]=='vlm':
                    request=request_data[key]
                    assert json['model']==request['model']
                    assert batch.clean_messages(json['messages'])==request['messages'],f'VLM dialogue replay changed at {key}'
                    assert json['max_tokens']==request['max_tokens'] and json['temperature']==request['temperature']
                else:
                    row=next(r for r in old['rounds'] if key[1]==f"generation_round_{r['round']}")
                    assert json['parameters']==row['request_parameters']
                    assert json['input']['messages'][0]['content'][0]['text']==row['full_prompt']
                return CachedResponse(raw[key])
            if key[0]=='wan':
                assert json['parameters']['n']==1 and 'seed' not in json['parameters']
                live=json_load(folder/'result.json')['rounds'][-1]
                assert live['prompt_core']==failed[0]['prompt_core']
                assert live['target_ratio']==failed[0]['target_ratio']
                assert live['sampled_xy']==failed[0]['sampled_xy']
                assert live['position_hint']==failed[0]['position_hint'] and live['orientation_hint']==failed[0]['orientation_hint']
                assert live['generation_image_inputs']==failed[0]['generation_image_inputs']
                assert live['request_parameters']==failed[0]['request_parameters']
            return self.actual.post(url,json=json,**kwargs)
        def get(self,url,**kwargs):
            if state['kind']=='wan' and state['stage'] in images:return CachedResponse(content=images[state['stage']])
            return self.actual.get(url,**kwargs)
    with patch.object(batch,'reserve',reserve),patch.object(batch,'finish',finish),patch.object(batch.requests,'Session',ReplaySession):
        result=batch.worker(str(run),job)
    for row in old['rounds']:
        if row['status']=='returned':assert batch.digest(row['image'])==row['raw_image_sha256'],'Original image modified'
    record=dict(case=case,failed_original_request=wan[0]['id'],retry_round=number,replayed_requests=replayed,
        earlier_images_byte_identical=True,original_result_backup=str(backup),status=result['status'],error=result.get('error'),
        failed_round_core_prompt_position_orientation_size_and_parameters_verified=True)
    batch.save(folder/'recovery.json',record);batch.summarize(run);print(json.dumps(record,ensure_ascii=False))
    return result

def json_load(path):return json.loads(Path(path).read_text())
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--case',required=True);a=p.parse_args();recover(a.run_dir.resolve(),a.case)
if __name__=='__main__':main()
