import json,shutil,tempfile,unittest,io
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from scripts import test_morning_three_objects as batch
from scripts import recover_morning_failed_round as recovery
class RecoveryCheck(unittest.TestCase):
    def test_rebuild_missing_round_uses_cached_dialogue_and_two_existing_images(self):
        source=batch.ROOT/'output/validation/morning_both_three_objects_20261014_corrected'
        case='no_reference_01_sealant_44';old=json.loads((source/'cases'/case/'result.json').read_text())
        self.assertEqual(old['status'],'failed')
        before=[batch.digest(r['image']) for r in old['rounds'][:2]]
        data=io.BytesIO();Image.new('RGB',(1536,864),'gray').save(data,format='PNG');calls=[]
        class Response:
            content=data.getvalue()
            def __init__(self,value):self.value=value
            def raise_for_status(self):pass
            def json(self):return self.value
        class FakeNetwork:
            def post(self,url,*,json,**kwargs):
                calls.append(json['model'])
                if json['model']=='wan2.7-image':return Response({'output':{'choices':[{'message':{'content':[{'image':'https://mock.invalid/third.png'}]}}]}})
                text=str(json['messages'][-1]['content'][0]);value={'object_present':False,'reference_match':'uncertain','uncertainties':['mock']} if '独立' in text else dict(size_judgment='acceptable',confidence=.9,quality_score=.8,fusion_quality='good',visible_ratio=.03,reason='mock')
                return Response({'choices':[{'message':{'content':__import__('json').dumps(value)}}]})
            def get(self,*args,**kwargs):return Response({})
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);shutil.copytree(source/'source_snapshot',run/'source_snapshot');shutil.copytree(source/'cases'/case,run/'cases'/case)
            for name in ['manifest.json','requests.json']:shutil.copyfile(source/name,run/name)
            with patch.object(batch.requests,'Session',FakeNetwork),patch('builtins.print'):
                result=recovery.recover(run,case)
            self.assertEqual(result['status'],'completed',result.get('error'))
            self.assertEqual(calls.count('wan2.7-image'),1)
            self.assertEqual(calls.count('doubao-seed-2.1-pro'),2)
            self.assertEqual(result['historical_stop_round'],3)
            self.assertEqual([batch.digest(r['image']) for r in old['rounds'][:2]],before)
            self.assertEqual([r['raw_image_sha256'] for r in result['rounds'][:2]],before)
            self.assertTrue((run/'cases'/case/'recovery.json').is_file())
if __name__=='__main__':unittest.main()
