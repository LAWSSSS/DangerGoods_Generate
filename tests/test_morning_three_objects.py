import io,json,shutil,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from scripts import test_morning_three_objects as batch

class MorningChecks(unittest.TestCase):
    def test_recovered_source_and_loop_only_change(self):
        for mode in batch.MODES:
            cfg=batch.configured(mode)
            self.assertEqual(batch.digest(Path(batch.app.__file__)),batch.EXPECTED_HASHES[mode])
            self.assertEqual((cfg.model,cfg.vlm_model),batch.EXPECTED_MODELS)
            forced=batch.force_three_rounds()
            self.assertNotIn('if not need_retry:',forced)
            self.assertIn('if False:',forced)
            self.assertFalse(hasattr(batch.app,'scores_pass'))
            self.assertEqual(batch.app.HORIZONTAL_ANGLE_SIGMA,12)
    def test_historical_deterministic_first_rounds_match(self):
        rows=batch.historical_crosscheck()
        self.assertEqual(len(rows),4)
        for r in rows:
            for field in ['exact_ratio_match','exact_prompt_match','exact_position_match','exact_orientation_match']:
                self.assertTrue(r[field],r)
    def test_no_uncertainty_or_default_passes(self):
        self.assertFalse(batch.audit_pass({}))
        self.assertFalse(batch.strict_old_evaluation(dict(size_judgment='acceptable',confidence=.5,quality_score=.5,fusion_quality='ok',reason='默认评估',visible_ratio=.02),True))
    def test_billed_request_is_once_with_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);batch.save(run/'requests.json',dict(limit=36,requests=[]))
            batch.reserve(run,'wan','j','r1')
            with self.assertRaises(RuntimeError):batch.reserve(run,'wan','j','r1')
            for i in range(35):batch.reserve(run,'wan','j',str(i))
            with self.assertRaises(RuntimeError):batch.reserve(run,'wan','x','r1')
    def test_two_original_pipelines_keep_three_rounds_without_wan_seed(self):
        image=io.BytesIO();Image.new('RGB',(1200,675),'gray').save(image,format='PNG')
        opening={'center':{'x':.5,'y':.5},'corners':[{'x':.1,'y':.1},{'x':.9,'y':.1},{'x':.9,'y':.9},{'x':.1,'y':.9}]}
        ev=dict(size_judgment='acceptable',confidence=.9,quality_score=.9,fusion_quality='good',reason='mock',visible_ratio=.03)
        audit=dict(object_present=False,reference_match='mismatch',uncertainties=['mock'])
        class Response:
            content=image.getvalue()
            def __init__(self,value):self.value=value
            def raise_for_status(self):pass
            def json(self):return self.value
        class Session:
            def post(self,url,*,json,**kwargs):
                if json['model']=='wan2.7-image':
                    self.assert_params=json['parameters']
                    if 'seed' in self.assert_params:raise AssertionError('unexpected Wan seed')
                    return Response({'output':{'choices':[{'message':{'content':[{'image':'https://mock.invalid/image.png'}]}}]}})
                text=str(json['messages'][-1]['content'][0])
                value=opening if 'corners' in text else audit if '独立' in text else {'generation_prompt':'尺寸遵循本轮设定。'} if 'generation_prompt' in text else ev
                return Response({'choices':[{'message':{'content':__import__('json').dumps(value)}}]})
            def get(self,*args,**kwargs):return Response({})
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);shutil.copytree(batch.RECOVERED,run/'source_snapshot');batch.save(run/'requests.json',dict(limit=36,requests=[]))
            for mode in batch.MODES:
                obj=str(batch.ROOT/'input/objects/密封胶罐.jpg')
                job=dict(mode=mode,id='mock_'+mode,object_class='密封胶罐',seed=20261012,reference_paths=[obj] if mode=='with_reference' else [],object_path=obj,background_path=str(batch.ROOT/'input/backgrounds'/(batch.BG_PREFIX+'44.jpg')),background_label='44')
                with patch.object(batch.requests,'Session',Session):result=batch.worker(str(run),job)
                self.assertEqual(result['status'],'completed',result.get('error'))
                self.assertEqual(len(result['rounds']),3)
                self.assertEqual(result['selected_round'],3)
                self.assertEqual(result['historical_stop_round'],1)
                for r in result['rounds']:
                    self.assertTrue(r['baseline_evaluation_valid'])
                    self.assertTrue(r['baseline_would_stop'])
                    self.assertIsNone(r['previous_image'])
                    self.assertEqual(len(r['generation_image_inputs']),1 if mode=='no_reference' else 2)
                    self.assertNotIn('seed',r['request_parameters'])
                    self.assertFalse(r['independent_passed'])
            ledger=json.loads((run/'requests.json').read_text())
            self.assertEqual(sum(r['kind']=='wan' for r in ledger['requests']),6)
            self.assertEqual(sum(r['kind']=='vlm' for r in ledger['requests']),20)
if __name__=='__main__':unittest.main()
