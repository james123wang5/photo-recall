import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import numpy as np
from app.engine import Engine
from app.hybrid import tokens
from app.pipeline import fuse
from app.ranking import evidence_order
from app.retrieval import inspection_grade, normalize_inspection, strict_condition
from app.providers.ocr import LocalOCR
import subprocess


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        root=Path(self.temp.name);folder=root/'photos';folder.mkdir()
        Image.new('RGB',(30,20)).save(folder/'a.jpg')
        self.e=Engine(root/'data');self.lib=self.e.add_library(str(folder))
        self.e.index_library(self.lib['id'])
        with self.e.db() as con:self.asset=dict(con.execute('SELECT * FROM assets WHERE library_id=?',(self.lib['id'],)).fetchone())

    def tearDown(self):self.temp.cleanup()

    def test_chinese_bigrams_and_english_tokens(self):
        self.assertIn('北京',tokens('北京旅行 iPod'))
        self.assertIn('ipod',tokens('北京旅行 iPod'))

    def test_ocr_keyword_search_and_provenance_are_connected(self):
        h=self.e.hybrid;h.save(self.asset,'ocr','ocr-v1',{'provider':'PaddleOCR','lines':[]},'北京旅行 iPod')
        rankings=h.rankings([self.asset],[{'kind':'include','label':'北京'}],semantic=False)
        self.assertEqual(rankings['照片文字'],[self.asset['id']])
        with self.e.db() as con:payload=json.loads(con.execute('SELECT payload FROM observations').fetchone()[0])
        self.assertEqual(payload['provider'],'PaddleOCR')

    def test_stale_ocr_does_not_enter_search(self):
        h=self.e.hybrid;h.save(self.asset,'ocr','v1',{},'北京')
        with self.e.db() as con:con.execute("UPDATE assets SET fingerprint='new' WHERE id=?",(self.asset['id'],))
        self.assertEqual(h.rankings([self.asset],[{'kind':'include','label':'北京'}],False),{})

    def test_multiple_windows_do_not_get_multiple_votes(self):
        a=fuse({'整图':['a','b'],'局部1':['a','b'],'局部2':['a','b']},grouped=True)[1]
        b=fuse({'整图':['a','b']},grouped=True)[1]
        self.assertEqual(a,b)

    def test_text_vectors_use_their_own_model_and_cache(self):
        h=self.e.hybrid;h.save(self.asset,'caption','v1',{},'a person beside a car')
        vector=np.ones(1024,dtype=np.float32)/32
        with self.e.db() as con:con.execute('UPDATE observations SET vector=?,vector_version=?',(vector.tobytes(),h.embedding.version))
        with patch.object(h.embedding,'encode',return_value=[vector]) as encode:
            c=[{'kind':'include','label':'人在车旁边'}]
            self.assertEqual(h.rankings([self.asset],c)['描述语义'],[self.asset['id']])
            h.rankings([self.asset],c)
            self.assertEqual(encode.call_count,1)

    def test_hybrid_normal_query_does_not_run_core(self):
        s=self.e.session(self.lib['id']);s['conditions']=[{'kind':'include','label':'汽车','concept':'object'}]
        aid=self.asset['id'];rankings={'整图':[aid]};order,scores,sources=fuse(rankings)
        job={'status':'running','started':time.time(),'cancel':threading.Event(),'results':[],'done':0,'errors':[]}
        with patch.object(self.e.pipeline,'recall',return_value=(rankings,(order,scores,sources))),patch.object(self.e.pipeline.reranker,'score') as score,patch.object(self.e,'inspect_candidates',return_value=[]) as inspect:
            self.e.hybrid.refine(self.e.pipeline,s,[self.asset],{},job)
            score.assert_not_called()
            self.assertEqual(job['status'],'complete')
            self.assertEqual(job['total'],1)
            self.assertEqual(inspect.call_count,1)
            self.assertEqual(job['results'][0]['id'],aid)

    def test_full_query_anchor_beats_a_crop_only_accidental_match(self):
        ids=['match']+[f'other{i}' for i in range(80)]+['crop-only']
        order,_,_=fuse({'完整描述':ids,'整图补漏':['crop-only']+ids[:-1],
                         '局部：one cue':['crop-only']+ids[:-1]},grouped=True)
        self.assertLess(order.index('match'),order.index('crop-only'))

    def test_manual_check_publishes_persistent_backend_order(self):
        p=self.e.pipeline;ids=[str(i) for i in range(50)]
        job={'pipeline':'vnext','key':'same','revision':0,'hybrid':True,'base_order':ids,
             'results':[{'id':aid,'rank_order':i} for i,aid in enumerate(ids)]}
        self.e.verifications['session']=job
        p.publish_inspections('session','same',[{'id':'40','inspection_tier':2,'inspection_score':1,
            'inspection':{'conditions':[]}}])
        self.assertEqual(self.e.verification_state('session')['results'][35]['id'],'40')
        self.assertEqual(p.ranking_snapshot('session')['ids'][35],'40')
        self.assertEqual(job['revision'],1)
        p.publish_inspections('session','old-query',[{'id':'4','inspection_tier':2}])
        self.assertEqual(job['revision'],1)

    def test_negative_evidence_moves_down_without_burying_to_end(self):
        ids=[str(i) for i in range(100)];lookup={aid:{'id':aid} for aid in ids}
        lookup['0'].update(inspection_tier=0,inspection_strict_conflict=True)
        lookup['70'].update(inspection_tier=2,inspection_score=1)
        order=evidence_order(ids,lookup)
        self.assertGreaterEqual(order.index('70'),64)
        self.assertLessEqual(order.index('0'),6)
        self.assertEqual(len(set(order)),100)

    def test_semantic_disagreement_and_uncertain_check_preserve_original_order(self):
        ids=[str(i) for i in range(100)];lookup={aid:{} for aid in ids}
        lookup['0'].update(inspection_tier=0,inspection_conflict=True)
        lookup['1'].update(inspection_tier=0,inspection_strict_conflict=True,inspection_uncertain=True)
        self.assertEqual(evidence_order(ids,lookup),ids)

    def test_conflicting_cached_verdict_is_unknown_not_inverted_to_match(self):
        raw={'summary':'符合','conditions':[{'term':'室外散步','status':'contradicted','evidence':'人物在户外行走，符合室外散步场景。'}]}
        normalized=normalize_inspection(raw)
        self.assertEqual(normalized['conditions'][0]['status'],'unknown')
        self.assertEqual(raw['conditions'][0]['status'],'contradicted')
        grade=inspection_grade(raw,[{'label':'室外散步','kind':'include'}])
        self.assertEqual(grade['inspection_tier'],1)
        self.assertTrue(grade['inspection_uncertain'])
        self.assertFalse(grade['inspection_strict_conflict'])

    def test_explicit_restrictions_are_distinct_from_semantic_scene(self):
        self.assertFalse(strict_condition({'kind':'include','label':'餐厅吃饭'}))
        self.assertTrue(strict_condition({'kind':'include','label':'餐厅','evidence':'必须在餐厅'}))
        self.assertTrue(strict_condition({'kind':'exclude','label':'生日'}))
        self.assertFalse(strict_condition({'kind':'include','label':'只有一个人','soft':True}))

    def test_existing_ocr_is_reused_and_visual_vectors_precede_optional_ocr(self):
        h=self.e.hybrid;h.save(self.asset,'ocr','previous-provider',{},'文字')
        events=[]
        def embed(*args):events.append('vector');return True
        job={'done':0,'processed':0,'skipped':0,'errors':[],'cancel':threading.Event()}
        with patch.object(h,'_wait',return_value=True),patch('app.hybrid.ollama.description_model',return_value='mock'),patch.object(h,'describe_safely',side_effect=lambda *a:events.append('caption') or {'caption':'a scene'}),patch.object(h,'embed_pending',side_effect=embed),patch.object(h.ocr,'read') as read:
            h.build(self.lib['id'],job)
        self.assertEqual(job['status'],'complete');read.assert_not_called()
        self.assertEqual(events[:2],['caption','vector'])
        self.assertTrue(job['visual_ready'])

    def test_optional_ocr_failure_does_not_block_completed_visual_index(self):
        h=self.e.hybrid;events=[]
        def read(*args):
            self.assertTrue(job['visual_ready']);self.assertEqual(events,['caption','vector'])
            raise RuntimeError('optional OCR unavailable')
        job={'done':0,'processed':0,'skipped':0,'errors':[],'cancel':threading.Event()}
        with patch.object(h,'_wait',return_value=True),patch('app.hybrid.ollama.description_model',return_value='mock'),patch.object(h,'describe_safely',side_effect=lambda *a:events.append('caption') or {'caption':'a scene'}),patch.object(h,'embed_pending',side_effect=lambda *a:events.append('vector') or True),patch.object(h.ocr,'read',side_effect=read):
            h.build(self.lib['id'],job)
        self.assertEqual(job['status'],'complete');self.assertEqual(len(job['errors']),1)
        self.assertEqual(h.coverage(self.lib['id'])['caption'],1)

    def test_empty_text_gate_skips_full_ocr(self):
        ocr=LocalOCR(Path(self.temp.name));ocr.vision=Path(self.temp.name)/'vision';ocr.vision.touch()
        reply=subprocess.CompletedProcess([],0,stdout='{"regions": []}',stderr='')
        with patch('app.providers.ocr.sys.platform','darwin'),patch('app.providers.ocr.subprocess.run',return_value=reply) as run:
            result=ocr.read(Path(self.lib['path'])/'a.jpg')
        self.assertEqual(result['text_gate'],'no_regions');self.assertEqual(result['lines'],[])
        self.assertEqual(run.call_count,1);self.assertIn('detect',run.call_args.args[0])

    def test_unreadable_caption_falls_back_without_stopping_library(self):
        h=self.e.hybrid
        with patch('app.hybrid.ollama.describe',side_effect=[RuntimeError('empty output'),{'caption':'a room'}]) as describe:
            payload=h.describe_safely(Path('unused'),'qwen3.5:4b')
        self.assertEqual(describe.call_count,2)
        self.assertTrue(payload['provenance']['fallback'])

    def test_two_bad_caption_outputs_are_a_file_error_not_global_runtime_error(self):
        with patch('app.hybrid.ollama.describe',side_effect=RuntimeError('empty output')):
            with self.assertRaises(ValueError):self.e.hybrid.describe_safely(Path('unused'),'qwen3.5:4b')


if __name__=='__main__':unittest.main()
