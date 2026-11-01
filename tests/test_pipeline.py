import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch, PropertyMock
import numpy as np
from PIL import Image
from app.engine import Engine
from app.pipeline import fuse, query_text, select_candidates, merge_reranked
from app.retrieval import percentile
from app.providers.multimodal import CoreReranker


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.folder = root / 'photos'
        self.folder.mkdir()
        self.engine = Engine(root / 'data')
        self.library = self.engine.add_library(str(self.folder))
        with self.engine.db() as con:
            con.execute("INSERT OR REPLACE INTO settings VALUES ('rerank_batch_size','1')")
            con.execute("INSERT OR REPLACE INTO settings VALUES ('hybrid_search','false')")

    def tearDown(self):
        self.temp.cleanup()

    def assets(self, count=3):
        for i in range(count):
            Image.new('RGB', (40,30), 'green').save(self.folder/f'{i}.png')
        self.engine.index_library(self.library['id'])
        with self.engine.db() as con:
            return [dict(r) for r in con.execute('SELECT * FROM assets WHERE library_id=?',(self.library['id'],))]

    def test_reserved_entry_preserves_local_candidate_and_fusion_deduplicates(self):
        rankings={'main':[f'a{i}' for i in range(300)],'local':['small-hand','a1']}
        order,scores,sources=fuse(rankings)
        selected=select_candidates(rankings,order,64)
        self.assertIn('small-hand',selected)
        self.assertEqual(len(selected),64)
        self.assertEqual(order.count('a1'),1)
        self.assertEqual(len(sources['a1']),2)

    def test_weak_checked_items_do_not_push_unchecked_tail_behind_entire_batch(self):
        order=['a','b','c','d','e']
        ranked={'a':.01,'b':.02,'e':.9}
        lookup={aid:{} for aid in order}
        self.assertEqual(merge_reranked(order,ranked,lookup),['e','c','d','a','b'])
        lookup['e']['inspection_tier']=0
        self.assertEqual(merge_reranked(order,ranked,lookup)[0],'e')

    def test_positive_tail_is_promoted_and_weak_head_is_demoted(self):
        order=[str(i) for i in range(100)]
        lookup={aid:{} for aid in order}
        ranked={'0':.1,'3':.8,'90':.95}
        result=merge_reranked(order,ranked,lookup)
        self.assertLess(result.index('90'),90)
        self.assertGreaterEqual(result.index('90'),66)
        self.assertLess(result.index('3'),3)
        self.assertLessEqual(result.index('0'),24)
        self.assertEqual(len(set(result)),100)

    def test_selection_audits_beyond_top64(self):
        order=[str(i) for i in range(500)]
        selected=select_candidates({'main':order},order,64)
        self.assertTrue(any(int(aid)>100 for aid in selected))
        self.assertEqual(len(selected),64)
        self.assertEqual(len(set(selected)),64)

    def test_text_vectors_survive_provider_reload_and_are_versioned(self):
        p=self.engine.pipeline
        with patch.object(p.main,'text',return_value=np.ones(4,dtype=np.float32)) as encode:
            p.text_vector(p.main,'v1','test')
            p.text_vector(p.main,'v1','test')
            self.assertEqual(encode.call_count,1)
            p.text_vector(p.main,'v2','test')
            self.assertEqual(encode.call_count,2)

    def test_soft_cue_relaxes_but_negation_is_preserved(self):
        conditions=[{'label':'一个人','english':'exactly one person','kind':'include'},
                    {'label':'蓝车','english':'a blue car','kind':'prefer'},
                    {'label':'猫','english':'a cat','kind':'exclude'}]
        text=query_text(conditions,True,True)
        self.assertIn('exactly one person',text)
        self.assertIn('without a cat',text)
        self.assertNotIn('blue',text)

    def test_percentiles_preserve_ties_and_old_semantics(self):
        values=np.array([4,2,4,9,-1,2],dtype=float)
        expected=[(np.sum(values<x)+.5*np.sum(values==x))/len(values) for x in values]
        np.testing.assert_allclose(percentile(values),expected)

    def test_stale_fingerprint_excludes_vectors(self):
        a=self.assets()[0]
        p=self.engine.pipeline
        with self.engine.db() as con:
            con.execute('INSERT INTO feature_vectors VALUES (?,?,?,?,?,?,?,?)',
                (a['id'],'stale',p.main.version,-1,'null',np.ones(2048,dtype=np.float32).tobytes(),2048,time.time()))
        self.assertEqual(p.counts(self.library['id'])['main'],0)

    def test_incremental_versioned_index_reuses_work_and_does_not_modify_original(self):
        self.assets(2)
        p=self.engine.pipeline
        job={'status':'running','errors':[],'processed':0,'skipped':0,'done':0,'cancel':threading.Event()}
        images_before={f:f.read_bytes() for f in self.folder.glob('*')}
        with patch.object(p.main,'image',return_value=np.ones(2048,dtype=np.float32)),patch.object(self.engine.provider,'image',return_value=np.ones(768,dtype=np.float32)):
            p.index_library(self.library['id'],job)
            self.assertEqual((p.counts(self.library['id'])['main'],p.counts(self.library['id'])['local']),(2,8))
            job.update(done=0,processed=0,skipped=0)
            p.index_library(self.library['id'],job)
            self.assertEqual(job['skipped'],4)
        for f,content in images_before.items():self.assertEqual(f.read_bytes(),content)

    def test_rerank_does_not_hide_unverified_candidates_and_reuses_query_cache(self):
        assets=self.assets(5)
        p=self.engine.pipeline
        s=self.engine.session(self.library['id'])
        s['conditions']=[{'label':'有人','kind':'include','concept':'person'}]
        rankings={'full':[a['id'] for a in assets]}
        order,scores,sources=fuse(rankings)
        def make_job():
            return {'status':'running','query':'有人','started':time.time(),'cancel':threading.Event(),
                    'results':[],'done':0,'errors':[],'indexed':{},'confirmed':0}
        with patch.object(p,'recall',return_value=(rankings,(order,scores,sources))),patch.object(type(p.reranker),'installed',new_callable=PropertyMock,return_value=True),patch.object(p.reranker,'score',side_effect=[.1,.8]) as score:
            job=make_job();p.refine(s,assets,{},job,budget=2)
            self.assertEqual((job['status'],len(job['results']),job['done']),('complete',5,2))
            job=make_job();p.refine(s,assets,{},job,budget=2)
            self.assertEqual(score.call_count,2)
            self.assertEqual(len(job['results']),5)

    def test_cancelled_job_does_not_call_model(self):
        assets=self.assets(1);p=self.engine.pipeline;s=self.engine.session(self.library['id'])
        s['conditions']=[{'label':'有人','kind':'include'}]
        cancel=threading.Event();cancel.set()
        job={'cancel':cancel,'status':'running','query':'有人','started':time.time(),'results':[],
             'done':0,'errors':[],'indexed':{}}
        with patch.object(p,'recall',return_value=({},([],{},{}))),patch.object(p.reranker,'score') as score:
            p.refine(s,assets,{},job)
            score.assert_not_called();self.assertEqual(job['status'],'paused')

    def test_cached_count_conflict_is_visible_but_not_sent_to_expensive_reranker(self):
        assets=self.assets(2);p=self.engine.pipeline;s=self.engine.session(self.library['id'])
        s['conditions']=[{'label':'只有一个人','kind':'include','constraint':{'entity':'person','op':'eq','count':1}}]
        with self.engine.db() as con:
            con.execute('INSERT INTO detections VALUES (?,?,?,?)',(assets[0]['id'],assets[0]['fingerprint'],self.engine.detector.version,
                json.dumps({'person':{'lower_bound':3,'possible_extra':0}})))
        rankings={'full':[a['id'] for a in assets]};order,scores,sources=fuse(rankings)
        job={'status':'running','query':'只有一个人','started':time.time(),'cancel':threading.Event(),
             'results':[],'done':0,'errors':[],'indexed':{},'confirmed':0}
        with patch.object(p,'recall',return_value=(rankings,(order,scores,sources))),patch.object(type(p.reranker),'installed',new_callable=PropertyMock,return_value=True),patch.object(p.reranker,'score',return_value=.5) as score:
            p.refine(s,assets,{},job,budget=2)
            self.assertEqual(score.call_count,1)
            self.assertEqual(len(job['results']),2)
            self.assertEqual(job['results'][-1]['inspection_tier'],0)
            self.assertNotIn('inspection_tier',job['results'][0])

    def test_fuzzy_date_and_missing_date_never_become_negative_evidence(self):
        asset=self.assets(1)[0];p=self.engine.pipeline;r={}
        p.apply_cached_counts(r,asset,[{'label':'2020','concept':'year','kind':'prefer','soft':True}])
        self.assertNotIn('inspection_tier',r)
        p.apply_cached_counts(r,asset,[{'label':'2020','concept':'year','kind':'include'}])
        self.assertNotIn('inspection_tier',r)
        asset['captured']='2018-01-01';p.apply_cached_counts(r,asset,[{'label':'2020','concept':'year','kind':'include'}])
        self.assertEqual(r['inspection_tier'],0)

    def test_same_filename_in_different_folders_keeps_distinct_identity(self):
        for directory in ['a','b']:
            folder=self.folder/directory;folder.mkdir();Image.new('RGB',(20,20)).save(folder/'same.jpg')
        self.engine.index_library(self.library['id'])
        with self.engine.db() as con:
            rows=con.execute('SELECT id,name FROM assets WHERE library_id=?',(self.library['id'],)).fetchall()
        self.assertEqual(len({r['id'] for r in rows}),2)
        self.assertEqual(len({r['name'] for r in rows}),1)

    def test_planner_cannot_silently_drop_uncovered_user_clause(self):
        parsed=[{'label':'生日照片','english':'birthday photo','kind':'include','concept':'scene','evidence':'生日照片'},
                {'label':'灯光很暗','english':'dim lighting','kind':'include','concept':'lighting','evidence':'灯光很暗'}]
        with patch('app.providers.ollama.plan_search',return_value=parsed):
            conditions=self.engine.parse_conditions([], '生日照片，有蛋糕和蜡烛，灯光很暗')
        self.assertIn('有蛋糕和蜡烛',[c['label'] for c in conditions])

    def test_reranker_sees_camera_orientation_consistent_with_gallery(self):
        import torch
        from types import SimpleNamespace
        image=Image.new('RGB',(20,30),'red');exif=Image.Exif();exif[274]=8
        path=self.folder/'rotated.jpg';image.save(path,exif=exif)
        p=CoreReranker(Path(self.temp.name)/'unused');p.torch=torch
        p.model=lambda **kw:SimpleNamespace(last_hidden_state=torch.zeros((1,1,2)))
        p.score_weight=torch.ones(2)
        seen=[]
        def inputs(messages,generation):
            seen.append(messages[1]['content'][-1]['image'].size)
            return {'input_ids':torch.ones((1,2),dtype=torch.long)}
        with patch.object(p,'_load'),patch.object(p,'_inputs',side_effect=inputs):
            p.score('left side of the photograph',str(path))
        self.assertEqual(seen,[(30,20)])
        p.unload()


if __name__ == '__main__':unittest.main()
