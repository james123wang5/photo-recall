import unittest
from app.matching import match_decision, match_summary
from app.engine import Engine


class MatchingTests(unittest.TestCase):
    conditions=[{'label':'打球','kind':'include'},{'label':'只有一个人','kind':'include'}]

    def photo(self,*statuses,summary=''):
        return {'inspection':{'summary':summary,'conditions':[{'term':c['label'],'status':s,'evidence':'可见依据'}
            for c,s in zip(self.conditions,statuses)]}}

    def test_high_retrieval_or_core_score_without_inspection_is_not_admitted(self):
        self.assertEqual(match_decision({'score':.99,'rerank_score':.999},self.conditions)['match_status'],'unreviewed')

    def test_every_required_condition_needs_positive_evidence(self):
        self.assertEqual(match_decision(self.photo('matched','matched'),self.conditions)['match_status'],'matched')
        self.assertEqual(match_decision(self.photo('matched'),self.conditions)['match_status'],'uncertain')
        self.assertEqual(match_decision(self.photo('matched','unknown'),self.conditions)['match_status'],'uncertain')
        self.assertEqual(match_decision(self.photo('matched','contradicted'),self.conditions)['match_status'],'not_matched')

    def test_inconsistent_model_cannot_pass_admission(self):
        p={'inspection':{'summary':'符合','conditions':[{'term':'打球','status':'contradicted','evidence':'符合打球场景。'}]}}
        self.assertEqual(match_decision(p,self.conditions[:1])['match_status'],'uncertain')

    def test_preferences_do_not_exclude_a_required_match(self):
        conditions=self.conditions[:1]+[{'label':'蓝衣服','kind':'prefer','soft':True}]
        p={'inspection':{'conditions':[{'term':'打球','status':'matched'},{'term':'蓝衣服','status':'unknown'}]}}
        self.assertEqual(match_decision(p,conditions)['match_status'],'matched')

    def test_explicit_exclusion_still_requires_verification(self):
        conditions=[{'label':'人群','kind':'exclude'}]
        p={'inspection':{'conditions':[{'term':'不含 人群','status':'contradicted'}]}}
        self.assertEqual(match_decision(p,conditions)['match_status'],'not_matched')

    def test_no_query_is_browsing_not_filtering(self):
        self.assertEqual(match_decision({},[])['match_status'],'browse')
        self.assertEqual(match_summary([{'match_status':'matched'},{'match_status':'uncertain'},{}]),
            {'matched':1,'uncertain':1,'unreviewed':1,'not_matched':0,'browse':0})

    def test_manual_updates_recompute_admission_even_with_stale_flags(self):
        from app.pipeline import RetrievalPipeline
        import threading
        class Fake:
            job_lock=threading.Lock()
            verifications={'s':{'pipeline':'vnext','key':'k','conditions':self.conditions,
                'results':[{'id':'p','match_status':'matched'}],'revision':0}}
        pipeline=object.__new__(RetrievalPipeline);pipeline.engine=Fake()
        pipeline.publish_inspections('s','k',[{'id':'p',**self.photo('matched','unknown')}])
        self.assertEqual(pipeline.engine.verifications['s']['results'][0]['match_status'],'uncertain')


if __name__=='__main__':unittest.main()
