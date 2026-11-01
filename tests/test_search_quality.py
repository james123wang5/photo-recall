import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch, PropertyMock

import numpy as np
from PIL import Image

from app.engine import Engine
from app.query import update_conditions, semantic_query
from app.retrieval import semantic_scores, inspection_grade, rank_key, count_evidence, DESCRIPTION_FIELDS
from app.providers import ollama


class SearchQualityTests(unittest.TestCase):
    def test_known_word_does_not_swallow_arbitrary_detail(self):
        message = "灯光比较暗，然后有人在许愿，人物在画面右侧，拍摄角度偏低"
        conditions = update_conditions([], message)
        self.assertIn(message, semantic_query(conditions))

    def test_open_vocabulary_parser_keeps_composition_and_angle(self):
        message = "逆光的人物剪影，主体在右下角，从低角度仰拍"
        response = {"conditions": [
            {"label": "逆光人物剪影", "english": "a backlit silhouette", "slot": "lighting", "kind": "include", "evidence": "逆光的人物剪影"},
            {"label": "主体在右下角", "english": "subject in the bottom right", "slot": "composition", "kind": "include", "evidence": "主体在右下角"},
            {"label": "低角度仰拍", "english": "a low-angle upward view", "slot": "viewpoint", "kind": "include", "evidence": "低角度仰拍"},
            {"label": "穿蓝衣服", "english": "wearing blue", "slot": "color", "kind": "include", "evidence": "蓝色衣服"}]}
        with patch.object(ollama, "installed_models", return_value=[ollama.MODEL]), patch.object(ollama, "request_json", return_value=response):
            parsed = ollama.plan_search([], message)
        self.assertEqual([x["slot"] for x in parsed], ["lighting", "composition", "viewpoint"])
        self.assertNotIn("穿蓝衣服", [x["label"] for x in parsed])

    def test_brightness_is_supporting_evidence_for_equal_semantics(self):
        vector = np.array([1, 0], dtype=np.float32).tobytes()
        assets = [{"id": "dim", "embedding": vector, "visual_stats": json.dumps({"luminance": .1, "dark_fraction": .8})},
                  {"id": "bright", "embedding": vector, "visual_stats": json.dumps({"luminance": .8, "dark_fraction": .05})}]
        class Provider:
            def text(self, query):
                return np.array([1, 0], dtype=np.float32)
        conditions = [{"label": "灯光昏暗", "english": "dim lighting", "kind": "include", "concept": "lighting"}]
        scores, _ = semantic_scores(assets, conditions, Provider(), {})
        self.assertGreater(scores[0], scores[1])
        neutral, _ = semantic_scores(assets, [{"label": "有人", "english": "people", "kind": "include", "concept": "person"}], Provider(), {})
        self.assertAlmostEqual(neutral[0], neutral[1])

    def test_verified_contradiction_does_not_get_promoted_above_unchecked(self):
        conditions = [{"label": "灯光暗", "kind": "include"}, {"label": "合十", "kind": "include"}]
        bad = {"id": "bright", "score": .95, **inspection_grade({"conditions": [{"term": "灯光暗", "status": "contradicted"}, {"term": "合十", "status": "matched"}]}, conditions)}
        unverified = {"id": "unknown", "score": .6}
        good = {"id": "right", "score": .5, **inspection_grade({"conditions": [{"term": "灯光暗", "status": "matched"}, {"term": "合十", "status": "matched"}]}, conditions)}
        self.assertEqual([x["id"] for x in sorted([bad, unverified, good], key=rank_key)], ["right", "unknown", "bright"])

    def test_viewpoint_description_changes_ranking_for_different_questions(self):
        vector = np.array([1, 0], dtype=np.float32).tobytes()
        assets = [{"id": "one", "embedding": vector}, {"id": "two", "embedding": vector}]
        annotations = {}
        for asset, value in zip(assets, [[1, 0], [0, 1]]):
            fields = np.zeros((len(DESCRIPTION_FIELDS), 2), dtype=np.float32)
            fields[DESCRIPTION_FIELDS.index("viewpoint")] = value
            annotations[asset["id"]] = {"field_vectors": fields.tobytes()}
        class Provider:
            def text(self, query):
                return np.array([0, 1] if "high-angle" in query else [1, 0], dtype=np.float32)
        low, _ = semantic_scores(assets, [{"label": "低角度", "english": "low-angle view", "slot": "viewpoint", "kind": "include"}], Provider(), annotations)
        high, _ = semantic_scores(assets, [{"label": "俯拍", "english": "high-angle view", "slot": "viewpoint", "kind": "include"}], Provider(), annotations)
        self.assertGreater(low[0], low[1]);self.assertGreater(high[1], high[0])

    def test_detailed_analysis_is_incremental_and_covers_any_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);source = root / "photos";source.mkdir()
            for name in ["arbitrary-1.png", "arbitrary-2.png"]:
                Image.new("RGB", (50, 40), "green").save(source / name)
            engine = Engine(root / "state");lib = engine.add_library(str(source));engine.index_library(lib["id"])
            def job():
                return {"status": "running", "done": 0, "processed": 0, "skipped": 0, "errors": [], "cancel": threading.Event()}
            with patch.object(ollama, "describe", return_value={"caption": "Green image with no people.", "lighting": "bright", "composition": "uniform", "viewpoint": "unknown", "objects": "unknown", "people": "none", "actions": "none", "colors": "green", "visible_text": "none"}) as describe:
                first = job();engine.analyze_library(lib["id"], first)
                second = job();engine.analyze_library(lib["id"], second)
                self.assertEqual((first["processed"], second["skipped"], describe.call_count), (2, 2, 2))
            self.assertEqual(next(x for x in engine.libraries() if x["id"] == lib["id"])["detailed_count"], 2)

    def test_old_session_recovers_original_missing_lighting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);source = root / "photos";source.mkdir()
            engine = Engine(root / "state");lib = engine.add_library(str(source));session = engine.session(lib["id"])
            session["conditions"] = [{"id": "old", "concept": "person", "label": "人物", "kind": "include"}]
            session["messages"] = [{"role": "user", "text": "生日照片"}, {"role": "user", "text": "灯光暗，有人许愿"}]
            engine.save_session(session)
            with engine.db() as con:
                con.execute("INSERT OR REPLACE INTO settings VALUES ('semantic','true')")
            response = [{"label": "灯光暗", "english": "dim lighting", "concept": "lighting", "kind": "include"}]
            with patch.object(ollama, "plan_search", return_value=response) as planner:
                result = engine.search(lib["id"], session_id=session["id"])
            self.assertIn("灯光暗，有人许愿", planner.call_args.args[1])
            self.assertEqual(result["session"]["conditions"][0]["label"], "灯光暗")
            self.assertEqual(result["session"]["plan"]["version"], 4)

    def test_numeric_constraints_are_not_fuzzy_similarity(self):
        conditions = [{"label": "只有一个人", "kind": "include", "constraint": {"entity": "person", "op": "eq", "count": 1}},
                      {"label": "至少两辆车", "kind": "include", "constraint": {"entity": "car", "op": "gte", "count": 2}},
                      {"label": "没有猫", "kind": "include", "constraint": {"entity": "cat", "op": "eq", "count": 0}}]
        checked = count_evidence(conditions, {"person": {"count": 3}, "car": {"count": 2}, "cat": {"count": None}})
        self.assertEqual([x["status"] for x in checked.values()], ["contradicted", "matched", "unknown"])
        self.assertEqual(inspection_grade({"conditions": list(checked.values())}, conditions)["inspection_tier"], 0)
        bounded=count_evidence(conditions,{'person':{'count':None,'lower_bound':3}})
        self.assertEqual(bounded['只有一个人']['status'],'contradicted')
        absent=[{'label':'没有人','kind':'include','constraint':{'entity':'person','op':'eq','count':0}}]
        self.assertEqual(count_evidence(absent,{'person':{'count':None,'lower_bound':0}})['没有人']['status'],'unknown')
        self.assertEqual(ollama.explicit_quantity({'label':'至少两个人','evidence':'至少两个人','english':'at least two people'}),{'entity':'person','op':'gte','count':2})
        self.assertEqual(ollama.explicit_quantity({'label':'不超过三只猫','evidence':'不超过三只猫','english':'at most three cats'}),{'entity':'cat','op':'lte','count':3})
        self.assertIsNone(ollama.explicit_quantity({'label':'有人','evidence':'有人','english':'two people'}))

    def test_count_facts_reused_across_queries_and_failure_skips_full_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'photos';source.mkdir();Image.new('RGB',(30,30),'red').save(source/'generic.jpg')
            engine=Engine(root/'data');lib=engine.add_library(str(source));engine.index_library(lib['id'])
            asset=engine.search(lib['id'])['results'][0];session=engine.session(lib['id'])
            session['conditions']=[{'label':'只有一个人','kind':'include','concept':'people','constraint':{'entity':'person','op':'eq','count':1}}];engine.save_session(session)
            with patch.object(ollama,'count_entities',return_value={'person':{'count':3,'evidence':'左中右三个人'}}) as counts, patch.object(ollama,'inspect') as inspect:
                first=engine.inspect_candidates(session['id'],[asset['id']])[0]
                self.assertEqual(first['inspection_tier'],0);inspect.assert_not_called()
                session['conditions'][0]['label']='恰好两个人';session['conditions'][0]['constraint']['count']=2;engine.save_session(session)
                engine.inspect_candidates(session['id'],[asset['id']]);self.assertEqual(counts.call_count,1)

    def test_possible_background_person_prevents_false_single_person_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'photos';source.mkdir();Image.new('RGB',(30,30),'blue').save(source/'generic.jpg')
            engine=Engine(root/'data');lib=engine.add_library(str(source));engine.index_library(lib['id']);asset=engine.search(lib['id'])['results'][0];session=engine.session(lib['id'])
            session['conditions']=[{'label':'恰好一个人','concept':'person','kind':'include','constraint':{'entity':'person','op':'eq','count':1}}];engine.save_session(session)
            with patch.object(type(engine.detector),'ready',new_callable=PropertyMock,return_value=True), patch.object(engine,'detect_asset',return_value={'person':{'lower_bound':1,'possible_extra':1}}), patch.object(ollama,'count_entities',return_value={'person':{'count':1,'evidence':'只看到主体'}}), patch.object(ollama,'inspect',return_value={'summary':'符合','conditions':[{'term':'恰好一个人','status':'matched','evidence':'一个人'}]}):
                result=engine.inspect_candidates(session['id'],[asset['id']])[0]
                self.assertEqual(result['inspection_tier'],1)
                self.assertEqual(result['inspection']['conditions'][0]['status'],'unknown')

    def test_automatic_verification_rejects_false_top_result_and_cancels_stale_job(self):
        with tempfile.TemporaryDirectory() as directory:
            engine=Engine(Path(directory));session={'id':'test','conditions':[{'label':'一个人','kind':'include'}]}
            candidates=[{'id':'group'}, {'id':'single'}]
            def inspected(sid,ids,expected_key=None):
                return [{'id':ids[0],'inspection_tier':0 if ids[0]=='group' else 2}]
            with patch.object(ollama,'installed_models',return_value=[ollama.MODEL]), patch.object(threading.Thread,'start'), patch.object(engine,'inspect_candidates',side_effect=inspected):
                engine.start_verification(session,candidates);job=engine.verifications['test'];engine.verify_search('test',['group','single'],job)
                self.assertEqual((job['confirmed'],job['done'],job['status']),(1,2,'complete'))
                session['conditions']=[];engine.start_verification(session,candidates)
                self.assertTrue(job['cancel'].is_set());self.assertFalse(engine.verification_state('test')['enabled'])

    def test_quantity_prefilter_reaches_match_beyond_first_48_semantic_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'photos';source.mkdir()
            for i in range(101):Image.new('RGB',(12,12),'green').save(source/f'{i:03}.png')
            engine=Engine(root/'data');lib=engine.add_library(str(source));engine.index_library(lib['id'])
            assets=sorted(engine.search(lib['id'])['results'],key=lambda a:a['name'])
            # Request all fixtures, rather than the default 100 response limit.
            assets=sorted(engine.search(lib['id'],limit=1000)['results'],key=lambda a:a['name'])
            target=assets[-1]['id'];session=engine.session(lib['id'])
            session['conditions']=[{'label':'只有一个人','kind':'include','constraint':{'entity':'person','op':'eq','count':1}}]
            def detect(asset):return {'person':{'lower_bound':1 if asset['id']==target else 2}}
            with patch.object(type(engine.detector),'ready',new_callable=PropertyMock,return_value=True), patch.object(ollama,'installed_models',return_value=[ollama.MODEL]), patch.object(threading.Thread,'start'), patch.object(engine,'detect_asset',side_effect=detect), patch.object(engine,'inspect_candidates',return_value=[{'id':target,'inspection_tier':2}]) as inspect:
                engine.start_verification(session,assets);job=engine.verifications[session['id']]
                engine.verify_search(session['id'],[a['id'] for a in assets],job)
                self.assertEqual((job['prefilter_done'],job['confirmed'],job['total']),(101,1,1))
                self.assertEqual(inspect.call_count,1)
                self.assertEqual(sum(r['inspection_tier']==0 for r in job['results']),100)


if __name__ == "__main__":
    unittest.main()
