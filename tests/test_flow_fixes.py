import itertools
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from app.engine import Engine
from app.flow.domain import DEFAULTS, fuse, fallback_conditions
from app.flow.indexing import FlowIndexer
from app.flow.people import face_components
from app.flow.service import FlowService
from app.metadata import (OfflinePlaces, aliases, captured_time, date_clues,
                          gps_coordinates, matches_date, metadata_only)
from app.providers import resources
from scripts.eval_retrieval import auroc, existence_summary, lane_timing, oracle_people
from scripts.replay_fusion import replay, compare


class FusionTests(unittest.TestCase):
    def test_family_max_and_rrf_constant(self):
        channels = [{'name': n, 'family': f, 'ids': ids, 'weight': w} for n, f, ids, w in [
            ('p', 'primary', ['b', 'a', 'b'], 1), ('p2', 'primary', ['a'], 1),
            ('s', 'siglip', ['b'], .1)]]
        order, scores, _ = fuse(channels, 10)
        self.assertEqual(order, ['b', 'a'])
        self.assertAlmostEqual(scores['b'], 1.1 / 61)
        self.assertAlmostEqual(scores['a'], 1 / 61)

    def test_replay_tail_primary_and_missing_targets(self):
        r = {'targets': ['a', 'absent'], 'final': {'channel_orders': {
            'primary：原文': ['a', 'b'], 'primary：条件': ['a', 'b'], 'siglip：原文': ['b', 'c', 'a']}}}
        self.assertEqual(replay(r, primary_only=True), ['a', 'b'])
        self.assertEqual(replay(r, tail=1)[0], 'a')
        self.assertEqual(replay(r, weight=.1)[0], 'a')
        row = compare([r, {'targets': [], 'final': r['final']}])[-1]
        self.assertEqual(row['n'], 1)
        self.assertEqual(row['recall@1'], .5)


class MetadataRulesTests(unittest.TestCase):
    def test_dates_and_residual(self):
        for text, yes, no in [('2015年1月', '2015-01-02', '2015-02-01'),
                              ('5月', '2024-05-05', '2024-06-05'),
                              ('2025年11月15日', '2025-11-15', '2025-11-16'),
                              ('2018', '2018-12-31', '2019-01-01'),
                              ('2023年劳动节', '2023-05-02', '2023-06-01'),
                              ('2024年春节', '2024-02-10', '2024-02-09'),
                              ('2025年中秋', '2025-10-06', '2025-09-06'),
                              ('国庆', '2025-10-07', '2025-10-08'),
                              ('元旦', '2025-01-01', '2025-01-02')]:
            with self.subTest(text=text):
                rules, _, warnings = date_clues(text)
                self.assertTrue(matches_date(yes, rules))
                self.assertFalse(matches_date(no, rules))
                self.assertFalse(warnings)
        rules, residual, _ = date_clues('厨房准备食材, 5月')
        self.assertEqual(rules, [(None, 5, None, 1)])
        self.assertFalse(metadata_only(residual))
        self.assertTrue(metadata_only(date_clues('过2023年劳动节拍的照片')[1]))

    def test_invalid_and_unsupported_dates_fail_closed(self):
        rules, _, warnings = date_clues('2025年2月30日')
        self.assertTrue(warnings)
        self.assertFalse(matches_date('2025-02-01', rules))
        self.assertFalse(matches_date(None, rules))
        self.assertTrue(date_clues('2001年春节')[2])
        self.assertFalse(date_clues('2025年春节')[2])

    def test_exif_gps_and_nested_date(self):
        exif = Image.Exif()
        exif[34853] = {1: 'N', 2: (31., 12., 0.), 3: 'E', 4: (121., 30., 0.)}
        exif[34665] = {36867: '2025:11:15 12:00:00'}
        lat, lon = gps_coordinates(exif)
        self.assertAlmostEqual(lat, 31.2)
        self.assertAlmostEqual(lon, 121.5)
        self.assertEqual(captured_time(exif), '2025-11-15T12:00:00')
        exif[34853][1] = b'S'
        self.assertAlmostEqual(gps_coordinates(exif)[0], -31.2)
        exif[34853][2] = (float('nan'), 0, 0)
        self.assertEqual(gps_coordinates(exif), (None, None))
        self.assertEqual(gps_coordinates(Image.Exif()), (None, None))

    def test_place_aliases_landmarks_holes_and_missing_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            outer = [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]
            hole = [[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]]
            for level, name in [('province', '江苏省'), ('city', '南京市'), ('district', '松江区')]:
                (folder / f'china_{level}.geojson').write_text(json.dumps({'features': [{
                    'properties': {'name': name}, 'geometry': {'type': 'Polygon', 'coordinates': [outer, hole]}}]}))
            places = OfflinePlaces(folder)
            self.assertIn('南京市', places.reverse(3, 3))
            self.assertFalse(places.reverse(1.5, 1.5))
            self.assertFalse(places.reverse(30, 121))
            assets = [{'place_names': json.dumps(['南京市', '松江区', '柯桥区', '禹城市', '嘉兴市', '海宁市'])}]
            for text in ['南京市拍的', '南京拍的', '松江区', '在松江拍的', '柯桥拍的', '禹城市', '嘉兴拍的', '海宁拍的']:
                self.assertTrue(places.clues(text, assets)[0], text)
            # City aliases at query start are soft clues under the third-round rule.
            self.assertEqual(places.clues('南京路的照片', assets)[0], {'南京市'})
            self.assertEqual(places.clues('南京博物院', assets)[0], {'南京市'})
            self.assertEqual(OfflinePlaces(folder / 'missing').reverse(30, 120), [])
            self.assertIn('海宁', aliases('海宁市'))


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.engine = Engine(self.root / 'isolated')
        self.flow = FlowService(self.engine)
        self.flow.recall.places = OfflinePlaces(self.root / 'missing-boundaries')
        self.photos = self.root / 'photos'
        self.photos.mkdir()

    def test_config_weights_validation(self):
        c = self.flow.configure({'fusion_weights': {'siglip': .3}})
        self.assertEqual(c['fusion_weights']['siglip'], .3)
        self.assertEqual(c['fusion_weights']['primary'], 1.)
        for value in [[], {'siglip': True}, {'siglip': float('nan')}, {'siglip': float('inf')}, {'siglip': -1}, {'bad-key': 1}]:
            with self.assertRaises(ValueError):
                self.flow.configure({'fusion_weights': value})
        self.assertEqual(DEFAULTS['fusion_weights']['siglip'], .1)

    def test_old_database_migrates(self):
        with self.engine.db() as con:
            for column in ('latitude', 'longitude', 'place_names', 'metadata_version'):
                con.execute('ALTER TABLE assets DROP COLUMN ' + column)
        migrated = Engine(self.root / 'isolated')
        with migrated.db() as con:
            columns = {r[1] for r in con.execute('PRAGMA table_info(assets)')}
        self.assertTrue({'latitude', 'longitude', 'place_names', 'metadata_version'} <= columns)

    def test_scan_metadata_backfill_unchanged_preserves_vectors(self):
        exif = Image.Exif()
        exif[34853] = {1: 'N', 2: (31., 12., 0.), 3: 'E', 4: (121., 30., 0.)}
        exif[34665] = {36867: '2025:11:15 12:00:00'}
        Image.new('RGB', (20, 20)).save(self.photos / 'a.jpg', exif=exif)
        lib = self.engine.add_library(self.photos)
        self.engine.index_library(lib['id'])
        before = self.flow.catalog(lib['id'])[0]
        self.assertAlmostEqual(before['latitude'], 31.2)
        self.assertEqual(before['captured'], '2025-11-15T12:00:00')
        with self.engine.db() as con:
            con.execute("UPDATE assets SET latitude=NULL,longitude=NULL,place_names=NULL,metadata_version=NULL,embedding=? WHERE id=?", (b'unchanged', before['id']))
        with patch('app.engine.OfflinePlaces') as cls:
            cls.return_value.reverse.return_value = ['上海市', '松江区']
            cls.return_value.version = 'test-boundaries'
            job = self.engine.index_library(lib['id'])
        after = self.flow.catalog(lib['id'])[0]
        self.assertEqual(job['skipped'], 1)
        self.assertEqual(before['fingerprint'], after['fingerprint'])
        self.assertEqual(after['embedding'], b'unchanged')
        self.assertEqual(json.loads(after['place_names']), ['上海市', '松江区'])
        self.assertAlmostEqual(after['longitude'], 121.5)

    def test_pure_metadata_no_models_and_mixed_visual_order(self):
        assets = [{'id': str(i), 'fingerprint': 'f', 'library_id': 'lib', 'captured': '2025-05-01',
                   'place_names': '["南京市"]'} for i in range(3)]
        state = lambda t: {'conditions': fallback_conditions(t), 'excluded': []}
        with patch.object(self.flow.recall, 'primary', side_effect=AssertionError('不得加载模型')):
            result = self.flow.recall.search(assets, state('南京市拍的，5月'), self.flow.store.config(), 200)
            self.assertEqual(set(result[0]), {'0', '1', '2'})
            self.assertEqual({c['family'] for c in result[-1]}, {'metadata_time', 'metadata_place'})
            self.assertEqual(self.flow.recall.search(assets, state('6月'), self.flow.store.config(), 200)[0], [])
        provider = Mock(installed=True, version='v')
        rows = [{'asset_id': str(i), 'vector': np.array([float(i), 1], dtype=np.float32).tobytes()} for i in range(3)]
        with patch.object(self.flow.recall, 'primary', return_value=provider), \
             patch.object(self.flow.recall, 'rows', return_value=rows), \
             patch.object(self.flow.recall, 'vector', return_value=np.array([1, 0], dtype=np.float32)), \
             patch.object(self.engine.hybrid, 'rankings', return_value={}):
            result = self.flow.recall.search(assets, state('厨房准备食材, 5月'), self.flow.store.config(), 200)
        self.assertEqual(result[0], ['2', '1', '0'])
        self.assertIn('metadata_time', {c['family'] for c in result[-1]})
        signal = next(c['existence_signal'] for c in result[-1] if 'existence_signal' in c)
        self.assertEqual(signal['primary_top1'], 2)
        self.assertIsNone(signal['primary_gap1_10'])

    def test_clustering_reuse_and_named_identity(self):
        Image.new('RGB', (20, 20)).save(self.photos / 'a.jpg')
        lib = self.engine.add_library(self.photos)
        self.engine.index_library(lib['id'])
        a = self.flow.catalog(lib['id'])[0]
        with self.engine.db() as con:
            for i, pid in enumerate(['named', 'new']):
                con.execute('INSERT INTO flow_people VALUES (?,?)', (pid, '用户命名' if i == 0 else ''))
                con.execute('INSERT INTO flow_faces VALUES (?,?,?,?,?,?,?)',
                    (a['id'], a['fingerprint'], str(i), pid, np.array([1, 0], dtype=np.float32).tobytes(), '[]', self.flow.tools.faces.version))
        indexer = FlowIndexer(self.flow)
        indexer._cluster_people(lib['id'])
        indexer._cluster_people(lib['id'])
        with self.engine.db() as con:
            self.assertEqual({r[0] for r in con.execute('SELECT person_id FROM flow_faces')}, {'named'})
            self.assertEqual(con.execute("SELECT name FROM flow_people WHERE id='named'").fetchone()[0], '用户命名')

    def test_build_records_lane_timing(self):
        Image.new('RGB', (20, 20)).save(self.photos / 'a.jpg')
        lib = self.engine.add_library(self.photos)
        self.engine.index_library(lib['id'])
        self.flow.configure({'people_enabled': True})
        self.flow.tools.faces.faces = lambda image: []
        indexer = FlowIndexer(self.flow)
        jobs = []
        for _ in range(2):
            job = {'library_id': lib['id'], 'status': 'running', 'done': 0, 'processed': 0, 'skipped': 0,
                   'errors': [], 'cancel': threading.Event()}
            indexer.build(job, ['people'])
            jobs.append(job)
        first, second = (j['timing']['people'] for j in jobs)
        self.assertEqual((first['processed'], first['skipped'], first['errors']), (1, 0, 0))
        self.assertIn('cluster_seconds', first)
        self.assertEqual((second['processed'], second['skipped']), (0, 1))
        lanes = lane_timing(jobs[0]['timing'])
        self.assertIsNotNone(lanes['people']['seconds_per_photo'])
        self.assertIsNone(lane_timing(jobs[1]['timing'])['people']['seconds_per_photo'])

    def test_oracle_is_explicit_ephemeral_and_scoped(self):
        Image.new('RGB', (20, 20)).save(self.photos / 'a.jpg')
        lib = self.engine.add_library(self.photos)
        self.engine.index_library(lib['id'])
        a = self.flow.catalog(lib['id'])[0]
        with self.engine.db() as con:
            con.execute("INSERT INTO flow_people VALUES ('group','')")
            con.execute('INSERT INTO flow_faces VALUES (?,?,?,?,?,?,?)', (a['id'], a['fingerprint'], '0', 'group',
                np.array([1, 0], dtype=np.float32).tobytes(), '[]', self.flow.tools.faces.version))
        suite = {'name': 'photobench_test', 'queries': [{'turns': ['我和晓彤'], 'targets': ['a.jpg'], 'tags': {'Person': 'fact'}}]}
        report = oracle_people(self.flow, lib, suite, {'a.jpg': a['id']}, ['晓彤', '浩哥'])
        self.assertTrue(report['enabled'])
        self.assertEqual(self.flow.recall.oracle_people['晓彤'], [a['id']])
        self.assertIsNone(report['names']['浩哥']['group'])
        with self.engine.db() as con:
            self.assertEqual(con.execute("SELECT name FROM flow_people WHERE id='group'").fetchone()[0], '')
        with self.assertRaises(ValueError):
            oracle_people(self.flow, lib, {**suite, 'name': 'flickr1k'}, {}, ['晓彤'])

    def test_reuses_exact_legacy_primary_vector(self):
        Image.new('RGB', (20, 20)).save(self.photos / 'a.jpg')
        lib = self.engine.add_library(self.photos)
        self.engine.index_library(lib['id'])
        a = self.flow.catalog(lib['id'])[0]
        vector = np.array([1, 0], dtype=np.float32).tobytes()
        with self.engine.db() as con:
            con.execute('INSERT INTO feature_vectors VALUES (?,?,?,?,?,?,?,?)', (a['id'], a['fingerprint'], 'test-v', -1, 'null', vector, 2, 0))
        provider = Mock()
        FlowIndexer(self.flow)._vector(a, provider, 'test-v', None, 'full')
        provider.image.assert_not_called()
        with self.engine.db() as con:
            self.assertEqual(con.execute('SELECT vector FROM flow_vectors WHERE asset_id=?', (a['id'],)).fetchone()[0], vector)


class PeopleAndMetricsTests(unittest.TestCase):
    def test_components_order_independent_transitive(self):
        rows = [{'key': k, 'vector': np.array(v)} for k, v in [('a', [1, 0]), ('b', [.7, .7]), ('c', [0, 1]), ('d', [-1, 0])]]
        expected = [['a', 'b', 'c'], ['d']]
        for perm in itertools.permutations(rows):
            self.assertEqual(face_components(perm), expected)
        self.assertEqual(face_components([]), [])
        with self.assertRaises(ValueError):
            face_components([{'key': 'x', 'vector': np.array([0, 0])}])

    def test_auroc_ties_missing_and_orientation(self):
        self.assertEqual(auroc([2, 3], [0, 1]), 1)
        self.assertEqual(auroc([0], [1]), 0)
        self.assertEqual(auroc([1, 1], [1]), .5)
        self.assertEqual(auroc([1, 2], [1, 3]), .375)
        self.assertIsNone(auroc([], [1]))
        self.assertIsNone(auroc([float('nan')], [1]))
        rows = [{'no_target': no, 'final': {'existence_signal': signal}} for no, signal in [
            (False, {'primary_top1': .8}), (True, {'primary_top1': .2}), (False, {})]]
        out = existence_summary(rows)['primary_top1']
        self.assertEqual(out['auroc'], 1)
        self.assertEqual(out['with_target']['missing'], 1)

    def test_model_residency_switch_and_default(self):
        owner, other = Mock(), Mock()
        with patch.object(resources, 'PROVIDERS', [owner, other]), patch.dict(os.environ, {'PHOTORECALL_KEEP_MODELS': '1'}):
            with resources.inference(owner):
                pass
        other.unload.assert_not_called()
        with patch.object(resources, 'PROVIDERS', [owner, other]), patch.dict(os.environ, {'PHOTORECALL_KEEP_MODELS': '0', 'PHOTORECALL_DISABLE_OLLAMA': '1'}):
            with resources.inference(owner):
                pass
        other.unload.assert_called_once()
        owner.unload.assert_not_called()
