import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from app.engine import Engine
from app.flow.domain import fallback_conditions
from app.flow.service import FlowService
from app.metadata import OfflinePlaces, aliases, date_clues, matches_date, metadata_only, _ring_contains
from scripts import prepare_geography as prepare


def feature(name, bbox, **properties):
    x1, y1, x2, y2 = bbox
    return {'type': 'Feature', 'properties': {'name': name, **properties}, 'geometry': {
        'type': 'Polygon', 'coordinates': [[[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]]}}


def write_geo(path, features):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'type': 'FeatureCollection', 'features': features}, ensure_ascii=False))


class PlaceReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.places = OfflinePlaces(self.folder)
        names = ['古塔区', '钟楼区', '天桥区', '老城区', '城区', '新城区', '市中区', '郊区', '矿区',
                 '新区', '开发区', '高新区', '西湖区', '天津市', '津市市', '拱墅区', '柯桥区', '钱塘区',
                 '松江区', '嘉定区', '宿州市', '秦皇岛市', '永年区', '南京市', '临夏回族自治州']
        self.assets = [{'place_names': json.dumps(names)}]

    def test_every_reported_false_positive_abstains(self):
        for query in ['绍兴古塔', '傍晚义乌钟楼', '宁波夜晚赛博朋克风过街天桥',
                      '夜晚和浩哥晨阳探访老城区', '西湖']:
            with self.subTest(query=query):
                self.assertEqual(self.places.clues(query, self.assets)[0], set())
        for name in ['城区', '老城区', '新城区', '市中区', '郊区', '矿区', '新区', '开发区',
                     '高新区', '古塔区', '钟楼区', '天桥区']:
            self.assertEqual(aliases(name), set())
            self.assertEqual(self.places.clues('在' + name + '拍的', self.assets)[0], set())

    def test_required_names_and_context(self):
        examples = [('柯桥拍的', '柯桥区'), ('在钱塘的照片', '钱塘区'), ('松江区拍的', '松江区'),
                    ('嘉定区', '嘉定区'), ('宿州市', '宿州市'), ('秦皇岛市拍的', '秦皇岛市'),
                    ('在永年拍的…', '永年区'), ('拱墅区河滨', '拱墅区'),
                    ('于松江拍照', '松江区'), ('去南京照片', '南京市'), ('到柯桥拍照', '柯桥区'),
                    ('南京的照片', '南京市'), ('钱塘区河边', '钱塘区'),
                    ('临夏回族自治州街道', '临夏回族自治州')]
        for query, name in examples:
            with self.subTest(query=query):
                self.assertEqual(self.places.clues(query, self.assets)[0], {name})
        self.assertEqual(self.places.clues('西湖的风景', self.assets)[0], set())
        self.assertEqual(self.places.clues('秦皇岛', self.assets)[0], {'秦皇岛市'})
        self.assertEqual(self.places.clues('柯桥', self.assets)[0], set())

    def test_longest_nonoverlap_and_residual(self):
        names, residual = self.places.clues('在天津市的照片', self.assets)
        self.assertEqual(names, {'天津市'})
        self.assertTrue(metadata_only(residual))
        names, residual = self.places.clues('拱墅区河滨，天津市拍的', self.assets)
        self.assertEqual(names, {'拱墅区', '天津市'})
        self.assertIn('河滨', residual)
        # Both occurrences are removed without changing unrelated text positions.
        self.assertEqual(self.places.clues('天津市与天津市', self.assets), ({'天津市'}, '   与   '))

    def test_foreign_optional_files_and_china_priority(self):
        world = self.folder / 'natural_earth'
        write_geo(world / 'natural_earth_countries.geojson', [
            feature('Korea', (126, 33, 130, 39), NAME_ZH='韩国', ADM0_A3='KOR'),
            feature('Thailand', (97, 5, 106, 21), NAME_ZH='泰国', ADM0_A3='THA'),
            feature('Vietnam', (102, 8, 110, 24), NAME_ZH='越南', ADM0_A3='VNM'),
            feature('China', (70, 10, 140, 55), NAME_ZH='中国', ADM0_A3='CHN')])
        write_geo(world / 'natural_earth_states.geojson', [
            feature('Seoul', (126.8, 37.4, 127.2, 37.8), name_zh='首尔', adm0_a3='KOR'),
            feature('Chiang Mai', (98, 18, 100, 20), name_zh='清迈府', adm0_a3='THA'),
            feature('China state', (120, 30, 123, 33), name_zh='错误外国省名', adm0_a3='CHN')])
        self.assertEqual(self.places.reverse(37.6, 127), sorted(['首尔', '韩国']))
        self.assertEqual(self.places.reverse(19, 99), ['泰国', '清迈府'])
        self.assertIn('越南', self.places.reverse(10, 108))
        self.assertEqual(self.places.reverse(31, 121), [])
        for query, wanted in [('在韩国拍的', '韩国'), ('首尔的夜景', '首尔'),
                              ('在清迈拍的', '清迈府'), ('越南长廊', '越南')]:
            self.assertIn(wanted, self.places.clues(query, [])[0])
        old_version = self.places.version
        # Even if coarse foreign geometry overlaps, detailed China wins.
        for level in ('province', 'city', 'district'):
            write_geo(self.folder / f'china_{level}.geojson', [feature('中国本地区域', (126, 37, 128, 38))])
        local = OfflinePlaces(self.folder)
        self.assertEqual(local.reverse(37.6, 127), ['中国本地区域'])
        self.assertNotEqual(old_version, local.version)
        (world / 'natural_earth_states.geojson').unlink()
        country_only = OfflinePlaces(self.folder)
        self.assertIn('泰国', country_only.reverse(19, 99))
        self.assertNotIn('清迈府', country_only.reverse(19, 99))

    def test_foreign_state_aliases_follow_boundaries(self):
        world = self.folder / 'natural_earth'
        write_geo(world / 'natural_earth_countries.geojson', [
            feature(name, (0, 0, 1, 1), NAME_ZH=name) for name in ('韩国', '泰国', '越南')])
        write_geo(world / 'natural_earth_states.geojson', [
            feature(name, (0, 0, 1, 1), name_zh=name) for name in ('海岸省', '清迈府', '普吉府', '首尔')])
        self.assertEqual(self.places.clues('普陀山海岸', [])[0], set())
        for word in ('海岸', '在海岸拍的', '中央', '北部'):
            self.assertNotIn('海岸省', self.places.clues(word, [])[0])
        self.assertIn('海岸省', self.places.clues('肯尼亚海岸省风景', [])[0])
        # Like Chinese provinces/cities, a state name at the query edge counts.
        for query in ('在清迈吃饭', '于清迈散步', '去清迈旅行', '到清迈旅行', '清迈拍的', '清迈的照片', '清迈照片', '清迈',
                      '清迈入境', '清迈动物园夜间部落表演', '普吉卡农乡的宁静海滩'):
            self.assertIn('清迈府' if '清迈' in query else '普吉府', self.places.clues(query, [])[0], query)
        for query in ('街头清迈建筑',):
            self.assertNotIn('清迈府', self.places.clues(query, [])[0], query)
        self.assertIn('普吉府', self.places.clues('普吉府海边', [])[0])
        self.assertIn('首尔', self.places.clues('首尔的夜景', [])[0])
        for name in ('韩国', '泰国', '越南'):
            self.assertIn(name, self.places.clues(name + '长廊', [])[0])

    def test_province_city_alias_boundaries_keep_counties_strict(self):
        for level, names in [('province', ['浙江省']), ('city', ['杭州市']),
                             ('district', ['西湖区', '钟楼区', '古塔区', '海宁市'])]:
            write_geo(self.folder / f'china_{level}.geojson', [feature(n, (0, 0, 1, 1)) for n in names])
        for query in ('杭州庆春路地铁站黄昏', '杭州香积寺建筑与城市巧妙融合', '跨年夜 杭州',
                      '杭州冬天吃的西式简餐', '街景，杭州，黄昏', '街景 杭州 黄昏', ' 杭州建筑'):
            self.assertEqual(self.places.clues(query, [])[0], {'杭州市'}, query)
        self.assertIn('浙江省', self.places.clues('浙江山里的照片', [])[0])
        for query in ('西湖', '钟楼', '古塔', '西湖风景', '钟楼建筑', '古塔夕阳', '海宁街景', '街景海宁'):
            self.assertFalse(self.places.clues(query, [])[0], query)
        self.assertFalse(self.places.clues('街头杭州建筑', [])[0])

    def test_world_bbox_grid_prunes_distant_polygons(self):
        world = self.folder / 'natural_earth'
        rows = [feature(str(i), (-150 + i * .1, -60, -149.95 + i * .1, -59), name_zh=f'远方{i}') for i in range(200)]
        rows.append(feature('Korea', (126, 33, 130, 39), name_zh='韩国'))
        write_geo(world / 'natural_earth_countries.geojson', rows)
        self.places.load()
        self.assertEqual(len(self.places._world_grid[(25, 7)]), 1)
        with patch('app.metadata._ring_contains', wraps=_ring_contains) as contains:
            self.assertEqual(self.places.reverse(37, 127), ['韩国'])
            self.assertEqual(contains.call_count, 1)
        self.assertEqual(self.places.reverse(float('nan'), 127), [])


class DateReviewTests(unittest.TestCase):
    def test_seasons_and_halves_boundaries(self):
        examples = [
            ('2019年春季拍的', ['2019-03-01', '2019-05-31'], ['2019-02-28', '2019-06-01']),
            ('2019年夏季拍的', ['2019-06-01', '2019-08-31'], ['2019-05-31', '2019-09-01']),
            ('2019年秋季拍的', ['2019-09-01', '2019-11-30'], ['2019-08-31', '2019-12-01']),
            ('2023年冬季的照片', ['2023-01-01', '2023-02-28', '2023-12-01'], ['2023-03-01', '2023-11-30', '2024-01-01']),
            ('2024年上半年拍的', ['2024-01-01', '2024-06-30'], ['2023-12-31', '2024-07-01']),
            ('2024年下半年拍的', ['2024-07-01', '2024-12-31'], ['2024-06-30', '2025-01-01'])]
        for query, positive, negative in examples:
            with self.subTest(query=query):
                rules, residual, warnings = date_clues(query)
                self.assertFalse(warnings)
                self.assertTrue(metadata_only(residual))
                for dt in positive:
                    self.assertTrue(matches_date(dt, rules), dt)
                for dt in negative:
                    self.assertFalse(matches_date(dt, rules), dt)

    def test_calendar_year_winter_includes_early_and_late_months(self):
        rules, residual, warnings = date_clues('2023年冬季的照片')
        self.assertFalse(warnings)
        self.assertTrue(metadata_only(residual))
        for dt in ('2023-01-01', '2023-02-28', '2023-12-31'):
            self.assertTrue(matches_date(dt, rules), dt)
        for dt in ('2022-12-31', '2023-03-01', '2023-11-30', '2024-01-01'):
            self.assertFalse(matches_date(dt, rules), dt)
        self.assertTrue(matches_date('2024-02-29', date_clues('2024年冬季')[0]))

    def test_no_year_and_chinese_months(self):
        for query, yes, no in [('春季', '2020-03-01', '2020-02-29'), ('夏季', '2024-08-31', '2024-09-01'),
                                ('秋季', '2024-11-30', '2024-12-01'), ('冬季', '2024-02-29', '2024-03-01'),
                                ('冬季', '2023-12-01', '2023-11-30'), ('上半年', '2024-06-30', '2024-07-01'),
                                ('下半年', '2024-07-01', '2024-06-30'), ('五月', '2024-05-01', '2024-06-01'),
                                ('七月份', '2024-07-31', '2024-08-01'), ('十二月', '2024-12-31', '2025-01-01'),
                                ('2025年十一月15日', '2025-11-15', '2025-11-16')]:
            with self.subTest(query=query):
                rules, residual, warnings = date_clues(query)
                self.assertTrue(matches_date(yes, rules))
                self.assertFalse(matches_date(no, rules))
                self.assertFalse(warnings)
                self.assertTrue(metadata_only(residual))


class EngineReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.engine = Engine(self.root / 'data')

    def test_ensure_stats_backfill_preserves_geography(self):
        photos = self.root / 'photos'
        photos.mkdir()
        Image.new('RGB', (24, 24), 'red').save(photos / 'a.jpg')
        lib = self.engine.add_library(photos)
        self.engine.index_library(lib['id'])
        with self.engine.db() as con:
            con.execute("UPDATE assets SET visual_stats=NULL,latitude=31,longitude=121,place_names='[\"上海市\"]' WHERE library_id=?", (lib['id'],))
            assets = [dict(r) for r in con.execute('SELECT * FROM assets WHERE library_id=?', (lib['id'],))]
        self.engine.ensure_stats(assets)
        self.assertTrue(assets[0]['visual_stats'])
        with self.engine.db() as con:
            stored = con.execute('SELECT * FROM assets WHERE id=?', (assets[0]['id'],)).fetchone()
        self.assertEqual(stored['visual_stats'], assets[0]['visual_stats'])
        self.assertEqual((stored['latitude'], stored['longitude'], stored['place_names']), (31, 121, '["上海市"]'))
        self.engine.ensure_stats(assets)
        self.engine.ensure_stats([{'id': 'missing', 'thumbnail': 'missing.jpg'}])

    def test_mixed_city_place_is_a_soft_vote_without_filtering(self):
        flow = FlowService(self.engine)
        folder = self.root / 'boundaries'
        for level in ('province', 'city', 'district'):
            write_geo(folder / f'china_{level}.geojson', [feature('杭州市' if level == 'city' else '浙江省' if level == 'province' else '西湖区', (0, 0, 1, 1))])
        flow.recall.places = OfflinePlaces(folder)
        assets = [{'id': aid, 'library_id': 'lib', 'fingerprint': 'f', 'place_names': json.dumps(names)}
                  for aid, names in [('in', ['杭州市']), ('out', ['南京市'])]]
        rows = [{'asset_id': aid, 'vector': np.array(v, dtype=np.float32).tobytes()}
                for aid, v in [('in', [0, 1]), ('out', [1, 0])]]
        provider = Mock(installed=True, version='v')
        config = flow.store.config()
        with patch.object(flow.recall, 'primary', return_value=provider), \
             patch.object(flow.recall, 'rows', return_value=rows), \
             patch.object(flow.recall, 'vector', return_value=np.array([1, 0], dtype=np.float32)), \
             patch.object(self.engine.hybrid, 'rankings', return_value={}):
            for text in ('杭州庆春路地铁站黄昏', '杭州香积寺建筑与城市巧妙融合', '跨年夜 杭州', '杭州冬天吃的西式简餐'):
                result = flow.recall.search(assets, {'conditions': fallback_conditions(text)}, config, 100)
                self.assertEqual(set(result[0]), {'in', 'out'}, text)
                channels = {c['family']: c['ids'] for c in result[-1]}
                self.assertEqual(channels['metadata_place'], ['in'])
                self.assertEqual(channels['primary'], ['out', 'in'])

    def test_pure_metadata_capture_order_and_new_period_routes(self):
        flow = FlowService(self.engine)
        flow.recall.places = OfflinePlaces(self.root / 'no-boundaries')
        assets = [{'id': aid, 'captured': captured, 'place_names': '["南京市"]'} for aid, captured in [
            ('missing', None), ('z-new', '2024-05-01T12:00:00'), ('a-old', '2019-07-01T12:00:00'),
            ('b-new', '2024-05-01T12:00:00'), ('invalid-id-order', '2023-12-15T12:00:00')]]
        with patch.object(flow.recall, 'primary', side_effect=AssertionError('纯元数据不得运行模型')):
            def search(text):
                return flow.recall.search(assets, {'conditions': fallback_conditions(text)}, flow.store.config(), 100)[0]
            self.assertEqual(search('南京市拍的'), ['b-new', 'z-new', 'invalid-id-order', 'a-old', 'missing'])
            self.assertEqual(search('2024年上半年拍的'), ['b-new', 'z-new'])
            self.assertEqual(search('2019年夏季拍的'), ['a-old'])
            self.assertEqual(search('五月拍的'), ['b-new', 'z-new'])
            self.assertEqual(search('2023年冬季拍的'), ['invalid-id-order'])
            config = flow.configure({'fusion_weights': {'metadata_place': 0}})
            result = flow.recall.search(assets, {'conditions': fallback_conditions('南京市拍的')}, config, 100)
            self.assertEqual(result[0], ['b-new', 'z-new', 'invalid-id-order', 'a-old', 'missing'])


class PreparationReviewTests(unittest.TestCase):
    def test_world_download_is_optional_pinned_and_preserves_china(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'geography'
            folder.mkdir()
            china = folder / 'china_district.geojson'
            china.write_text('preserved')
            raw = json.dumps({'features': [feature('Korea', (126, 33, 130, 39), NAME_ZH='韩国')]}).encode()
            def open_url(url, timeout):
                self.assertIn('/v5.1.2/geojson/ne_10m_admin_', url)
                return io.BytesIO(raw)
            with patch('sys.argv', ['prepare_geography.py', '--natural-earth', '--output', str(folder)]), \
                 patch.object(prepare.urllib.request, 'urlopen', side_effect=open_url) as network:
                prepare.main()
                self.assertEqual(network.call_count, 2)
                with self.assertRaises(SystemExit):
                    prepare.main()
                self.assertEqual(network.call_count, 2)
            self.assertEqual(china.read_text(), 'preserved')
            manifest = json.loads((folder / 'natural_earth/manifest.json').read_text())
            self.assertEqual(manifest['data_license'], 'Public domain')
            self.assertEqual(manifest['tag'], 'v5.1.2')
            self.assertEqual(len(manifest['files']), 2)

    def test_failed_preparation_leaves_no_partial_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'natural_earth'
            with patch.object(prepare.urllib.request, 'urlopen', return_value=io.BytesIO(b'{"features": []}')):
                with self.assertRaises(ValueError):
                    prepare.download_set(target, {'bad.geojson': 'https://example.invalid'}, {'data_license': 'Public domain'})
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(tmp).iterdir()), [])
