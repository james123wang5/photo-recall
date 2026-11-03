import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import eval_retrieval as ev


class EvaluationSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.args = Namespace(config=None, index=[], offset=0, limit=0, stage='recall',
                              depth=200, mode='find_one', timeout=2)
        self.flow = Mock()
        self.flow.catalog.return_value = [{'name': 'a.jpg', 'id': 'a'}]
        self.flow.store.config.return_value = {}
        self.suite = {'name': 'test', 'images': self.tmp.name, 'queries': [
            {'id': 'q', 'turns': ['test'], 'targets': ['a.jpg'], 'tags': {}}]}
        for name, value in [('RESULTS', Path(self.tmp.name)), ('engine_for', Mock(return_value=(Mock(), self.flow, Mock()))),
                            ('build_index', Mock(return_value={'id': 'lib'}))]:
            patcher = patch.object(ev, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_missing_target_is_rejected(self):
        self.suite['queries'][0]['targets'].append('missing.jpg')
        with self.assertRaisesRegex(RuntimeError, '缺少答案图片'):
            ev.evaluate(self.suite, self.args)
        self.assertFalse(list(Path(self.tmp.name).glob('test-*')))

    def test_failures_are_reported_and_do_not_look_successful(self):
        with patch.object(ev, 'run_recall', side_effect=ValueError('failure')):
            with self.assertRaisesRegex(RuntimeError, '1 条查询失败'):
                ev.evaluate(self.suite, self.args)
        result = next(Path(self.tmp.name).glob('test-*/summary.json'))
        summary = json.loads(result.read_text())
        self.assertEqual((summary['requested_queries'], summary['successful_queries'], summary['failed_queries']), (1, 0, 1))
        self.assertIsNone(summary['hit@1'])

    def test_channel_metrics_only_count_queries_where_channel_ran(self):
        self.suite['queries'].append({'id': 'q2', 'turns': ['other'], 'targets': ['a.jpg'], 'tags': {}})
        turns = [(["a"], {"primary": ["a"], "conditional": ["a"]}, [])]
        other = [(["a"], {"primary": ["a"]}, [])]
        with patch.object(ev, 'run_recall', side_effect=[turns, other]):
            ev.evaluate(self.suite, self.args)
        summary = json.loads(next(Path(self.tmp.name).glob('test-*/summary.json')).read_text())
        self.assertEqual(summary['channel_query_count']['conditional'], 1)
        self.assertEqual(summary['channel_hit@50']['conditional'], 1.0)
        result_path = next(Path(self.tmp.name).glob('test-*/queries.jsonl'))
        records = [json.loads(line) for line in result_path.read_text().splitlines()]
        self.assertEqual(records[0]['final']['channel_orders']['conditional'], ['a.jpg'])

    def test_multi_target_recall_uses_all_targets(self):
        metrics = ev.summarize([1, None], 2)
        self.assertEqual(metrics['hit@1'], 1)
        self.assertEqual(metrics['recall@1'], 0.5)


if __name__ == '__main__':
    unittest.main()
