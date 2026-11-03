"""只读通道顺序，离线重放 family RRF；不加载模型。"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.flow.domain import fuse
from scripts.eval_retrieval import KS, summarize


def family(name):
    return name.split('：')[0] if '：' in name else {
        '照片文字': 'ocr', '画面描述关键词': 'description', '画面描述语义': 'description',
        '用户命名人物': 'people', '拍摄日期线索': 'metadata_time',
        '拍摄地点': 'metadata_place', '拍摄日期': 'metadata_time',
    }.get(name, name)


def replay(record, weight=1., primary_only=False, tail=None):
    route = lambda name: record['final'].get('channel_families', {}).get(name, family(name))
    channels = [{'name': name, 'family': route(name), 'ids': ids,
                 'weight': weight if route(name) == 'siglip' else 1.}
                for name, ids in record['final']['channel_orders'].items()
                if not primary_only or route(name) == 'primary']
    order = fuse(channels, 1000000)[0]
    if tail is not None:
        # Preserve primary top K exactly, append fused remainder (not a new RRF formula).
        primary = fuse([c for c in channels if c['family'] == 'primary'], tail)[0]
        order = primary + [a for a in order if a not in set(primary)]
    return order


def compare(records):
    schemes = [('等权 RRF', {}), *[(f'SigLIP {w}', {'weight': w}) for w in (.1, .3, .5)],
               *[(f'主路前{k}保留', {'tail': k}) for k in (10, 50)], ('纯 WeMM', {'primary_only': True})]
    out = []
    for name, options in schemes:
        metrics = []
        for r in records:
            if not r.get('targets') or not r.get('final', {}).get('channel_orders'):
                continue
            order = replay(r, **options)
            positions = {a: i + 1 for i, a in enumerate(order)}
            metrics.append(summarize([positions.get(t) for t in r['targets']], len(r['targets'])))
        out.append({'方案': name, 'n': len(metrics), **{k: sum(m[k] for m in metrics) / len(metrics)
                    if metrics else None for k in [*[f'hit@{k}' for k in KS], *[f'recall@{k}' for k in KS], 'mrr']}})
    return out


def table(rows):
    keys = list(rows[0])
    return '\n'.join(['| ' + ' | '.join(keys) + ' |', '| ' + ' | '.join(['---'] * len(keys)) + ' |',
        *['| ' + ' | '.join(f'{r[k]:.4f}' if isinstance(r[k], float) else str(r[k]) for k in keys) + ' |' for r in rows]])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('queries', nargs='+', type=Path)
    p.add_argument('--output', type=Path, help='可选新建 JSON 结果（拒绝覆盖）')
    args = p.parse_args()
    result = {}
    for path in args.queries:
        rows = compare([json.loads(l) for l in path.read_text().splitlines() if l.strip()])
        result[str(path)] = rows
        print(str(path) + '\n' + table(rows))
    if args.output:
        with args.output.open('x') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
