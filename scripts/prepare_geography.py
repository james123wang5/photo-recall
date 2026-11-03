"""显式准备中国或 Natural Earth 离线边界；不发送照片或坐标。"""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path
import urllib.request

NATURAL_EARTH_TAG = 'v5.1.2'


def download_set(output, urls, manifest):
    if output.exists():
        raise SystemExit(f'拒绝覆盖 {output}；请使用新的输出目录')
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {**manifest, 'crs': {}, 'files': {}}
    with tempfile.TemporaryDirectory(prefix='.geography-', dir=output.parent) as tmp:
        staging = Path(tmp) / 'ready'
        staging.mkdir()
        for name, url in urls.items():
            with urllib.request.urlopen(url, timeout=120) as response:
                raw = response.read()
            if name.endswith('.geojson'):
                geo = json.loads(raw)
                manifest['crs'][name] = geo.get('crs', 'GeoJSON 默认 WGS84；未额外声明 CRS')
                rows = geo['features']
                if not rows:
                    raise ValueError('边界文件为空')
                if manifest.get('data_license') == 'Public domain':
                    named = sum(bool(f['properties'].get('NAME_ZH') or f['properties'].get('name_zh')) for f in rows)
                    if not named:
                        raise ValueError('Natural Earth 文件缺少中文名称 NAME_ZH/name_zh')
                    manifest.setdefault('chinese_names', {})[name] = {'named': named, 'total': len(rows)}
                elif not all('name' in f['properties'] for f in rows):
                    raise ValueError('边界文件缺少中文名称')
            (staging / name).write_bytes(raw)
            manifest['files'][name] = {'url': url, 'sha256': hashlib.sha256(raw).hexdigest()}
        (staging / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        staging.rename(output)
    print('准备完成：', output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'resources/geography')
    p.add_argument('--revision', help='中国边界 commit SHA；默认显式查询 PyGeoCN main 当前 SHA 并记录')
    p.add_argument('--natural-earth', action='store_true', help='仅增量准备境外边界到 output/natural_earth，不改中国数据')
    p.add_argument('--scale', choices=['10m', '50m'], default='10m', help='Natural Earth 比例尺；默认10m的一级行政区覆盖较全')
    args = p.parse_args()
    if args.natural_earth:
        base = f'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/{NATURAL_EARTH_TAG}/geojson/'
        urls = {f'natural_earth_{level}.geojson': base + f'ne_{args.scale}_{source}.geojson'
                for level, source in [('countries', 'admin_0_countries'), ('states', 'admin_1_states_provinces')]}
        download_set(args.output / 'natural_earth', urls, {'source': base, 'tag': NATURAL_EARTH_TAG,
            'scale': args.scale, 'data_license': 'Public domain',
            'license_url': 'https://www.naturalearthdata.com/about/terms-of-use/'})
        return
    if not args.revision:
        with urllib.request.urlopen('https://api.github.com/repos/CZD-MO/PyGeoCN/commits/main', timeout=30) as response:
            args.revision = json.load(response)['sha']
    if len(args.revision) != 40 or any(c not in '0123456789abcdef' for c in args.revision):
        p.error('--revision 必须为 40 位小写 commit SHA')
    base = f'https://raw.githubusercontent.com/CZD-MO/PyGeoCN/{args.revision}/'
    urls = {Path(name).name: base + name for name in ['LICENSE', *[f'PyGeoCN/china_{x}.geojson' for x in ('province', 'city', 'district')]]}
    download_set(args.output, urls, {'source': base, 'code_license': 'MIT', 'data_source': '天地图行政区划（经 PyGeoCN 分发）'})


if __name__ == '__main__':
    main()
