"""EXIF 和离线行政区/日期规则。运行时不访问网络，不下载数据。"""
import hashlib
import json
import math
import os
import re
import unicodedata
from datetime import date, datetime
from pathlib import Path

# 节日当日到随后两天；不是国务院逐年调休假期表。
LUNAR = {
    2015: ('0219', '0927'), 2016: ('0208', '0915'), 2017: ('0128', '1004'),
    2018: ('0216', '0924'), 2019: ('0205', '0913'), 2020: ('0125', '1001'),
    2021: ('0212', '0921'), 2022: ('0201', '0910'), 2023: ('0122', '0929'),
    2024: ('0210', '0917'), 2025: ('0129', '1006'), 2026: ('0217', '0925'),
    2027: ('0206', '0915'), 2028: ('0126', '1003'), 2029: ('0213', '0922'),
    2030: ('0203', '0912'),
}


def gps_coordinates(exif):
    try:
        gps = exif.get(34853)
        if not isinstance(gps, dict):
            gps = exif.get_ifd(34853)
        def degrees(values, ref, positive, negative, bound):
            if isinstance(ref, bytes):
                ref = ref.decode('ascii')
            if ref not in (positive, negative) or len(values) != 3:
                raise ValueError('GPS 格式')
            d, m, s = map(float, values)
            if not all(math.isfinite(v) for v in (d, m, s)) or not (0 <= d <= bound and 0 <= m < 60 and 0 <= s < 60):
                raise ValueError('GPS 范围')
            value = d + m / 60 + s / 3600
            if value > bound:
                raise ValueError('GPS 范围')
            return value if ref == positive else -value
        return degrees(gps[2], gps[1], 'N', 'S', 90), degrees(gps[4], gps[3], 'E', 'W', 180)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, AttributeError, OverflowError, OSError, SyntaxError):
        return None, None


def captured_time(exif):
    try:
        nested = exif.get(34665)
        if not isinstance(nested, dict):
            nested = exif.get_ifd(34665)
    except (AttributeError, ValueError, KeyError, TypeError):
        nested = {}
    value = exif.get(36867) or nested.get(36867) or exif.get(306)
    try:
        return datetime.strptime(str(value), '%Y:%m:%d %H:%M:%S').isoformat()
    except ValueError:
        return None


def date_clues(text):
    """返回 (year|None, month|None, day|None, span_days) 的并集、残余文字和警告。"""
    rules, warnings = [], []
    holiday = r'(?:(?P<year>(?:19|20)\d{2})年?)?(?P<holiday>劳动节|五一|国庆节?|春节|过年|中秋节?|元旦)'
    def holiday_rule(m):
        year = int(m['year']) if m['year'] else None
        name = m['holiday']
        fixed = {'劳动节': (5, 1, 3), '五一': (5, 1, 3), '国庆': (10, 1, 7),
                 '国庆节': (10, 1, 7), '元旦': (1, 1, 1)}
        if name in fixed:
            month, day, span = fixed[name]
            rules.append((year, month, day, span))
        else:
            years = [year] if year else sorted(LUNAR)
            if year and year not in LUNAR:
                warnings.append('农历节日仅支持 2015～2030 年；该年份不作为日期命中')
            for y in years:
                if y in LUNAR:
                    value = LUNAR[y][0 if name in ('春节', '过年') else 1]
                    rules.append((y, int(value[:2]), int(value[2:]), 7 if name in ('春节', '过年') else 3))
        return ' '
    residual = re.sub(holiday, holiday_rule, text)
    periods = {'春季': (3, 6), '夏季': (6, 9), '秋季': (9, 12),
               '冬季': (12, 3), '上半年': (1, 7), '下半年': (7, 1)}
    def period_rule(m):
        year = int(m['year']) if m['year'] else None
        start, end = periods[m['period']]
        if year and m['period'] == '冬季':
            # Calendar-year wording is ambiguous: include both winter fragments.
            rules.extend((year, month, None, 1) for month in (1, 2, 12))
        elif year:
            first = date(year, start, 1)
            last = date(year + (end <= start), end, 1)
            rules.append((year, start, 1, (last - first).days))
        else:
            months = range(start, end) if end > start else (*range(start, 13), *range(1, end))
            rules.extend((None, month, None, 1) for month in months)
        return ' '
    residual = re.sub(r'(?:(?P<year>(?:19|20)\d{2})年\s*)?(?P<period>春季|夏季|秋季|冬季|上半年|下半年)', period_rule, residual)
    chinese_months = dict(zip(('一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '十一', '十二'), range(1, 13)))
    residual = re.sub(r'(?<![一二三四五六七八九十])(?P<month>十二|十一|十|[一二三四五六七八九])月',
                      lambda m: str(chinese_months[m['month']]) + '月', residual)
    pattern = r'(?<!\d)(?:(?P<year>(?:19|20)\d{2})年\s*)?(?P<month>\d{1,2})月份?(?:\s*(?P<day>\d{1,2})[日号])?'
    def month_rule(m):
        y, mo, d = int(m['year']) if m['year'] else None, int(m['month']), int(m['day']) if m['day'] else None
        try:
            date(y or 2000, mo, d or 1)
            rules.append((y, mo, d, 1))
        except ValueError:
            # Invalid explicit dates must not degrade to a year/month match.
            rules.append((0, 0, 0, 1))
            warnings.append('日期格式无效')
        return ' '
    residual = re.sub(pattern, month_rule, residual)
    def year_rule(m):
        rules.append((int(m[0].rstrip('年')), None, None, 1))
        return ' '
    residual = re.sub(r'(?<!\d)(?:19|20)\d{2}年?(?!\d)', year_rule, residual)
    return rules, residual, warnings


def matches_date(captured, rules):
    try:
        dt = date.fromisoformat(captured[:10])
    except (ValueError, TypeError):
        return False
    for y, m, d, span in rules:
        if m and d:
            # Check previous year too for a range crossing New Year.
            for year in ([y] if y else [dt.year - 1, dt.year]):
                try:
                    if 0 <= (dt - date(year, m, d)).days < span:
                        return True
                except ValueError:
                    pass
        elif (y is None or dt.year == y) and (m is None or dt.month == m):
            return True
    return False


# These names are also ordinary scene descriptions. Administrative data alone
# cannot distinguish the landmark/scene, so even exact official names abstain.
PLACE_STOPWORDS = {'城区', '老城区', '新城区', '市中区', '郊区', '矿区', '新区',
                   '开发区', '高新区', '古塔区', '钟楼区', '天桥区', '工业园区', '经济开发区',
                   '海岸', '沿海', '中央', '中部', '北部', '南部', '东部', '西部'}
PLACE_SUFFIX = r'(?:特别行政区|自治区|自治州|自治县|特别市|省|市|区|县|州|府)$'
FOREIGN_ALIASES = {'大韩民国': {'韩国'}, '首尔特别市': {'首尔'}, '清迈府': {'清迈'}}


def aliases(name):
    if name in PLACE_STOPWORDS:
        return set()
    short = re.sub(PLACE_SUFFIX, '', name)
    return (({name, short} if len(short) >= 2 else {name}) | FOREIGN_ALIASES.get(name, set())) - PLACE_STOPWORDS


def _ring_contains(x, y, ring):
    inside = False
    for first, second in zip(ring, ring[1:] + ring[:1]):
        x1, y1 = first[:2]
        x2, y2 = second[:2]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


class OfflinePlaces:
    cell_size = 5

    def __init__(self, folder=None):
        self.folder = Path(folder or os.environ.get('PHOTORECALL_GEO_DIR') or Path(__file__).resolve().parents[1] / 'resources' / 'geography')
        self.regions, self.version = [], 'missing'
        self.loaded = False
        self.foreign_names, self.country_names, self.province_city_names = set(), set(), set()
        self._china_grid, self._world_grid = {}, {}

    def _add(self, name, geometry, grid):
        polygons = [geometry['coordinates']] if geometry['type'] == 'Polygon' else geometry['coordinates'] if geometry['type'] == 'MultiPolygon' else []
        for rings in polygons:
            if not rings or not rings[0]:
                continue
            xs, ys = zip(*(point[:2] for point in rings[0]))
            bbox = min(xs), min(ys), max(xs), max(ys)
            index = len(self.regions)
            self.regions.append((name, bbox, rings))
            x1, y1, x2, y2 = [math.floor(v / self.cell_size) for v in bbox]
            for x in range(x1, x2 + 1):
                for y in range(y1, y2 + 1):
                    grid.setdefault((x, y), []).append(index)

    def load(self):
        if self.loaded:
            return
        digest = hashlib.sha256()
        china = [self.folder / f'china_{level}.geojson' for level in ('province', 'city', 'district')]
        files = [(p, False) for p in china] if all(p.is_file() for p in china) else []
        files += [(self.folder / 'natural_earth' / f'natural_earth_{level}.geojson', True) for level in ('countries', 'states')
                  if (self.folder / 'natural_earth' / f'natural_earth_{level}.geojson').is_file()]
        for path, foreign in files:
            raw = path.read_bytes()
            digest.update(path.name.encode())
            digest.update(raw)
            for f in json.loads(raw)['features']:
                props = f['properties']
                if foreign:
                    # Keep China's detailed local boundaries as the only authority.
                    codes = {str(props.get(k, '')).upper() for k in ('ADM0_A3', 'adm0_a3', 'ISO_A3', 'iso_a3', 'SOV_A3', 'sov_a3')}
                    if codes & {'CHN', 'CH1'}:
                        continue
                    name = props.get('NAME_ZH') or props.get('name_zh')
                    if not name:
                        continue
                    self.foreign_names.add(name)
                    if path.name == 'natural_earth_countries.geojson':
                        self.country_names.add(name)
                else:
                    name = props['name']
                    if path.name in ('china_province.geojson', 'china_city.geojson'):
                        self.province_city_names.add(name)
                if f.get('geometry'):
                    self._add(name, f['geometry'], self._world_grid if foreign else self._china_grid)
        if files:
            self.version = 'polygons-v2:' + digest.hexdigest()[:16]
        self.loaded = True

    def _lookup(self, lat, lon, grid):
        key = math.floor(lon / self.cell_size), math.floor(lat / self.cell_size)
        names = set()
        for index in grid.get(key, ()):
            name, (x1, y1, x2, y2), rings = self.regions[index]
            if x1 < lon < x2 and y1 < lat < y2 and _ring_contains(lon, lat, rings[0]) and not any(_ring_contains(lon, lat, r) for r in rings[1:]):
                names.add(name)
        return names

    def reverse(self, lat, lon):
        self.load()
        if lat is None or lon is None or not (math.isfinite(lat) and math.isfinite(lon)) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return []
        local = self._lookup(lat, lon, self._china_grid)
        return sorted(local or self._lookup(lat, lon, self._world_grid))

    def clues(self, text, assets):
        self.load()
        names = {r[0] for r in self.regions}
        for a in assets:
            names.update(json.loads(a.get('place_names') or '[]'))
        def boundary(char):
            return char.isspace() or unicodedata.category(char).startswith('P')
        candidates = {}
        for name in sorted(names):
            for alias in aliases(name):
                official = alias == name and bool(re.search(PLACE_SUFFIX, name))
                foreign = name in self.foreign_names
                for match in re.finditer(re.escape(alias), text):
                    start, end = match.span()
                    context = (start > 0 and text[start - 1] in '在于去到'
                               or re.match(r'拍|的照片|照片' if foreign else r'拍|的照片|照片|区', text[end:])
                               or text.strip() == alias and (foreign or len(alias) >= 3))
                    # Foreign first-level regions (清迈府, 普吉府) behave like Chinese
                    # provinces/cities; generic words such as 海岸 are blocked in aliases().
                    broad_context = ((name in self.province_city_names or foreign)
                        and (not text[:start].strip() or not text[end:].strip()
                             or boundary(text[start - 1]) and boundary(text[end])))
                    # Country names stay unrestricted. A suffix-free state name
                    # supplied by the data is a full name, not a stripped alias.
                    foreign_full = foreign and (name in self.country_names or alias == name)
                    if official or foreign_full or context or broad_context:
                        candidates.setdefault((start, end), set()).add(name)
        occupied, found, spans = set(), set(), []
        # Longest first; merge homonyms sharing one span, never count nested names.
        for (start, end), matched in sorted(candidates.items(), key=lambda item: (-(item[0][1] - item[0][0]), item[0][0])):
            if occupied.intersection(range(start, end)):
                continue
            occupied.update(range(start, end))
            found.update(matched)
            spans.append((start, end))
        chars = list(text)
        for start, end in spans:
            chars[start:end] = ' ' * (end - start)
        return found, ''.join(chars)


def metadata_only(residual):
    return not re.sub(r'照片|图片|拍摄|拍的|拍|摄于|摄|的|在|于|去|到|过|找|所有|全部|期间|时候|时候|和|与|[\W_]', '', residual)
