"""Run PhotoRecall's flow retrieval on public, annotated benchmarks.

Each dataset is converted into one or more "suites": a folder of images plus
queries with ground-truth filenames. Every suite gets its own isolated data
directory (data/eval_runs/<suite>) whose `models` links to data/models, so the
user's real library database is never touched.

    # list suites that are ready on disk
    .venv/bin/python scripts/eval_retrieval.py --list
    # recall-stage evaluation with per-channel ranks (no vision verification)
    .venv/bin/python scripts/eval_retrieval.py photobench_album1 --index visual
    # full controller (planner, rerank, verification) on the first 20 queries
    .venv/bin/python scripts/eval_retrieval.py photobench_album1 --stage full --limit 20

Results: data/eval_results/<suite>-<stage>-<time>/{queries.jsonl,summary.json}.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import ast
import csv
import io
import json
import math
import threading
import time
import traceback
import zipfile

DATA = ROOT / "data"
SETS = DATA / "eval_datasets"
RUNS = DATA / "eval_runs"
RESULTS = DATA / "eval_results"
KS = (1, 5, 10, 50)
# Match Engine.index_library: supported files only, including nested paths.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------- adapters
# A suite is {"images": folder, "queries": [{"id", "turns": [text, ...],
# "targets": [filename relative to images], "tags": {...}}]}. A single-turn
# query has one turn; an empty target list is a "target does not exist" query.

def photobench(album, lang):
    def load():
        images = SETS / "photobench/raw_albums" / f"album{album}"
        if not images.is_dir():
            # gdown keeps the Drive folder name; accept one extra nesting level.
            images = next(SETS.glob(f"photobench/raw_albums/**/album{album}"), images)
        rows = json.loads((SETS / f"photobench/validation/album{album}_validation.json").read_text())
        # Category fields (Location/Time/Person/Object/...) allow per-type breakdowns.
        tags = ("Location", "Time", "Person", "Object", "Concept", "Genre", "Source")
        queries = [{"id": f"a{album}-{i}", "turns": [row["query_cn" if lang == "cn" else "query_en"]],
                    "targets": row.get("ground_truth") or [],
                    "tags": {"album": album, **{k: row.get(k) for k in tags}}} for i, row in enumerate(rows)]
        return {"images": images, "queries": queries}
    return load


def disbench(user=None):
    def load():
        rows = [json.loads(line) for line in (SETS / "disbench/queries.jsonl").read_text().splitlines() if line.strip()]
        users = sorted({r["user_id"] for r in rows}) if user is None else [user]
        return [{"name": "disbench_" + u.replace("@", "_"), "images": SETS / "disbench/images" / u,
                 "queries": [{"id": r["query_id"], "turns": [r["query"]], "targets": [a + ".jpg" for a in r["answer"]],
                              "tags": {"event_type": r["event_type"]}} for r in rows if r["user_id"] == u]}
                for u in users]
    return load


def flickr1k():
    images = SETS / "flickr1k"
    with open(images / "test_1k_flickr.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    folder = next((p.parent for p in images.rglob("*.jpg")), images)
    queries = []
    for row in rows:
        captions = json.loads(row["raw"]) if row.get("raw", "").startswith("[") else [row.get("caption", "")]
        name = Path(row.get("filename") or row.get("image")).name
        for j, text in enumerate(captions):
            queries.append({"id": f"{name}-{j}", "turns": [text], "targets": [name], "tags": {}})
    return {"images": folder, "queries": queries}


def urban1k():
    base = SETS / "urban1k"
    image_dir = next(base.rglob("image"), None)
    caption_dir = next(base.rglob("caption"), None)
    queries = []
    for txt in sorted(caption_dir.glob("*.txt")):
        image = next((p for p in image_dir.glob(txt.stem + ".*")), None)
        if image:
            queries.append({"id": txt.stem, "turns": [txt.read_text().strip()], "targets": [image.name], "tags": {}})
    return {"images": image_dir, "queries": queries}


def countbench():
    import pyarrow.parquet as pq
    base = SETS / "countbench"
    images = base / "images"
    images.mkdir(exist_ok=True)
    table = pq.read_table(base / "countbench.parquet").to_pylist()
    queries = []
    for i, row in enumerate(table):
        name = f"{i:04d}.jpg"
        if not (images / name).exists() and row.get("image") and row["image"].get("bytes"):
            from PIL import Image
            Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB").save(images / name, quality=95)
        if (images / name).exists():
            queries.append({"id": name, "turns": [row["text"]], "targets": [name], "tags": {"number": row["number"]}})
    return {"images": images, "queries": queries}


def negbench():
    # COCO val2017 retrieval with negated captions (5k images, 5 captions each).
    path = SETS / "negbench/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    queries = []
    for row in rows:
        name = Path(row["filepath"]).name
        # Columns hold Python list literals, e.g. "['a', 'b']".
        tags = {"negative_objects": ast.literal_eval(row["negative_objects"])}
        for j, text in enumerate(ast.literal_eval(row["captions"])):
            queries.append({"id": f"{name}-{j}", "turns": [text], "targets": [name], "tags": tags})
    return {"images": SETS / "coco/val2017", "queries": queries}


def chatir(dialogues="VisDial_v1.0_queries_val.json"):
    # Reduced protocol: search the 2,064 VisDial val images instead of the
    # paper's 50k COCO corpus (that needs coco/unlabeled2017, see README).
    def load():
        rows = json.loads((SETS / "chatir/dialogues" / dialogues).read_text())
        files = list((SETS / "visdial").rglob("*.jpg"))
        images = files[0].parent if files else SETS / "visdial"
        # Dialogues name targets "unlabeled2017/000000185565.jpg"; the VisDial
        # val download uses the same 12-digit ID as VisualDialog_val2018_000000185565.jpg.
        by_id = {f.stem[-12:]: f.name for f in files}
        queries = [{"id": str(i), "turns": row["dialog"],
                    "targets": [by_id.get(Path(row["img"]).stem[-12:], Path(row["img"]).name)],
                    "tags": {"dialogues": dialogues}} for i, row in enumerate(rows)]
        return {"images": images, "queries": queries}
    return load


def docci():
    base = SETS / "docci"
    rows = [json.loads(l) for l in (base / "docci_descriptions.jsonlines").read_text().splitlines() if l.strip()]
    images = next((p.parent for p in base.rglob("test_*.jpg")), base / "images")
    test = images.parent / "images_test"
    # Only the 5k test split forms the gallery; link it into its own folder.
    test.mkdir(exist_ok=True)
    queries = []
    for row in rows:
        if row.get("split") != "test":
            continue
        source = images / row["image_file"]
        link = test / row["image_file"]
        if source.exists() and not link.exists():
            link.symlink_to(source)
        queries.append({"id": row["example_id"], "turns": [row["description"]], "targets": [row["image_file"]], "tags": {}})
    return {"images": test, "queries": queries}


def circo():
    # Reference photo + modification text, several correct targets per query.
    # Gallery is COCO unlabeled2017 (~123k images): download group coco_unlabeled.
    rows = json.loads((SETS / "circo/annotations/val.json").read_text())
    name = lambda i: f"{int(i):012d}.jpg"
    queries = [{"id": str(r["id"]), "turns": [r["relative_caption"]], "reference": name(r["reference_img_id"]),
                "targets": [name(i) for i in r["gt_img_ids"]],
                "tags": {"shared_concept": r["shared_concept"], "aspects": r["semantic_aspects"]}} for r in rows]
    return {"images": SETS / "coco/unlabeled2017", "queries": queries}


SUITES = {
    **{f"photobench_album{a}": photobench(a, "cn") for a in (1, 2, 3)},
    **{f"photobench_album{a}_en": photobench(a, "en") for a in (1, 2, 3)},
    "flickr1k": flickr1k,
    "urban1k": urban1k,
    "countbench": countbench,
    "negbench_coco": negbench,
    "chatir_visdial": chatir(),
    "chatir_visdial_chatgpt": chatir("ChatGPT_BLIP2.json"),
    "docci_test": docci,
    "circo_val": circo,
}


def suites(name):
    if name == "disbench":
        return disbench()()
    if name.startswith("disbench_"):
        user = name[len("disbench_"):].replace("_", "@", 1)
        return disbench(user)()
    suite = SUITES[name]()
    return [{"name": name, **suite}]


def ready():
    out = []
    for name in list(SUITES) + ["disbench"]:
        try:
            for suite in suites(name):
                count = sum(1 for p in Path(suite["images"]).rglob("*") if p.is_file() and not p.is_symlink() and p.suffix.lower() in IMAGE_EXTENSIONS) if Path(suite["images"]).is_dir() else 0
                out.append((suite["name"], count, len(suite["queries"])))
        except Exception as exc:
            out.append((name, 0, "未就绪：" + str(exc)[:80]))
    return out


# ---------------------------------------------------------------- engine

def engine_for(suite):
    """Isolated data dir per suite; models are shared through a symlink."""
    from app.engine import Engine
    from app.flow.service import FlowService
    from app.flow.indexing import FlowIndexer
    data = RUNS / suite["name"]
    data.mkdir(parents=True, exist_ok=True)
    if not (data / "models").exists():
        (data / "models").symlink_to(DATA / "models", target_is_directory=True)
    engine = Engine(data)
    flow = FlowService(engine)
    return engine, flow, FlowIndexer(flow)


def build_index(engine, flow, indexer, suite, components, timing=None):
    """`timing` (optional dict) receives scan seconds and per-lane indexer timing."""
    timing = {} if timing is None else timing
    scan_started = time.monotonic()
    library = engine.add_library(suite["images"])
    print("扫描文件", suite["images"], flush=True)
    job = engine.index_library(library["id"], {"id": "eval", "library_id": library["id"], "status": "scanning",
        "total": 0, "done": 0, "processed": 0, "skipped": 0, "removed": 0, "errors": [], "current": "",
        "cancel": threading.Event(), "scan_only": True})
    if job.get("status") == "error":
        raise RuntimeError(job.get("current"))
    timing["scan_seconds"] = round(time.monotonic() - scan_started, 2)
    catalog = flow.catalog(library["id"])
    if suite['name'].startswith('photobench_') and any(a.get('latitude') is not None for a in catalog):
        from app.metadata import OfflinePlaces
        places = OfflinePlaces()
        places.load()
        if places.version == 'missing':
            raise RuntimeError('评测照片有 GPS，但缺少离线行政区数据；请先运行 scripts/prepare_geography.py')
    if components:
        job = {"id": "eval-index", "library_id": library["id"], "status": "running", "total": 0, "done": 0,
               "processed": 0, "skipped": 0, "removed": 0, "errors": [], "current": "", "started": time.time(),
               "cancel": threading.Event()}
        reporter = threading.Thread(target=_report, args=(job,), daemon=True)
        reporter.start()
        indexer.build(job, components)
        timing["lanes"] = lane_timing(job.get("timing", {}))
        if job.get("status") != "complete" or job.get("error_count", 0):
            raise RuntimeError(f"索引未完整建立: status={job.get('status')}, errors={job.get('error_count', 0)}, details={job.get('errors')}, current={job.get('current')}")
        print(f"\n索引完成：处理 {job['processed']}，复用 {job['skipped']}，错误 {job.get('error_count', 0)}", flush=True)
    return library


def lane_timing(raw):
    """Round indexer timing and add per-processed-photo seconds."""
    lanes = {}
    for lane, t in raw.items():
        n = t["processed"]
        lanes[lane] = {**{k: round(v, 2) if isinstance(v, float) else v for k, v in t.items()},
                       "seconds_per_photo": round((t["load_seconds"] + t["work_seconds"]) / n, 3) if n else None,
                       "work_seconds_per_photo": round(t["work_seconds"] / n, 3) if n else None}
    return lanes


def device_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        return "unknown"


def _report(job):
    while "finished" not in job:
        print(f"\r  {job['done']}/{job['total']} {str(job.get('current', ''))[:60]:<60}", end="", flush=True)
        time.sleep(5)


# ---------------------------------------------------------------- metrics

def ranks_of(order, targets):
    position = {aid: i + 1 for i, aid in enumerate(order)}
    return [position.get(t) for t in targets]


def summarize(ranks, total_targets):
    """Per query: hit@k = any target within k; recall@k = share of targets within k."""
    found = [r for r in ranks if r]
    out = {f"hit@{k}": float(any(r <= k for r in found)) for k in KS}
    out.update({f"recall@{k}": sum(r <= k for r in found) / total_targets for k in KS})
    out["mrr"] = 1 / min(found) if found else 0.0
    return out


def mean(rows, key):
    values = [r[key] for r in rows if key in r]
    return round(sum(values) / len(values), 4) if values else None


def auroc(positive, negative):
    """Mann–Whitney AUROC；分数越大越可能有目标，平分计半。"""
    positive = [v for v in positive if v is not None and math.isfinite(v)]
    negative = [v for v in negative if v is not None and math.isfinite(v)]
    if not positive or not negative:
        return None
    pairs = sorted([(v, 1) for v in positive] + [(v, 0) for v in negative])
    wins, lower, i = 0., 0, 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        p = sum(label for _, label in pairs[i:j])
        n = j - i - p
        wins += p * (lower + n / 2)
        lower += n
        i = j
    return wins / (len(positive) * len(negative))


def existence_summary(rows):
    out = {}
    for key in ('primary_top1', 'primary_gap1_10'):
        groups = []
        distributions = {}
        for no_target, name in ((False, 'with_target'), (True, 'no_target')):
            selected = [r for r in rows if r['no_target'] == no_target]
            values = sorted(v for r in selected if (v := r['final'].get('existence_signal', {}).get(key)) is not None and math.isfinite(v))
            groups.append(values)
            distributions[name] = {'n': len(values), 'missing': len(selected) - len(values),
                'mean': sum(values) / len(values) if values else None,
                **{label: values[round((len(values) - 1) * fraction)] if values else None
                   for label, fraction in [('min', 0), ('p25', .25), ('median', .5), ('p75', .75), ('max', 1)]}}
        out[key] = {**distributions, 'auroc': auroc(*groups), 'positive_class': 'with_target'}
    return out


def oracle_people(flow, library, suite, by_name, names):
    """答案泄漏的命名上限代理；只注入本次进程，不修改用户命名或数据库。"""
    if not suite['name'].startswith('photobench_'):
        raise ValueError('oracle 人物实验仅用于 PhotoBench')
    with flow.e.db() as con:
        rows = con.execute("""SELECT DISTINCT f.person_id,f.asset_id FROM flow_faces f JOIN assets a
            ON a.id=f.asset_id AND a.fingerprint=f.fingerprint WHERE a.library_id=? AND f.version=?""",
            (library['id'], flow.tools.faces.version)).fetchall()
    groups = {}
    for r in rows:
        groups.setdefault(r['person_id'], set()).add(r['asset_id'])
    mapping = {}
    report = {}
    for name in names:
        targets = {by_name[t] for q in suite['queries'] if q['tags'].get('Person') and any(name in text for text in q['turns'])
                   for t in q['targets'] if t in by_name}
        candidates = sorted(groups, key=lambda pid: (-len(groups[pid] & targets), pid))
        if candidates and groups[candidates[0]] & targets:
            pid = candidates[0]
            mapping[name] = sorted(groups[pid])
            report[name] = {'group': pid, 'overlap_targets': len(groups[pid] & targets), 'group_images': len(groups[pid])}
        else:
            report[name] = {'group': None, 'overlap_targets': 0}
    flow.recall.oracle_people = mapping
    return {'enabled': True, 'label': 'ORACLE：使用验证答案命名组，含泄漏，非正常评测', 'names': report,
            'limitation': '按答案照片重叠最多的组命名，无法区分合照中的真实身份；不是身份识别真值'}


# ---------------------------------------------------------------- runners

def run_recall(engine, flow, library, query, args):
    """Recall stage only: returns fused order and each channel's own order."""
    from app.flow.domain import fallback_conditions, expression_for
    assets = flow.catalog(library["id"])
    config = flow.store.config()
    conditions, per_turn = [], []
    for text in query["turns"]:
        conditions = fallback_conditions(text, conditions)
        state = {"conditions": conditions, "expression": expression_for(conditions),
                 "raw_query": "；".join(query["turns"][:len(per_turn) + 1]), "excluded": [],
                 "reference": {"asset_id": query["reference_id"]} if query.get("reference_id") else None}
        order, _, _, warnings, _, channels = flow.recall.search(assets, state, config, args.depth)
        signal = next((c['existence_signal'] for c in channels if 'existence_signal' in c), {})
        per_turn.append((order, {c["name"]: c["ids"] for c in channels}, warnings, {'existence_signal': signal,
            'channel_families': {c['name']: c['family'] for c in channels},
            'channel_weights': {c['name']: c.get('weight', 1) for c in channels}}))
    return per_turn


def run_full(engine, flow, library, query, args):
    """Whole controller, one turn after another in one session, as the UI does."""
    session, revision, per_turn = None, 0, []
    for text in query["turns"]:
        reference = {"asset_id": query["reference_id"]} if query.get("reference_id") and session is None else None
        view = flow.start(library["id"], message=text, session_id=session, expected_revision=revision,
                          mode=args.mode, reference=reference)
        session, revision, rid = view["session"]["id"], view["session"]["revision"], view["id"]
        deadline = time.time() + args.timeout
        while time.time() < deadline:
            run = flow.store.get_run(rid)
            if run["status"] not in {"running", "queued", "pausing"}:
                break
            time.sleep(1)
        else:
            flow.pause(rid)
            raise TimeoutError(f"完整链路超过 {args.timeout} 秒；已请求暂停，不能作为完成结果计分")
        run = flow.store.get_run(rid)
        if run["status"] in {"running", "queued", "pausing", "error", "failed"}:
            raise RuntimeError(f"完整链路未成功结束: status={run['status']}, reason={run.get('stop_reason')}")
        revision = flow.store.get_session(session)["revision"]
        per_turn.append((run["order"][:args.depth], {}, run["warnings"], run["stop_reason"],
                         flow.coverage(rid), run.get("elapsed_ms"), run["status"], {"existence_signal": run.get("existence_signal", {})}))
    return per_turn


def evaluate(suite, args):
    engine, flow, indexer = engine_for(suite)
    oracle_names = getattr(args, 'oracle_people', None)
    if oracle_names:
        if args.stage != 'recall':
            raise ValueError('oracle 人物实验只允许 recall 阶段')
        flow.configure({'people_enabled': True})
        if 'people' not in args.index:
            args.index = [*args.index, 'people']
    if args.config:
        flow.configure(json.loads(args.config))
    index_started = time.monotonic()
    index_timing = {}
    library = build_index(engine, flow, indexer, suite, args.index, index_timing)
    index_seconds = round(time.monotonic() - index_started, 2)
    assets = flow.catalog(library["id"])
    by_name = {a["name"]: a["id"] for a in assets}
    if len(by_name) != len(assets):
        raise RuntimeError("图库中存在重复文件名，无法可靠映射答案")
    oracle = oracle_people(flow, library, suite, by_name, oracle_names) if oracle_names else {'enabled': False}
    names_by_id = {a["id"]: a["name"] for a in assets}
    queries = suite["queries"][args.offset:args.offset + args.limit if args.limit else None]
    missing_files = sorted({t for q in queries for t in q["targets"] if t not in by_name})
    missing_refs = sorted({q["reference"] for q in queries if q.get("reference") and q["reference"] not in by_name})
    if missing_files or missing_refs:
        raise RuntimeError(f"图库缺少答案图片 {len(missing_files)} 张或参考图 {len(missing_refs)} 张: {(missing_files + missing_refs)[:10]}")
    out = RESULTS / f"{suite['name']}-{args.stage}-{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1000000:06d}"
    out.mkdir(parents=True, exist_ok=True)
    rows, missing, errors = [], set(), []
    with open(out / "queries.jsonl", "w") as log:
        for n, query in enumerate(queries, 1):
            targets = [by_name[t] for t in query["targets"] if t in by_name]
            query = {**query, "reference_id": by_name.get(query.get("reference"))}
            missing |= {t for t in query["targets"] if t not in by_name}
            started = time.time()
            try:
                turns = (run_recall if args.stage == "recall" else run_full)(engine, flow, library, query, args)
            except Exception as exc:
                error = {"id": query["id"], "error": str(exc)[:400], "traceback": traceback.format_exc()}
                errors.append(error)
                log.write(json.dumps(error, ensure_ascii=False) + "\n")
                log.flush()
                print(f"[{n}/{len(queries)}] {query['id']} 错误: {exc}", flush=True)
                continue
            record = {"id": query["id"], "turns": query["turns"], "targets": query["targets"], "tags": query["tags"],
                      "seconds": round(time.time() - started, 2), "per_turn": []}
            for turn in turns:
                order, channels = turn[0], turn[1]
                item = {"ranks": ranks_of(order, targets), "top10": order[:10], "top10_names": [names_by_id.get(a, a) for a in order[:10]], "warnings": turn[2]}
                if targets:
                    item.update(summarize(item["ranks"], len(targets)))
                # Channel ranks show which route found or lost each target.
                item["channel_ranks"] = {name: ranks_of(ids, targets) for name, ids in channels.items()}
                # Preserve each channel's candidate order for reproducible fusion analysis.
                item["channel_orders"] = {name: [names_by_id.get(a, a) for a in ids[:args.depth]]
                                          for name, ids in channels.items()}
                if len(turn) == 4 and isinstance(turn[3], dict):
                    item.update(turn[3])
                elif len(turn) > 3:
                    item.update(stop_reason=turn[3], coverage=turn[4], elapsed_ms=turn[5], status=turn[6])
                    if len(turn) > 7:
                        item.update(turn[7])
                record["per_turn"].append(item)
            record["final"] = record["per_turn"][-1] if record["per_turn"] else {}
            record["no_target"] = not query["targets"]
            rows.append(record)
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.flush()
            final = record["final"]
            print(f"[{n}/{len(queries)}] {query['id']} rank={final.get('ranks')} {record['seconds']}s", flush=True)
    scored = [r["final"] for r in rows if not r["no_target"] and r["final"]]
    summary = {"suite": suite["name"], "stage": args.stage, "index": args.index, "queries": len(rows),
               "requested_queries": len(queries), "successful_queries": len(rows), "failed_queries": len(errors),
               "queries_with_warnings": sum(any(t.get("warnings") for t in r["per_turn"]) for r in rows),
               "queries_paused": sum(any(t.get("status") == "paused" for t in r["per_turn"]) for r in rows),
               "selection": {"offset": args.offset, "limit": args.limit}, "depth": args.depth, "mode": args.mode,
               "metadata_coverage": {"gps": sum(a.get('latitude') is not None for a in assets),
                   "place": sum(bool(json.loads(a.get('place_names') or '[]')) for a in assets),
                   "captured": sum(bool(a.get('captured')) for a in assets),
                   "versions": sorted({a.get('metadata_version') for a in assets if a.get('metadata_version')})},
               "gallery_images": len(assets), "index_seconds": index_seconds,
               "device": device_name(), "index_timing": index_timing,
               "query_seconds_total": round(sum(r["seconds"] for r in rows), 2),
               "with_target": len(scored), "no_target": sum(r["no_target"] for r in rows),
               "missing_target_files": sorted(missing)[:50],
               **{key: mean(scored, key) for key in [f"hit@{k}" for k in KS] + [f"recall@{k}" for k in KS] + ["mrr"]},
               "mean_seconds": mean(rows, "seconds"), "config": flow.store.config()}
    summary['oracle_people'] = oracle
    summary['existence_signal'] = existence_summary(rows)
    # Multi-turn suites: hit@10 after each round shows whether more turns help.
    rounds = max((len(r["per_turn"]) for r in rows), default=0)
    if rounds > 1:
        summary["hit@10_by_turn"] = [mean([r["per_turn"][i] for r in rows if not r["no_target"] and len(r["per_turn"]) > i], "hit@10")
                                     for i in range(rounds)]
    channels = sorted({name for r in scored for name in r.get("channel_ranks", {})})
    summary["channel_query_count"] = {name: sum(name in r.get("channel_ranks", {}) for r in scored) for name in channels}
    summary["channel_hit@50"] = {name: mean([{"v": float(any(x and x <= 50 for x in r["channel_ranks"][name]))}
                                            for r in scored if name in r.get("channel_ranks", {})], "v") for name in channels}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, ensure_ascii=False, indent=2))
    print("结果保存在", out)
    if errors:
        raise RuntimeError(f"{len(errors)} 条查询失败；汇总指标仅包含成功查询，详见 {out / 'queries.jsonl'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("suite", nargs="?", help="见 --list；disbench 会依次评测全部 57 个用户")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--stage", choices=["recall", "full"], default="recall",
                        help="recall=只看召回与各通道名次；full=完整控制器（解析/精排/核验）")
    parser.add_argument("--index", nargs="*", default=["visual"], choices=["visual", "ocr", "description", "people"],
                        help="先建立的索引组件；传空（--index）只扫描文件、复用已有索引")
    parser.add_argument("--mode", default="find_one", choices=["find_one", "set", "all"])
    parser.add_argument("--depth", type=int, default=200, help="记录名次的候选深度")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条查询（0=全部）")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=600, help="full 模式每轮最长等待秒数")
    parser.add_argument('--oracle-people', nargs='+', metavar='姓名',
                        help='ORACLE/答案泄漏上限：用完整验证集答案为重叠最多的人脸组命名，默认关闭；仅 recall')
    parser.add_argument("--config", help='覆盖 flow 设置，例如 \'{"auto_rerank": true, "verify_budget": 8}\'')
    args = parser.parse_args()
    if args.list or not args.suite:
        for name, images, queries in ready():
            print(f"{name:<34} 图片 {images:<7} 查询 {queries}")
        return
    for suite in suites(args.suite):
        evaluate(suite, args)


if __name__ == "__main__":
    main()
