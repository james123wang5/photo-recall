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
import threading
import time
import zipfile

DATA = ROOT / "data"
SETS = DATA / "eval_datasets"
RUNS = DATA / "eval_runs"
RESULTS = DATA / "eval_results"
KS = (1, 5, 10, 50)


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
                count = sum(1 for _ in Path(suite["images"]).glob("*")) if Path(suite["images"]).is_dir() else 0
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


def build_index(engine, flow, indexer, suite, components):
    library = engine.add_library(suite["images"])
    print("扫描文件", suite["images"], flush=True)
    job = engine.index_library(library["id"], {"id": "eval", "library_id": library["id"], "status": "scanning",
        "total": 0, "done": 0, "processed": 0, "skipped": 0, "removed": 0, "errors": [], "current": "",
        "cancel": threading.Event(), "scan_only": True})
    if job.get("status") == "error":
        raise RuntimeError(job.get("current"))
    if components:
        job = {"id": "eval-index", "library_id": library["id"], "status": "running", "total": 0, "done": 0,
               "processed": 0, "skipped": 0, "removed": 0, "errors": [], "current": "", "started": time.time(),
               "cancel": threading.Event()}
        reporter = threading.Thread(target=_report, args=(job,), daemon=True)
        reporter.start()
        indexer.build(job, components)
        print(f"\n索引完成：处理 {job['processed']}，复用 {job['skipped']}，错误 {job.get('error_count', 0)}", flush=True)
    return library


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
        per_turn.append((order, {c["name"]: c["ids"] for c in channels}, warnings))
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
        run = flow.store.get_run(rid)
        revision = flow.store.get_session(session)["revision"]
        per_turn.append((run["order"][:args.depth], {}, run["warnings"], run["stop_reason"],
                         flow.coverage(rid), run.get("elapsed_ms")))
    return per_turn


def evaluate(suite, args):
    engine, flow, indexer = engine_for(suite)
    if args.config:
        flow.configure(json.loads(args.config))
    library = build_index(engine, flow, indexer, suite, args.index)
    by_name = {a["name"]: a["id"] for a in flow.catalog(library["id"])}
    queries = suite["queries"][args.offset:args.offset + args.limit if args.limit else None]
    out = RESULTS / f"{suite['name']}-{args.stage}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    rows, missing = [], set()
    with open(out / "queries.jsonl", "w") as log:
        for n, query in enumerate(queries, 1):
            targets = [by_name[t] for t in query["targets"] if t in by_name]
            query = {**query, "reference_id": by_name.get(query.get("reference"))}
            missing |= {t for t in query["targets"] if t not in by_name}
            started = time.time()
            try:
                turns = (run_recall if args.stage == "recall" else run_full)(engine, flow, library, query, args)
            except Exception as exc:
                log.write(json.dumps({"id": query["id"], "error": str(exc)[:400]}, ensure_ascii=False) + "\n")
                continue
            record = {"id": query["id"], "turns": query["turns"], "targets": query["targets"], "tags": query["tags"],
                      "seconds": round(time.time() - started, 2), "per_turn": []}
            for turn in turns:
                order, channels = turn[0], turn[1]
                item = {"ranks": ranks_of(order, targets), "top10": order[:10], "warnings": turn[2]}
                if targets:
                    item.update(summarize(item["ranks"], len(targets)))
                # Channel ranks show which route found or lost each target.
                item["channel_ranks"] = {name: ranks_of(ids, targets) for name, ids in channels.items()}
                if len(turn) > 3:
                    item.update(stop_reason=turn[3], coverage=turn[4], elapsed_ms=turn[5])
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
               "with_target": len(scored), "no_target": sum(r["no_target"] for r in rows),
               "missing_target_files": sorted(missing)[:50],
               **{key: mean(scored, key) for key in [f"hit@{k}" for k in KS] + [f"recall@{k}" for k in KS] + ["mrr"]},
               "mean_seconds": mean(rows, "seconds"), "config": flow.store.config()}
    # Multi-turn suites: hit@10 after each round shows whether more turns help.
    rounds = max((len(r["per_turn"]) for r in rows), default=0)
    if rounds > 1:
        summary["hit@10_by_turn"] = [mean([r["per_turn"][i] for r in rows if not r["no_target"] and len(r["per_turn"]) > i], "hit@10")
                                     for i in range(rounds)]
    channels = sorted({name for r in scored for name in r.get("channel_ranks", {})})
    summary["channel_hit@50"] = {name: mean([{"v": float(any(x and x <= 50 for x in r["channel_ranks"].get(name, [])))}
                                            for r in scored], "v") for name in channels}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, ensure_ascii=False, indent=2))
    print("结果保存在", out)


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
