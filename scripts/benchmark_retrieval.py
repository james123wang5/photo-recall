"""Known user regressions, not a held-out accuracy benchmark.

Run after the complete parallel index has been built. No filename-specific
ranking is used; answer IDs are only used to measure each stage's ranks.
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
import json
import argparse
import threading
import time
from app.engine import Engine
from app.retrieval import semantic_scores

e = Engine()
library = "ebd5889d68aaac92"
with e.db() as con:
    assets = [dict(r) for r in con.execute("SELECT * FROM assets WHERE library_id=? AND error IS NULL", (library,))]
cases = [
    {"query": "生日蛋糕，灯光暗，有人双手合十许愿", "targets": ["1dc046067828e9f8496cfef4", "95a2523f446c1de62d24497a"],
     "conditions": [
        {"label": "生日蛋糕", "english": "a birthday cake with candles", "kind": "include", "concept": "object", "slot": "object"},
        {"label": "灯光暗", "english": "dim lighting", "kind": "include", "concept": "lighting", "slot": "lighting"},
        {"label": "双手合十许愿", "english": "a person holding their hands together making a wish", "kind": "include", "concept": "gesture", "slot": "gesture"}]},
    {"query": "只有一个人，和汽车合照", "targets": ["263d944700429c339e4a635e"],
     "conditions": [
        {"label": "和汽车合照", "english": "a person posing with a car", "kind": "include", "concept": "composition", "slot": "composition"},
        {"label": "只有一个人", "english": "exactly one person", "kind": "include", "concept": "person", "slot": "person", "constraint": {"entity": "person", "op": "eq", "count": 1}}]},
    {"query": "一个人坐在汽车驾驶位，手扶方向盘", "targets": ["263d944700429c339e4a635e"],
     "conditions": [
        {"label": "坐在汽车驾驶位", "english": "a person sitting in the driver's seat of a car", "kind": "include", "concept": "composition", "slot": "composition"},
        {"label": "手扶方向盘", "english": "a hand resting on the steering wheel", "kind": "include", "concept": "gesture", "slot": "gesture"}]},
    {"query": "戴皇冠，闭着眼睛，蛋糕前双手合十", "targets": ["1dc046067828e9f8496cfef4", "95a2523f446c1de62d24497a"],
     "conditions": [
        {"label": "戴皇冠", "english": "wearing a paper birthday crown", "kind": "include", "concept": "object", "slot": "object"},
        {"label": "闭着眼睛双手合十", "english": "eyes closed with palms together", "kind": "include", "concept": "gesture", "slot": "gesture"},
        {"label": "蛋糕前", "english": "in front of a birthday cake", "kind": "include", "concept": "composition", "slot": "composition"}]},
]
report = {"kind": "known-regressions-not-held-out", "library_count": len(assets), "cases": []}
parser = argparse.ArgumentParser()
parser.add_argument("--cases")
parser.add_argument("--output", default="benchmark-v04.json")
parser.add_argument("--budget", type=int, default=64)
args = parser.parse_args()
if args.cases:
    cases = json.loads(__import__('pathlib').Path(args.cases).read_text())
    report["kind"] = "small-manually-reviewed-development-tasks-not-held-out"
report["budget"] = args.budget
for case in cases:
    before = time.time()
    conditions = case["conditions"]
    base, _ = semantic_scores(assets, conditions, e.provider, {})
    old_order = [assets[i]["id"] for i in __import__('numpy').argsort(-base)]
    s = e.session(library)
    s["conditions"] = conditions
    e.save_session(s)
    job = {"status": "running", "query": case["query"], "started": time.time(), "cancel": threading.Event(),
           "results": [], "done": 0, "errors": [], "indexed": e.pipeline.counts(library), "confirmed": 0}
    e.pipeline.refine(s, assets, {}, job, budget=args.budget)
    order = [r["id"] for r in job["results"]]
    item = {"query": case["query"], "status": job["status"], "elapsed_seconds": round(time.time()-before,2),
        "baseline_ranks": {aid: old_order.index(aid)+1 for aid in case["targets"]},
        "new_ranks": {aid: order.index(aid)+1 if aid in order else None for aid in case["targets"]},
        "top5": [{"id":r["id"],"name":r["name"],"rerank_score":r.get("rerank_score"),
                   "inspection_tier":r.get("inspection_tier")} for r in job["results"][:5]],
        "reranked": job["done"], "errors": job["errors"]}
    report["cases"].append(item)
    (e.data / args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(item,ensure_ascii=False),flush=True)
print("已保存 data/benchmark-v04.json；以上仅为已知回归，不是泛化准确率。")
