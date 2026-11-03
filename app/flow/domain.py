"""Pure query/evidence rules shared by the controller, storage and API.

No similarity score is a probability. Unknown, unreviewed and tool failure are
distinct; only explicit user exclusions remove an asset from the candidate view.
Compound fallback clauses stay intact so entity binding is not lost by splitting.
"""
import copy
import hashlib
import json
import re

UNCERTAIN = re.compile(r"可能|好像|大概|似乎|不确定|也许|maybe|perhaps|possibly", re.I)
DETAIL = re.compile(r"只有|至少|至多|没有|不含|不要|左|右|文字|写着|招牌|号码|手|喜鹊|乌鸦|exactly|without|left|right|text|count", re.I)
DEFAULTS = {
    "primary": "wemm", "reranker": "core", "vision_model": "qwen3-vl:4b-instruct",
    "planner": True, "auto_verify": True, "auto_rerank": False,
    "recall_limit": 200, "recovery_limit": 1000, "verify_budget": 4,
    "rerank_budget": 24, "time_slice": 60, "collection_batch": 8,
    "max_recoveries": 2, "max_pixels": 262144,
    "fusion_weights": {"primary": 1.0, "siglip": 0.1, "metadata_place": 1.0, "metadata_time": 1.0},
    "people_enabled": False, "grounding_enabled": False,
}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def condition(label, previous=(), soft=None, **extra):
    label = str(label).strip()[:1000]
    soft = bool(UNCERTAIN.search(label)) if soft is None else bool(soft)
    semantic = {"label": label, "soft": soft, **extra}
    key = digest(semantic)
    old = next((c for c in previous if c.get("semantic_key") == key), None)
    return {**semantic, "id": old["id"] if old else key[:16],
            "version": old.get("version", 1) if old else 1, "semantic_key": key}


def fallback_conditions(message, previous=(), replace=False):
    """Keep the user's full clauses, including negation, quantities and relations."""
    items = [] if replace else copy.deepcopy(list(previous))
    correction = re.search(r"不是(.+?)[，,]?(?:而是|应该是|是)(.+)", message)
    if correction:
        obsolete, replacement = correction.groups()
        items = [c for c in items if obsolete.strip() not in c["label"]]
        message = replacement
    for clause in re.split(r"[；;\n]+", message):
        if clause.strip():
            item = condition(clause, items, source=clause, parsed=False)
            if not any(c["semantic_key"] == item["semantic_key"] for c in items):
                items.append(item)
    return items[:32]


def expression_for(conditions):
    return {"all": [{"condition": c["id"]} for c in conditions if not c.get("soft")]}


def validate_expression(node, ids, depth=0):
    if depth > 8 or not isinstance(node, dict) or len(node) != 1:
        raise ValueError("条件表达式过深或格式不正确")
    if "condition" in node:
        if node["condition"] not in ids:
            raise ValueError("表达式引用了不存在的条件")
    elif "not" in node:
        validate_expression(node["not"], ids, depth + 1)
    elif next(iter(node)) in {"all", "any"}:
        children = next(iter(node.values()))
        if not isinstance(children, list) or len(children) > 32:
            raise ValueError("条件组合数量不正确")
        for child in children:
            validate_expression(child, ids, depth + 1)
    else:
        raise ValueError("不支持的条件运算")
    return node


def evaluate(node, statuses):
    if "condition" in node:
        return statuses.get(node["condition"], "unknown")
    if "not" in node:
        return {"supported": "conflict", "conflict": "supported"}.get(evaluate(node["not"], statuses), "unknown")
    op = "all" if "all" in node else "any"
    values = [evaluate(child, statuses) for child in node[op]]
    if not values:
        return "unknown"
    decisive, complete = ("conflict", "supported") if op == "all" else ("supported", "conflict")
    return decisive if decisive in values else complete if all(v == complete for v in values) else "unknown"


def query_text(query, relaxed=False):
    return "；".join(c["label"] for c in query["conditions"] if not (relaxed and c.get("soft")))


def evidence_key(query):
    return digest({"conditions": query["conditions"], "expression": query["expression"]})


def assess(query, evidence):
    if not query["conditions"]:
        return "browse"
    if not evidence:
        return "unreviewed"
    if evidence.get("tool_error"):
        return "error"
    verdict = evaluate(query["expression"], {c["condition_id"]: c["status"] for c in evidence.get("conditions", [])})
    # Joint checks prevent attributes from different entities being combined.
    joint = evidence.get("joint_status", "unknown")
    if verdict == "conflict" or joint == "conflict":
        return "conflict"
    if verdict == "supported" and joint == "supported":
        return "supported"
    return "unknown"


def fuse(channels, limit):
    """RRF with one bounded vote per independent representation family."""
    votes, sources = {}, {}
    for channel in channels:
        for rank, aid in enumerate(dict.fromkeys(channel["ids"]), 1):
            key = (channel["family"], aid)
            votes[key] = max(votes.get(key, 0), channel.get("weight", 1) / (60 + rank))
            sources.setdefault(aid, []).append({"channel": channel["name"], "family": channel["family"], "rank": rank})
    totals = {}
    for (_, aid), score in votes.items():
        totals[aid] = totals.get(aid, 0) + score
    order = sorted(totals, key=lambda aid: (-totals[aid], aid))[:limit]
    return order, totals, sources


def rank_results(base, evidence, query, reranked=None):
    """Bounded evidence offsets; recovery builds a NEW base before this step."""
    positions = {aid: i for i, aid in enumerate(base)}
    reranked = reranked or {}
    # Partial reranking only permutes the selected positions, never hides tails.
    selected = [aid for aid in base if aid in reranked]
    slots = sorted(positions[aid] for aid in selected)
    ordered = sorted(selected, key=lambda aid: (-reranked[aid], positions[aid]))
    positions.update(dict(zip(ordered, slots)))
    def key(aid):
        status = assess(query, evidence.get(aid))
        offset = -6 if status == "supported" else 6 if status == "conflict" else 0
        return positions[aid] + offset, positions[aid]
    return sorted(base, key=key)
