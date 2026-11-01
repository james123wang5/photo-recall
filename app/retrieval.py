"""General scene retrieval and evidence ranking; no asset-specific rules."""
import json
import re
import numpy as np
from PIL import Image

DESCRIPTION_FIELDS = ("caption", "lighting", "composition", "viewpoint", "objects", "people", "actions", "colors", "visible_text")


def description_vector(annotation, slot, dimension):
    if not annotation or not annotation.get("field_vectors"):
        return None
    fields = np.frombuffer(annotation["field_vectors"], dtype=np.float32)
    if fields.size != len(DESCRIPTION_FIELDS) * dimension:
        return None
    fields = fields.reshape(len(DESCRIPTION_FIELDS), dimension)
    name = {"object": "objects", "person": "people", "gesture": "actions", "action": "actions",
            "color": "colors", "scene": "caption", "text": "visible_text"}.get(slot, slot)
    if name in DESCRIPTION_FIELDS:
        vector = fields[DESCRIPTION_FIELDS.index(name)]
        return vector if np.linalg.norm(vector) > .1 else None
    return None


def visual_stats(path):
    with Image.open(path) as source:
        source = source.convert("RGB")
        source.thumbnail((96, 96))
        rgb = np.asarray(source, dtype=np.float32) / 255
    luminance = rgb @ np.array([.2126, .7152, .0722], dtype=np.float32)
    return {"luminance": float(luminance.mean()), "dark_fraction": float((luminance < .22).mean()),
            "contrast": float(luminance.std())}


def percentile(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.ptp(values) < 1e-8:
        return np.full(len(values), .5)
    # Equal scores must get equal ranks; filenames never affect semantic ranking.
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    return ((starts + .5 * counts) / len(values))[inverse]


def condition_terms(conditions):
    return [("不含 " if c["kind"] == "exclude" else "") + c["label"] for c in conditions]


def count_evidence(conditions, facts):
    evidence = {}
    for term, condition in zip(condition_terms(conditions), conditions):
        constraint = condition.get("constraint")
        if not constraint:
            continue
        fact = facts.get(constraint["entity"], {})
        count = fact.get("count")
        lower_bound = fact.get("lower_bound", 0)
        expected = constraint["count"]
        if count is None:
            if constraint["op"] in {"eq", "lte"} and lower_bound > expected:
                status = "contradicted"
            elif constraint["op"] == "gte" and lower_bound >= expected:
                status = "matched"
            else:
                status = "unknown"
            if condition["kind"] == "exclude" and status != "unknown":
                status = "matched" if status == "contradicted" else "contradicted"
        else:
            matched = {"eq": count == expected, "gte": count >= expected, "lte": count <= expected}[constraint["op"]]
            if condition["kind"] == "exclude":
                matched = not matched
            status = "matched" if matched else "contradicted"
        evidence[term] = {"term": term, "status": status,
                          "evidence": f"可见数量：{count if count is not None else ('至少 ' + str(lower_bound) if lower_bound else '无法确定')}。" + fact.get("evidence", "未获得可靠计数")}
    return evidence


def normalize_inspection(result):
    """Downgrade internally inconsistent verdicts; never guess the opposite verdict.

    Also applied when reading old caches. This intentionally catches only explicit
    linguistic contradictions, rather than claiming to verify visual evidence.
    """
    normalized = dict(result)
    items = [dict(item) for item in result.get("conditions", [])]
    summary = str(result.get("summary", "")).strip().lower()
    summary_positive = summary in {"符合", "匹配", "matched", "符合要求", "全部符合", "完全符合"}
    summary_negative = summary in {"不符合", "不匹配", "contradicted", "不符合要求", "全部不符合", "完全不符合"}
    warnings = list(result.get("warnings", []))
    for item in items:
        evidence = str(item.get("evidence", ""))
        # A direct positive conclusion and a negative status must not demote an
        # image. Negative wording is excluded, e.g. 不符合/无法确认符合.
        positive = bool(re.search(r"(?:^|[，。；,;\s])(?:完全|基本|均|都)?符合[^，。；,;]*[。！.!]?$", evidence))
        positive = positive or bool(re.search(r"\b(?:matches|satisfies) (?:the |this )?(?:condition|requirement|scene)\b", evidence, re.I))
        negative = bool(re.search(r"不符合|并非|无法|不能|不确定|未能|不满足|not |doesn't|unclear", evidence, re.I))
        mismatch = item.get("status") == "contradicted" and ((positive and not negative) or (summary_positive and len(items) == 1))
        mismatch = mismatch or (item.get("status") == "matched" and summary_negative and len(items) == 1)
        if mismatch:
            item.update(raw_status=item["status"], status="unknown", consistency="inconsistent")
            warnings.append("核对结论与依据存在矛盾，保留原始检索排名。")
    normalized["conditions"] = items
    normalized["warnings"] = list(dict.fromkeys(warnings))
    if warnings:
        normalized["raw_summary"] = result.get("raw_summary", result.get("summary", ""))
        normalized["summary"] = "核对结论不一致，保留原始检索依据。"
    return normalized


def strict_condition(condition):
    """Explicit user restrictions, distinct from ordinary semantic similarity."""
    if condition.get("soft") or condition.get("kind") == "prefer":
        return False
    if condition.get("constraint") or condition.get("concept") == "year" or condition.get("kind") == "exclude":
        return True
    original = str(condition.get("evidence") or condition.get("label", ""))
    return bool(re.search(r"必须|一定要|只能|仅有|只有|不要|不能|不得|恰好|至少|至多|\b(?:must|exactly|only|without|at least|at most)\b", original, re.I))


def inspection_grade(result, conditions):
    result = normalize_inspection(result)
    terms = condition_terms(conditions)
    statuses = {x["term"]: x["status"] for x in result.get("conditions", [])}
    required = [statuses.get(term, "unknown") for term, c in zip(terms, conditions) if c["kind"] != "prefer" and not c.get("soft")]
    contradiction = any(x == "contradicted" for x in required)
    tier = 0 if contradiction else (2 if required and all(x == "matched" for x in required) else 1)
    score = sum((.4 if c["kind"] == "prefer" or c.get("soft") else 1) * {"matched": 1, "contradicted": -2, "unknown": -.15}.get(statuses.get(term), -.15)
                for term, c in zip(terms, conditions))
    return {"inspection_tier": tier, "inspection_score": score,
            "inspection_uncertain": bool(result.get("warnings")),
            "inspection_strict_conflict": any(statuses.get(term)=="contradicted" and strict_condition(c)
                for term,c in zip(terms,conditions)),
            "required_matched": sum(x == "matched" for x in required), "required_count": len(required)}


def rank_key(item):
    return (-item.get("inspection_tier", 1), -item.get("inspection_score", 0), -item["score"], item["id"])


def semantic_scores(assets, conditions, provider, annotations):
    matrix = np.stack([np.frombuffer(a["embedding"], dtype=np.float32) for a in assets])
    positive = [c for c in conditions if c["kind"] != "exclude" and c.get("concept") != "year"]
    full_en = "A photograph of " + ", ".join(c.get("english", c["label"]) for c in positive)
    full_cn = "，".join(c["label"] for c in positive)
    if not positive:
        full_en = "A photograph"
        full_cn = "照片"
    full_vector = provider.text(full_en)
    base = .8 * percentile(matrix @ full_vector) + .2 * percentile(matrix @ provider.text(full_cn))
    facet_values, weights = [], []
    reasons = [[] for _ in assets]
    for condition in conditions:
        if condition.get("concept") == "year":
            continue
        english = condition.get("english", condition["label"])
        values = percentile(matrix @ provider.text("A photograph showing " + english))
        # Query the appropriate reusable description dimension, rather than only a generic caption.
        facet_vector = provider.text(english)
        described = np.array([float(vector @ facet_vector) if (vector := description_vector(annotations.get(a["id"]), condition.get("slot", condition.get("concept")), len(facet_vector))) is not None else np.nan for a in assets])
        available = np.isfinite(described)
        if available.sum() > 1:
            values[available] = .65 * values[available] + .35 * percentile(described[available])
        weight = .4 if condition["kind"] == "prefer" or condition.get("soft") else 1
        if condition["kind"] == "exclude":
            values = 1 - values
            weight = .4 if condition.get("soft") else .7
        # Luminance is supporting evidence only, never a hard filter or sole ranking basis.
        cue = (condition["label"] + " " + english).lower()
        dark = any(x in cue for x in ["昏暗", "较暗", "比较暗", "dim", "low light", "dark lighting", "dark room"])
        bright = any(x in cue for x in ["明亮", "bright lighting", "brightly lit"])
        if dark or bright:
            measurements = [json.loads(a.get("visual_stats") or "{}") for a in assets]
            measured = np.array([.65 * m.get("dark_fraction", .5) + .35 * (1 - m.get("luminance", .5)) for m in measurements])
            metric = percentile(measured)
            if bright or condition["kind"] == "exclude":
                metric = 1 - metric
            values = .65 * values + .35 * metric
        facet_values.append(values)
        weights.append(weight)
    if facet_values:
        facets = np.stack(facet_values)
        combined = .55 * base + .3 * np.average(facets, axis=0, weights=weights) + .15 * np.quantile(facets, .25, axis=0)
    else:
        combined = base
    # Rich descriptions improve retrieval after optional per-photo analysis.
    for i, asset in enumerate(assets):
        annotation = annotations.get(asset["id"])
        if annotation and annotation.get("embedding"):
            caption = np.frombuffer(annotation["embedding"], dtype=np.float32)
            if caption.shape == full_vector.shape:
                # Compare captions in the same library, rather than comparing raw text/image scales.
                reasons[i].append("详细画面索引")
    if annotations:
        caption_values = np.array([float(np.frombuffer(annotations[a["id"]]["embedding"], dtype=np.float32) @ full_vector)
                                  if a["id"] in annotations and annotations[a["id"]].get("embedding") else np.nan for a in assets])
        available = np.isfinite(caption_values)
        if available.sum() > 1:
            caption_rank = percentile(caption_values[available])
            combined[available] = .7 * combined[available] + .3 * caption_rank
    return combined, reasons
