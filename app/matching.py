"""Result admission is separate from ranking; unchecked is never a match."""
from collections import Counter
from .retrieval import condition_terms, normalize_inspection


def match_decision(result, conditions):
    if not conditions:
        return {"match_status": "browse", "match_reason": "浏览照片"}
    if result.get("fact_conflict"):
        return {"match_status": "not_matched", "match_reason": "已记录的数量或时间与条件不符"}
    inspection = result.get("inspection")
    if not inspection:
        return {"match_status": "unreviewed", "match_reason": "尚未核对，仅为召回候选"}
    inspection = normalize_inspection(inspection)
    if inspection.get("warnings"):
        return {"match_status": "uncertain", "match_reason": "核对存在矛盾，暂不作为符合结果"}
    statuses = {item.get("term"): item.get("status") for item in inspection.get("conditions", [])}
    required = [statuses.get(term, "unknown") for term, c in zip(condition_terms(conditions), conditions)
                if c.get("kind") != "prefer" and not c.get("soft")]
    if any(status == "contradicted" for status in required):
        return {"match_status": "not_matched", "match_reason": "核对认为有条件不符，可在其他候选中复查"}
    if required and all(status == "matched" for status in required):
        return {"match_status": "matched", "match_reason": "模型逐项核对较符合，仍可能误判"}
    if not required and any(statuses.get(term) == "matched" for term in condition_terms(conditions)):
        return {"match_status": "matched", "match_reason": "有线索获得正向核对依据"}
    return {"match_status": "uncertain", "match_reason": "部分条件无法确认"}


def match_summary(results):
    counts = Counter(result.get("match_status", "unreviewed") for result in results)
    return {key: counts[key] for key in ("matched", "uncertain", "unreviewed", "not_matched", "browse")}
