"""A transparent prototype parser; terms are preserved, never invented by a model."""
import re
import uuid

CONCEPTS = {
    "tree": ["树", "树林", "森林", "tree", "trees", "forest"],
    "car": ["车", "汽车", "car", "cars", "vehicle"],
    "white": ["白色", "白车", "white"],
    "silver": ["银色", "silver"],
    "red": ["红色", "红衣", "red"],
    "person": ["人", "人物", "person", "people"],
    "beach": ["海边", "沙滩", "海滩", "beach", "seaside"],
    "sunset": ["日落", "夕阳", "黄昏", "sunset"],
    "mountain": ["山", "雪山", "mountain", "mountains"],
    "snow": ["雪", "snow"],
    "cat": ["猫", "cat", "cats"],
    "dog": ["狗", "dog", "dogs"],
    "birthday": ["生日", "蛋糕", "birthday", "cake"],
    "night": ["夜晚", "夜景", "晚上", "night"],
    "city": ["城市", "街道", "city", "street"],
    "lake": ["湖", "lake"],
    "countryside": ["郊外", "乡村", "countryside", "rural"],
}
LABELS = {"tree": "树", "car": "车", "white": "白色", "silver": "银色", "red": "红色", "person": "人物",
          "beach": "海边", "sunset": "日落", "mountain": "山", "snow": "雪", "cat": "猫", "dog": "狗",
          "birthday": "生日 / 蛋糕", "night": "夜晚", "city": "城市", "lake": "湖", "countryside": "郊外"}


def contains(text, alias):
    return bool(re.search(r"\b" + re.escape(alias) + r"\b", text)) if alias.isascii() else alias in text


def update_conditions(previous, message, mode="append"):
    items = [dict(x) for x in previous] if mode == "append" else []
    text = message.lower().strip()
    uncertain = bool(re.search(r"可能|好像|大概|不确定|maybe|perhaps|possibly", text))
    # Explicit colour corrections replace the old colour instead of accumulating conflicts.
    if re.search(r"不是.*(?:是|而是)|actually|instead", text) and any(contains(text, a) for k in ["white", "silver", "red"] for a in CONCEPTS[k]):
        items = [x for x in items if x.get("concept") not in ["white", "silver", "red"]]
    found = []
    for key, aliases in CONCEPTS.items():
        hits = []
        for alias in aliases:
            match = re.search((r"\b" + re.escape(alias) + r"\b") if alias.isascii() else re.escape(alias), text)
            if match:
                hits.append((match.start(), match.end()))
        if not hits:
            continue
        start, end = min(hits)
        prefix = text[max(0, start - 9):start]
        negative = bool(re.search(r"没有\s*$|不要\s*$|不含\s*$|不是\s*$|without\s*$|no\s*$|not\s*$", prefix))
        is_remove = bool(re.search(r"取消|去掉|撤销|不限定|remove", prefix))
        items = [x for x in items if x.get("concept") != key]
        if is_remove:
            continue
        condition = {"id": uuid.uuid4().hex[:10], "concept": key, "label": LABELS[key],
                     "kind": "exclude" if negative else ("prefer" if uncertain else "include")}
        items.append(condition)
        found.append(key)
    for year in re.findall(r"\b(?:19|20)\d{2}\b", text):
        items = [x for x in items if x.get("concept") != "year"]
        items.append({"id": uuid.uuid4().hex[:10], "concept": "year", "label": year,
                      "kind": "prefer" if uncertain else "include"})
    # Preserve unsupported detail as a visible cue; the vision adapter can inspect it.
    # A known word must never swallow the rest of the sentence (lighting, gestures, etc.).
    if not re.search(r"取消|去掉|撤销|remove", text) and not re.fullmatch(r"(?:大概是|可能是)?(?:19|20)\d{2}年?", text):
        if not any(x.get("concept") == "detail" and x["label"] == message for x in items):
            items.append({"id": uuid.uuid4().hex[:10], "concept": "detail", "label": message[:1000], "kind": "prefer"})
    return items


def semantic_query(conditions):
    parts = []
    for c in conditions:
        label = c["label"]
        parts.append(("没有 " if c["kind"] == "exclude" else "") + label)
    return "; ".join(parts)
