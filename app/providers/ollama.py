"""Bounded local vision inspection, using an already installed Ollama model."""
import base64
import json
import urllib.request
import urllib.error
import threading
import re
import io
from PIL import Image, ImageOps
from .resources import inference, MANAGED_MODELS

BASE = "http://127.0.0.1:11434"
MODEL = "qwen3-vl:4b-instruct"
TEXT_MODEL = "qwen3:4b"
MODEL_LOCK = threading.RLock()


def explicit_quantity(item):
    """Recover literal quantities if the language model omitted their typed constraint."""
    digits = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    words = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10}
    if item.get("slot") == "year":
        return None
    text = item.get("label", "") + " " + item.get("evidence", "")
    match = re.search(r"(至少|最少|不少于|至多|最多|不超过|不多于|超过|多于|少于|不到|只有|仅有|恰好|正好)?\s*(\d+|[零一二两三四五六七八九十]+)\s*(?:个|位|名|只|辆|棵|座|把|张|幢|栋|件|台|条|支|架|艘|朵|(?=人))", text)
    english = item.get("english", "").lower()
    em = re.search(r"(?:(at least|at most|no more than|more than|less than|exactly|only)\s+)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+([a-z]+)", english)
    if not em:
        return None
    if match:
        modifier, number = match.groups()
        if number.isdigit():
            count = int(number)
        elif "十" in number:
            parts = number.split("十")
            if len(parts) != 2:
                return None
            count = (digits.get(parts[0], 1) if parts[0] else 1)*10 + (digits.get(parts[1], 0) if parts[1] else 0)
        else:
            count = digits.get(number)
            if count is None:
                return None
    else:
        modifier, number = em.group(1), em.group(2)
        count = int(number) if number.isdigit() else words[number]
        # English quantities must also be present in the user-backed source, not invented by a translation.
        if not re.search(r"\b"+re.escape(number)+r"\b", text, re.I):
            return None
    op = "gte" if modifier in {"至少","最少","不少于","at least","超过","多于","more than"} else "lte" if modifier in {"至多","最多","不超过","不多于","at most","no more than","少于","不到","less than"} else "eq"
    count += 1 if modifier in {"超过","多于","more than"} else -1 if modifier in {"少于","不到","less than"} else 0
    entity = em.group(3)
    entity = {"people":"person","persons":"person","children":"child","men":"person","women":"person"}.get(entity, entity)
    if entity.endswith("s") and not entity.endswith("ss"):
        entity = entity[:-1]
    if entity in {"red","white","black","blue","green","small","large","old","young"} or not 0 <= count <= 1000:
        return None
    return {"entity":entity,"op":op,"count":count}


def request_json(prompt, schema, image_path=None, budget=1000, model=None,keep_alive=0):
    MANAGED_MODELS.add(model or MODEL)
    content = {"role": "user", "content": prompt}
    if image_path:
        # Normalize orientation and supported transport format from the original,
        # retaining more detail than the 768px gallery thumbnail.
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((1536, 1536))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=92)
        content["images"] = [base64.b64encode(buffer.getvalue()).decode()]
    request = urllib.request.Request(BASE + "/api/chat", method="POST", headers={"Content-Type": "application/json"},
        data=json.dumps({"model": model or MODEL, "stream": False, "think": False, "format": schema,
            "messages": [content], "keep_alive": keep_alive, "options": {"temperature": 0, "num_ctx": 4096, "num_predict": budget}}).encode())
    with inference("ollama"), MODEL_LOCK:
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.load(response)
            content = payload.get("message", {}).get("content", "")
            if not content.strip():
                raise RuntimeError("本地模型未返回结构化结果，请重试或减少单次条件数量")
            return json.loads(content)


def plan_search(previous, message):
    """Open vocabulary intent parsing; no filename or query-specific ranking rules."""
    available = installed_models()
    if MODEL not in available and TEXT_MODEL not in available:
        raise RuntimeError("本地对话模型未就绪")
    item_schema = {"type": "object", "properties": {
        "label": {"type": "string"}, "english": {"type": "string"},
        "slot": {"type": "string"}, "evidence": {"type": "string"},
        "kind": {"type": "string", "enum": ["include", "prefer", "exclude"]},
        "constraint": {"anyOf": [{"type": "null"}, {"type": "object", "properties": {
            "entity": {"type": "string"}, "op": {"type": "string", "enum": ["eq", "gte", "lte"]},
            "count": {"type": "integer", "minimum": 0, "maximum": 1000}}, "required": ["entity", "op", "count"]}]}},
        "required": ["label", "english", "slot", "evidence", "kind", "constraint"]}
    schema = {"type": "object", "properties": {"conditions": {"type": "array", "maxItems": 16, "items": item_schema}},
              "required": ["conditions"]}
    prompt = ("你是照片搜索意图解析器。将用户的新描述合并到既有条件，返回完整且去重的条件列表。"
        "每个条件必须保留用户表达的一个独立画面要求。不限关键词，支持任意物体、人物、动作、光影、"
        "构图、视角、颜色、场景、文字或关系。拆分复合描述，不遗漏细节。label用简短中文，english为"
        "同一条件的简短英文视觉描述，必须保留数量和空间关系；exclude条件的english写被排除的正向内容。"
        "例如和车合拍应保留person posing with a car而非仅car。kind:确定要求include，可能/大概prefer，"
        "不含/没有exclude。不确定只影响对应的那条要求。slot是条件属性名称，用于合并修正，例如"
        "lighting/gesture/viewpoint/composition/scene/object/color/year；允许其他属性名称。"
        "用户纠正一个属性时替换旧值，撤销条件时移除；其他条件保留。evidence必须原样摘自新描述"
        "或既有条件label。不能推测未说过的细节：许愿不自动等于双手合十；不能猜人物身份、年份、地点。"
        "数量、只有、至少、至多、没有必须保留严格含义。constraint仅用于明确数量要求，entity为英文单数实体名称，"
        "op为eq/gte/lte，count为非负整数；其他要求constraint为null。只有一个人就是person eq 1，"
        "至少两个人必须是person gte 2，至多三辆车必须是car lte 3；数量条件的constraint不能为null，english写数量加实体，避免插入修饰词。"
        "不能简化为有人。明确没有某实体可用entity eq 0且kind include，普通排除条件用exclude。"
        "最多16条，不生成文件名搜索规则。\n既有条件：" + json.dumps(previous, ensure_ascii=False)
        + "\n新描述：" + message)
    result = request_json(prompt, schema, budget=1200, model=TEXT_MODEL if TEXT_MODEL in available else MODEL)
    source = message + "\n" + "\n".join(x["label"] for x in previous)
    valid = []
    for item in result.get("conditions", [])[:16]:
        if (isinstance(item, dict) and item.get("kind") in {"include", "prefer", "exclude"}
                and all(isinstance(item.get(k), str) and item[k].strip() for k in ["label", "english", "slot", "evidence"])
                and item["evidence"] in source):
            kind = item["kind"]
            uncertain = bool(re.search(r"好像|可能|大概|不确定|maybe|perhaps|possibly", item["evidence"], re.I))
            if uncertain and kind == "include":
                kind = "prefer"
            valid.append({"label": item["label"][:160], "english": item["english"][:200],
                          "concept": item["slot"][:40], "slot": item["slot"][:40],
                          "kind": kind, "soft": uncertain, "evidence": item["evidence"]})
            constraint = item.get("constraint")
            literal = explicit_quantity(item)
            if literal:
                constraint = literal
            if isinstance(constraint, dict) and isinstance(constraint.get("entity"), str) and constraint["entity"].strip() and constraint.get("op") in {"eq", "gte", "lte"} and type(constraint.get("count")) is int and 0 <= constraint["count"] <= 1000:
                valid[-1]["constraint"] = {"entity": constraint["entity"].strip().lower()[:60], "op": constraint["op"], "count": constraint["count"]}
    if not valid and result.get("conditions"):
        raise RuntimeError("对话模型未能返回有原文依据的条件")
    return valid


def description_model():
    available=installed_models()
    return "qwen3.5:4b" if "qwen3.5:4b" in available else MODEL


def describe(image_path,model=None,keep_alive=0):
    fields = ["caption", "lighting", "composition", "viewpoint", "objects", "people", "actions", "colors", "visible_text"]
    schema = {"type": "object", "properties": {field: {"type": "string"} for field in fields}, "required": fields}
    prompt = ("Describe observable visual content in English for later photo retrieval. Give a concise caption "
        "(at most 60 words). Separately describe lighting and shadows, framing/composition, camera viewpoint, "
        "objects, people's visible appearance, gestures/actions, colors, and legible text. Look carefully at hands "
        "and facial expressions. Use 'unknown' when unclear. Do not invent names, relationships, intentions, "
        "capture dates, or exact locations. Do not describe an unseen object. Return the requested JSON.")
    return request_json(prompt, schema, image_path, budget=650,model=model or description_model(),keep_alive=keep_alive)


def installed_models():
    try:
        with urllib.request.urlopen(BASE + "/api/tags", timeout=2) as response:
            return [x["name"] for x in json.load(response).get("models", [])]
    except (OSError, ValueError):
        return []


def count_entities(image_path, entities):
    """Reusable visible counts, independent of the desired answer or current query."""
    schema = {"type": "object", "properties": {"observations": {"type": "array", "minItems": len(entities),
        "maxItems": len(entities), "items": {"type": "object", "properties": {
            "entity": {"type": "string", "enum": entities},
            "count": {"type": ["integer", "null"], "minimum": 0},
            "evidence": {"type": "string", "maxLength": 100}}, "required": ["entity", "count", "evidence"]}}}, "required": ["observations"]}
    prompt = ("Count all distinctly visible instances of each requested entity in this photo. Inspect foreground AND background, "
              "including partially visible people. Count people, not just faces or the main subject. Enumerate their positions "
              "briefly in Chinese evidence (at most 40 Chinese characters). Do not count people printed on posters or photographs as real people. "
              "If occlusion, reflection, blur or small figures make the total uncertain, count must be null, never guess. "
              "An absent entity is 0. You are measuring facts, not trying to match a search query. Entities: " + json.dumps(entities))
    result = request_json(prompt, schema, image_path, budget=280)
    return {x["entity"]: {"count": x["count"], "evidence": str(x["evidence"])[:300]} for x in result.get("observations", [])
            if x.get("entity") in entities and (x.get("count") is None or type(x.get("count")) is int and x["count"] >= 0)}


def inspect(image_path, query, conditions):
    if MODEL not in installed_models():
        raise RuntimeError(f"本地 Ollama 中没有 {MODEL}。请先按 README 准备模型。")
    schema = {"type": "object", "properties": {
        "summary": {"type": "string", "maxLength": 200},
        "conditions": {"type": "array", "minItems": len(conditions), "maxItems": len(conditions), "items": {"type": "object", "properties": {
            "term": {"type": "string", "enum": conditions},
            "evidence": {"type": "string", "maxLength": 160},
            "status": {"type": "string", "enum": ["matched", "contradicted", "unknown"]}}, "required": ["term", "evidence", "status"]}}},
        "required": ["summary", "conditions"]}
    prompt = ("核对照片中可观察的事实，用中文回答。不要猜测拍摄年份、地点或人物身份。"
              "每条条件独立判断，包括光影、构图、视角、物体、人物和动作。不能因为总体像就全部判符合。"
              "数量要求按全画面判断，包括背景和部分可见的人物。合照、旁边、左侧等关系要求必须核对关系本身，"
              "仅在背景出现一个物体不能证明主体与该物体合照。先写可观察依据，再给结论；结论必须与依据一致。"
              "没有看见某细节、无法确认场所、场景存在多种解释，均应返回unknown；只有直接反证才能返回contradicted。"
              "普通场景描述按语义相关性判断，不擅自追加用户没要求的限制；吃饭不要求人物恰好正在咀嚼。"
              "条件看不清或不确定时标记 unknown，不能凭空补充。必须逐项核对所给条件，"
              "term 原样返回。用户描述：" + query + "。条件：" + json.dumps(conditions, ensure_ascii=False))
    result = request_json(prompt, schema, image_path, budget=1000)
    if not isinstance(result.get("summary"), str) or not isinstance(result.get("conditions"), list):
        raise RuntimeError("视觉模型返回格式不完整，请重试。")
    allowed = set(conditions)
    statuses = {"matched", "contradicted", "unknown"}
    checked = {}
    for item in result["conditions"]:
        if item.get("term") in allowed and item.get("status") in statuses:
            checked[item["term"]] = {"term": item["term"], "status": item["status"],
                                     "evidence": str(item.get("evidence", ""))[:400]}
    result["conditions"] = [checked.get(term, {"term": term, "status": "unknown", "evidence": "未返回可靠依据"})
                            for term in conditions]
    from ..retrieval import normalize_inspection
    return normalize_inspection(result)
