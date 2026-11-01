"""Whole-library retrieval, grouped fusion and independent reference/event paths."""
import json
import re
from datetime import datetime
import numpy as np
from PIL import Image, ImageOps
from ..hybrid import tokens
from .domain import digest, query_text, fuse
from .providers import QwenRetrieval
from .tools import crop_image


class Recall:
    def __init__(self, engine, store, tools):
        self.e, self.store, self.tools = engine, store, tools
        self.qwen = {}

    def primary(self, config):
        if config["primary"] == "wemm":
            return self.e.pipeline.main
        key = ("embedding", config["max_pixels"])
        if key not in self.qwen:
            self.qwen[key] = QwenRetrieval(self.e.data / "models", *key)
        return self.qwen[key]

    def reranker(self, config):
        if config["reranker"] == "core":
            return self.e.pipeline.reranker
        key = ("reranker", config["max_pixels"])
        if key not in self.qwen:
            self.qwen[key] = QwenRetrieval(self.e.data / "models", *key)
        return self.qwen[key]

    def vector(self, provider, version, text):
        return self.e.pipeline.text_vector(provider, version, text)

    def rows(self, assets, version, legacy=False, siglip=False):
        if not assets:
            return []
        valid = {a["id"]: a["fingerprint"] for a in assets}
        with self.e.db() as con:
            rows = [dict(r) for r in con.execute("""SELECT v.* FROM flow_vectors v JOIN assets a
                ON a.id=v.asset_id AND a.fingerprint=v.fingerprint WHERE a.library_id=? AND v.version=?""", (assets[0]["library_id"], version))]
            if legacy:
                rows += [dict(r) for r in con.execute("""SELECT v.* FROM feature_vectors v JOIN assets a
                    ON a.id=v.asset_id AND a.fingerprint=v.fingerprint WHERE a.library_id=? AND v.version=?""", (assets[0]["library_id"], version))]
        if siglip:
            rows += [{"asset_id": a["id"], "fingerprint": a["fingerprint"], "region": "full",
                      "vector": a["embedding"]} for a in assets if a.get("embedding") and a.get("model") == self.e.provider.name]
            with self.e.db() as con:
                rows += [dict(r) for r in con.execute("""SELECT v.* FROM feature_vectors v JOIN assets a
                    ON a.id=v.asset_id AND a.fingerprint=v.fingerprint WHERE a.library_id=? AND v.version=?""",
                    (assets[0]["library_id"], self.e.pipeline.local_version))]
        return [r for r in rows if valid.get(r["asset_id"]) == r["fingerprint"]]

    @staticmethod
    def rank_vectors(rows, vector, limit):
        best, regions = {}, {}
        compatible = [r for r in rows if len(r["vector"]) == vector.size * 4]
        for start in range(0, len(compatible), 256):
            batch = compatible[start:start + 256]
            matrix = np.stack([np.frombuffer(r["vector"], dtype=np.float32) for r in batch])
            values = matrix @ vector
            for row, score in zip(batch, values):
                aid = row["asset_id"]
                if np.isfinite(score) and (aid not in best or score > best[aid]):
                    best[aid] = float(score)
                    regions[aid] = row.get("region", "full")
        ids = sorted(best, key=lambda aid: (-best[aid], aid))[:limit]
        return ids, {aid: regions[aid] for aid in ids}

    def search(self, assets, query, config, limit, recovery=0, cancel=None, quick=False):
        channels, warnings, regions = [], [], {}
        def stopped():
            return cancel is not None and cancel.is_set()
        def add(name, family, ids, weight=1):
            if ids:
                channels.append({"name": name, "family": family, "ids": ids, "weight": weight})
        text = query_text(query)
        if not text and not query.get("reference") and not query.get("person_id"):
            order = [a["id"] for a in sorted(assets, key=lambda a: (a.get("captured") or "", a["id"]), reverse=True)]
            return order[:limit], {}, {}, warnings, regions, [{"name": "浏览", "family": "browse", "ids": order}]
        primary = self.primary(config)
        # Keep raw input intact at the initial revision. After removal/correction,
        # query.raw_query is regenerated from ACTIVE conditions by the controller.
        variants = [value for value in dict.fromkeys([query.get("raw_query") or text, text]) if value.strip()]
        relaxed = query_text(query, relaxed=True)
        if relaxed and relaxed != text:
            variants.append(relaxed)
        variants = variants[:3]
        for provider, version, family, legacy in [
            (self.e.provider, self.e.provider.name, "siglip", False),
            (primary, primary.version, "primary", config["primary"] == "wemm")]:
            if quick and family == "primary":
                continue
            if stopped():
                break
            rows = self.rows(assets, version, legacy=legacy, siglip=family == "siglip")
            if not rows or not provider.installed:
                warnings.append(f"{family} 索引缺失或模型未准备，保留其他检索入口")
                continue
            try:
                for i, value in enumerate(variants):
                    if stopped():
                        break
                    vector = self.vector(provider, version, value)
                    ids, hits = self.rank_vectors(rows, vector, limit)
                    add(f"{family}：" + ("原文" if i == 0 else "条件" if i == 1 else "放松猜测"), family, ids)
                    regions.update({aid: box for aid, box in hits.items() if aid not in regions})
                ref = query.get("reference")
                if ref and not stopped():
                    reference = next((a for a in assets if a["id"] == ref["asset_id"]), None)
                    if reference:
                        path = self.tools.original(reference)
                        with Image.open(path) as source:
                            image = ImageOps.exif_transpose(source).convert("RGB")
                        if ref.get("box"):
                            image = crop_image(image, ref["box"])
                        vector = provider.image(image)
                        ids, hits = self.rank_vectors(rows, vector, limit)
                        add(family + "：参考局部" if ref.get("box") else family + "：参考照片", family, ids)
                        regions.update(hits)
            except Exception as exc:
                warnings.append(family + "：" + str(exc)[:240])
        if assets and not stopped():
            terms = tokens(text)[:64]
            if terms:
                expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
                with self.e.db() as con:
                    for kind, family in [("ocr", "ocr"), ("caption", "description")]:
                        rows = con.execute("""SELECT f.asset_id,bm25(observation_fts) AS score FROM observation_fts f
                            JOIN observations o ON o.asset_id=f.asset_id AND o.kind=f.kind JOIN assets a ON a.id=o.asset_id
                            AND a.fingerprint=o.fingerprint WHERE observation_fts MATCH ? AND f.kind=? AND a.library_id=?
                            ORDER BY score LIMIT ?""", (expression, kind, assets[0]["library_id"], limit)).fetchall()
                        add("照片文字" if kind == "ocr" else "画面描述关键词", family, [r["asset_id"] for r in rows])
            try:
                if quick:
                    raise LookupError("quick-path")
                semantic = self.e.hybrid.rankings(assets, [{"label": text, "english": text, "kind": "include"}], semantic=True)
                add("画面描述语义", "description", semantic.get("描述语义", [])[:limit])
            except LookupError:
                pass
            except Exception as exc:
                warnings.append("描述语义：" + str(exc)[:160])
        for c in query["conditions"]:
            years = re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", c["label"])
            if years:
                add("拍摄日期线索", "metadata", [a["id"] for a in assets if a.get("captured") and a["captured"][:4] in years][:limit])
        # User named identities have independent provenance; clustering alone
        # never turns a guessed name into an identity.
        if assets:
            with self.e.db() as con:
                people = con.execute("SELECT id,name FROM flow_people WHERE name!=''").fetchall()
                names = [p["id"] for p in people if p["name"] in text]
                if query.get("person_id"):
                    names = list(dict.fromkeys(names + [query["person_id"]]))
                for pid in names:
                    ids = [r[0] for r in con.execute("""SELECT DISTINCT f.asset_id FROM flow_faces f JOIN assets a
                        ON a.id=f.asset_id AND a.fingerprint=f.fingerprint WHERE a.library_id=? AND f.person_id=?""", (assets[0]["library_id"], pid))]
                    add("用户命名人物", "people", ids[:limit])
        excluded = set(query.get("excluded", []))
        allowed = {a["id"] for a in assets} - excluded
        for channel in channels:
            channel["ids"] = [aid for aid in channel["ids"] if aid in allowed]
        order, scores, sources = fuse(channels, limit)
        # A recovery uses fresh sources and a new base, not a tiny score offset.
        if recovery and order:
            seeds = order[:8]
            seed_dates = [a.get("captured") for a in assets if a["id"] in seeds and a.get("captured")]
            neighbors = []
            for asset in assets:
                try:
                    dt = datetime.fromisoformat(asset["captured"]) if asset.get("captured") else None
                    distance = min(abs((dt - datetime.fromisoformat(d)).total_seconds()) for d in seed_dates) if dt and seed_dates else float("inf")
                    if distance <= 36 * 3600 and asset["id"] in allowed:
                        neighbors.append((distance, asset["id"]))
                except (ValueError, TypeError):
                    continue
            add("同事件与邻近时间线索", "event", [aid for _, aid in sorted(neighbors)][:limit], .6)
            order, scores, sources = fuse(channels, limit)
        # Reserve OCR/identity entry points before truncating; collection later
        # visits every snapshot member regardless of its similarity score.
        for channel in channels:
            if channel["family"] in {"ocr", "people"}:
                for aid in channel["ids"][:8]:
                    if aid not in order:
                        if len(order) >= limit:
                            order.pop()
                        order.append(aid)
        if not channels:
            order = [a["id"] for a in assets if a["id"] in allowed][:limit]
            warnings.append("当前返回可浏览候选；没有可用的内容检索入口，不代表已识别画面")
        return order, scores, sources, warnings, regions, channels
