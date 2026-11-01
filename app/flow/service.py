"""Finite search controller: recall -> rerank/check -> recover/scan -> pause.

Workers never alter a newer query. Collection membership is a persisted snapshot,
and per-photo coverage is separate from similarity ranking and model evidence.
"""
import copy
import json
import re
import threading
import time
import uuid
from collections import Counter
from .domain import (DEFAULTS, DETAIL, condition, fallback_conditions, expression_for,
                     validate_expression, query_text, assess, rank_results)
from .store import FlowStore, encode
from .tools import Toolset, box_valid
from .recall import Recall


class FlowService:
    def __init__(self, engine):
        self.e = engine
        self.store = FlowStore(engine)
        self.tools = Toolset(engine, self.store)
        self.recall = Recall(engine, self.store, self.tools)
        self.lock = threading.RLock()
        self.workers = {}

    def configure(self, values):
        config = self.store.config()
        bounds = {"recall_limit": (20, 2000), "recovery_limit": (200, 20000), "verify_budget": (0, 32),
                  "rerank_budget": (1, 128), "time_slice": (5, 600), "collection_batch": (1, 64),
                  "max_recoveries": (1, 5), "max_pixels": (65536, 1843200)}
        for key, value in values.items():
            if key not in DEFAULTS:
                raise ValueError("未知设置：" + key)
            if key in bounds:
                low, high = bounds[key]
                if type(value) is not int or not low <= value <= high:
                    raise ValueError(f"{key} 必须在 {low} 到 {high} 之间")
            elif isinstance(DEFAULTS[key], bool) and type(value) is not bool:
                raise ValueError(key + " 必须为开关")
            elif key == "primary" and value not in {"wemm", "qwen"}:
                raise ValueError("不支持的召回模型")
            elif key == "reranker" and value not in {"core", "qwen"}:
                raise ValueError("不支持的精排模型")
            elif key == "vision_model" and (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,120}", value)):
                raise ValueError("视觉模型名称不正确")
            config[key] = value
        with self.e.db() as con:
            con.execute("INSERT OR REPLACE INTO flow_config VALUES (1,?)", (encode(config),))
        return config

    def catalog(self, library_id):
        with self.e.db() as con:
            return [dict(r) for r in con.execute("SELECT * FROM assets WHERE library_id=? ORDER BY id", (library_id,))]

    def start(self, library_id, message="", session_id=None, expected_revision=0,
              mode="find_one", replace=False, remove_id=None, reference=None,
              exclude_id=None, action="search", expression=None, scope=None, person_id=None):
        with self.lock:
            library = self.e.library(library_id)
            if library["is_demo"]:
                raise ValueError("完整链路使用真实照片库；预设插画可在旧版页面浏览")
            if mode not in {"find_one", "set", "all"}:
                raise ValueError("未知任务模式")
            if action not in {"search", "recover", "reset"}:
                raise ValueError("未知搜索动作")
            old = self.store.get_session(session_id) if session_id else None
            if old and old["library_id"] != library_id:
                raise ValueError("搜索会话不属于当前照片库")
            if old and old["revision"] != expected_revision:
                raise ValueError("搜索状态已更新，请刷新后重试")
            if old and self.workers.get(old.get("current_run")):
                self.workers[old["current_run"]][1].set()
            reset = action == "reset" or replace
            prior = [] if reset or not old else old["conditions"]
            conditions = fallback_conditions(message, prior, replace=reset) if message.strip() else copy.deepcopy(prior)
            if remove_id:
                if not any(c["id"] == remove_id for c in conditions):
                    raise ValueError("该条件已更新，请重新选择")
                conditions = [c for c in conditions if c["id"] != remove_id]
            assets = self.catalog(library_id)
            if scope is None and old and not reset:
                scope = old.get("scope")
            if scope is not None:
                selected = set(scope)
                if not selected <= {a["id"] for a in assets}:
                    raise ValueError("指定范围含有其他图库或已移除的照片")
                assets = [a for a in assets if a["id"] in selected]
            allowed = {a["id"] for a in assets}
            ref = reference if reference is not None else (None if reset or not old else old.get("reference"))
            if ref:
                if ref.get("asset_id") not in allowed or ref.get("box") is not None and not box_valid(ref["box"]):
                    raise ValueError("参考照片或局部不属于当前有效范围")
            exclusions = [] if reset or not old else list(old.get("excluded", []))
            if exclude_id:
                if exclude_id not in allowed:
                    raise ValueError("照片不属于当前范围")
                exclusions = list(dict.fromkeys(exclusions + [exclude_id]))
            recoveries = old.get("recoveries", 0) + 1 if action == "recover" and old else 0
            config = self.store.config()
            if recoveries > config["max_recoveries"]:
                raise ValueError("已达到本轮恢复次数，请修改线索或选择参考照片后继续")
            sid = old["id"] if old else uuid.uuid4().hex
            rid = uuid.uuid4().hex
            query = {"id": sid, "library_id": library_id, "revision": expected_revision + 1,
                "current_run": rid, "mode": mode, "conditions": conditions,
                "expression": expression if expression is not None else (copy.deepcopy(old["expression"])
                    if old and not reset and conditions == prior else expression_for(conditions)), "reference": ref,
                "excluded": exclusions, "recoveries": recoveries,
                "scope": scope,
                "person_id": person_id if person_id is not None else (old.get("person_id") if old and not reset else None),
                "messages": ([] if reset or not old else old.get("messages", []))[-29:] + ([message] if message.strip() else [])}
            query["raw_query"] = message if message.strip() and not prior and not remove_id else query_text(query)
            validate_expression(query["expression"], {c["id"] for c in conditions})
            run = {"id": rid, "session_id": sid, "query": query, "config": config, "status": "queued",
                "stage": "准备搜索", "revision": 0, "created": time.time(), "total": len(assets),
                "base": [], "order": [], "sources": {}, "scores": {}, "regions": {},
                "warnings": [], "stop_reason": "", "recall_done": False, "parser_done": False,
                "parse_message": message if action == "search" and not remove_id and expression is None else "",
                "previous_conditions": prior, "recoveries": recoveries, "rerank_scores": {}, "checked_this_slice": 0}
            self.store.replace(query, run, expected_revision, assets)
            if exclusions:
                with self.e.db() as con:
                    con.executemany("UPDATE flow_members SET state='excluded' WHERE run_id=? AND asset_id=?",
                                    [(rid, aid) for aid in exclusions])
            self.store.action(rid, "created", reason="用户恢复搜索" if recoveries else "新查询", query_revision=query["revision"])
            self._launch(run)
            return self.view(rid)

    def _launch(self, run, selected=None, regions=None):
        with self.lock:
            old = self.workers.get(run["id"])
            if old and old[0].is_alive():
                raise ValueError("当前任务正在执行或结束中，请稍后继续")
            cancel = threading.Event()
            run.update(status="running", stop_reason="", checked_this_slice=0)
            self.store.save(run)
            thread = threading.Thread(target=self._work, args=(run, cancel, selected, regions), daemon=True,
                                      name="photo-flow-" + run["id"][:8])
            self.workers[run["id"]] = (thread, cancel)
            thread.start()

    def _valid(self, run, cancel):
        return not cancel.is_set() and self.store.get_session(run["session_id"]).get("current_run") == run["id"]

    def _save(self, run):
        run["revision"] += 1
        if not self.store.save(run):
            raise InterruptedError("搜索条件已更新")

    def _members(self, run, pending=False, ids=None, limit=None):
        sql = "SELECT * FROM flow_members WHERE run_id=?"
        params = [run["id"]]
        if pending:
            sql += " AND state='unreviewed'"
        if ids is not None:
            if not ids:
                return []
            sql += " AND asset_id IN (" + ",".join("?" for _ in ids) + ")"
            params += list(ids)
        sql += " ORDER BY priority,ordinal"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self.e.db() as con:
            return [dict(r) for r in con.execute(sql, params)]

    def _assets(self, run):
        snapshot = {r["asset_id"]: r["fingerprint"] for r in self._members(run)}
        return [a for a in self.catalog(run["query"]["library_id"]) if snapshot.get(a["id"], object()) == a["fingerprint"]]

    def _apply_recall(self, run, result):
        order, scores, sources, warnings, regions, channels = result
        run.update(base=order, order=order, scores=scores, sources=sources, regions=regions,
                   warnings=list(dict.fromkeys(run["warnings"] + warnings)))
        if order and "first_preview_ms" not in run:
            run["first_preview_ms"] = round((time.time() - run["created"]) * 1000)
        with self.e.db() as con:
            con.execute("UPDATE flow_members SET candidate=0,priority=1000000+ordinal WHERE run_id=?", (run["id"],))
            con.executemany("UPDATE flow_members SET candidate=1,priority=? WHERE run_id=? AND asset_id=?",
                            [(i, run["id"], aid) for i, aid in enumerate(order)])
        self.store.action(run["id"], "recall", candidates=len(order), channels=[{"name": c["name"], "count": len(c["ids"])} for c in channels])
        self._save(run)

    def _reorder(self, run):
        evidence = {r["asset_id"]: json.loads(r["evidence"]) for r in self._members(run, ids=run["base"]) if r["evidence"]}
        run["order"] = rank_results(run["base"], evidence, run["query"], run["rerank_scores"])
        with self.e.db() as con:
            con.executemany("UPDATE flow_members SET priority=? WHERE run_id=? AND asset_id=?",
                            [(i, run["id"], aid) for i, aid in enumerate(run["order"])])

    def _record(self, run, aid, evidence):
        state = assess(run["query"], evidence)
        with self.e.db() as con:
            con.execute("UPDATE flow_members SET state=?,evidence=? WHERE run_id=? AND asset_id=?", (state, encode(evidence), run["id"], aid))
        return state

    def _work(self, run, cancel, selected, regions):
        started = time.monotonic()
        config, query = run["config"], run["query"]
        stop = "本轮处理完成"
        try:
            with self.e.hybrid.foreground():
                assets = self._assets(run)
                lookup = {a["id"]: a for a in assets}
                if not run["recall_done"]:
                    run["stage"] = "读取快速候选"
                    self._apply_recall(run, self.recall.search(assets, query, config, config["recall_limit"], cancel=cancel, quick=True))
                    if not self._valid(run, cancel):
                        raise InterruptedError()
                    if config["planner"] and run["parse_message"] and not run["parser_done"]:
                        run["stage"] = "整理可修改线索"
                        self._save(run)
                        try:
                            query["conditions"] = self.tools.parse(query, run["parse_message"], run["previous_conditions"])
                            query["expression"] = expression_for(query["conditions"])
                        except Exception as exc:
                            run["warnings"].append("条件解析暂不可用，保留完整原文：" + str(exc)[:180])
                        run["parser_done"] = True
                    if not self._valid(run, cancel):
                        raise InterruptedError()
                    run["stage"] = "合并全库检索与恢复线索"
                    self._save(run)
                    limit = config["recovery_limit"] if run["recoveries"] else config["recall_limit"]
                    self._apply_recall(run, self.recall.search(assets, query, config, limit, run["recoveries"], cancel))
                    run["recall_done"] = True
                    self._save(run)
                if not self._valid(run, cancel):
                    raise InterruptedError()
                if not query["conditions"]:
                    stop = "浏览候选；输入条件后可核验"
                    return
                if config["auto_rerank"] and not run.get("rerank_done"):
                    model = self.recall.reranker(config)
                    rerank_failed = False
                    if model.installed:
                        run["stage"] = "小批相关性精排"
                        self._save(run)
                        for aid in run["base"][:config["rerank_budget"]]:
                            if not self._valid(run, cancel) or time.monotonic() - started >= config["time_slice"]:
                                break
                            if aid not in lookup or aid in run["rerank_scores"]:
                                continue
                            try:
                                path = self.tools.original(lookup[aid])
                                run["rerank_scores"][aid] = model.score(query_text(query), str(path))
                            except Exception as exc:
                                run["warnings"].append("精排保留基础顺序：" + str(exc)[:160])
                                rerank_failed = True
                                break
                        self._reorder(run)
                    else:
                        run["warnings"].append("精排权重未准备，使用基础排序")
                        rerank_failed = True
                    run["rerank_done"] = rerank_failed or all(aid in run["rerank_scores"] or aid not in lookup
                        for aid in run["base"][:config["rerank_budget"]])
                if not self._valid(run, cancel):
                    raise InterruptedError()
                is_all = query["mode"] == "all"
                run["stage"] = "分批扫描图库范围" if is_all else "按需核对原图与局部"
                self._save(run)
                signature = self.tools.vision_signature(config["vision_model"])
                previous_signature = config.get("vision_signature")
                if previous_signature and not previous_signature.endswith(":unavailable") and previous_signature != signature:
                    raise RuntimeError("本地视觉模型版本已改变，请重新开始查询，避免混合不同版本的观察")
                config["vision_signature"] = signature
                budget = len(assets) if is_all else config["verify_budget"]
                if selected is not None:
                    budget = len(selected)
                elif not is_all and not config["auto_verify"]:
                    budget = 0
                # Easy semantic queries keep the fast path. Manual checks remain available.
                elif not is_all and not DETAIL.search(query_text(query)) and len(query["conditions"]) < 2:
                    budget = 0
                errors = 0
                while run["checked_this_slice"] < budget:
                    if not self._valid(run, cancel):
                        raise InterruptedError()
                    if time.monotonic() - started >= config["time_slice"]:
                        stop = "达到本轮时间预算，可以继续"
                        break
                    candidates = self._members(run, pending=selected is None,
                        ids=selected if selected is not None else (None if is_all else run["order"]),
                        limit=config["collection_batch"] if is_all else max(1, budget))
                    if selected is not None:
                        done = set(run.get("manual_done", []))
                        candidates = [r for r in candidates if r["asset_id"] not in done]
                    candidates = [r for r in candidates if r["asset_id"] not in query["excluded"]]
                    if not candidates:
                        break
                    for member in candidates:
                        if run["checked_this_slice"] >= budget or time.monotonic() - started >= config["time_slice"]:
                            break
                        if not self._valid(run, cancel):
                            raise InterruptedError()
                        aid = member["asset_id"]
                        try:
                            if aid not in lookup:
                                raise ValueError("照片已移除或内容变化，不属于当前范围快照，请重新搜索")
                            boxes = (regions or {}).get(aid)
                            if boxes is None:
                                hit = run.get("regions", {}).get(aid)
                                if isinstance(hit, str):
                                    try:
                                        hit = json.loads(hit)
                                    except ValueError:
                                        hit = None
                                if type(hit) is int and 0 <= hit < 4:
                                    hit = [[0, 0, .7, .7], [.3, 0, 1, .7], [0, .3, .7, 1], [.3, .3, 1, 1]][hit]
                                boxes = [hit] if box_valid(hit) else None
                            evidence = self.tools.inspect(lookup[aid], query, config, boxes)
                            errors = 0
                        except Exception as exc:
                            evidence = {"tool_error": str(exc)[:400], "conditions": [], "joint_status": "unknown"}
                            errors += 1
                        if not self._valid(run, cancel):
                            raise InterruptedError()
                        state = self._record(run, aid, evidence)
                        run["checked_this_slice"] += 1
                        if selected is not None:
                            run.setdefault("manual_done", []).append(aid)
                        self.store.action(run["id"], "inspect", asset_id=aid, outcome=state,
                            cache_hit=evidence.get("cache_hit", False), elapsed_ms=evidence.get("elapsed_ms"),
                            reason="集合覆盖" if is_all else "用户指定" if selected is not None else "前排或互补来源的条件缺口")
                        if aid in run["base"]:
                            self._reorder(run)
                        self._save(run)
                        if errors >= 2:
                            stop = "连续工具失败，已保存进度；检查模型状态后可重试失败项"
                            break
                    if errors >= 2:
                        break
                if is_all:
                    coverage = self.coverage(run["id"])
                    if coverage.get("unreviewed", 0):
                        stop = stop if stop != "本轮处理完成" else "本轮扫描暂停，可以继续剩余照片"
                    else:
                        stop = "指定范围本轮扫描完成；未知和失败项仍需复查，不代表保证找齐"
                elif errors >= 2:
                    pass
                elif budget == 0:
                    stop = "初步相关候选已返回，可手动核验或继续找"
                elif stop == "本轮处理完成":
                    stop = "本轮有限核验完成；可继续检查或恢复搜索"
                # One bounded exploratory expansion is allowed when checked
                # candidates have explicit conflicts. It does not prove absence,
                # relax a hard condition, or trigger an unbounded vision loop.
                counts = self.coverage(run["id"])
                if (not is_all and selected is None and errors == 0 and counts["conflict"] >= 2
                        and counts["supported"] == 0 and not run.get("automatic_expansion")
                        and time.monotonic() - started < config["time_slice"] and self._valid(run, cancel)):
                    run["stage"] = "补充候选与邻近事件线索"
                    run["automatic_expansion"] = True
                    self.store.action(run["id"], "expand", reason="已核对候选有明确冲突，有限扩大搜索；不判断目标不存在")
                    self._apply_recall(run, self.recall.search(assets, query, config, config["recovery_limit"], 1, cancel))
                    self._reorder(run)
                    stop = "已补充候选；可继续核验或选一张接近的照片"
        except InterruptedError:
            stop = "用户暂停或查询已更新"
        except Exception as exc:
            stop = "任务遇到错误，进度已保存：" + str(exc)[:400]
            run["warnings"].append(stop)
        finally:
            coverage = self.coverage(run["id"])
            unfinished = query["mode"] == "all" and coverage.get("unreviewed", 0) > 0
            run.update(status="paused" if unfinished or cancel.is_set() or stop.startswith(("任务遇到", "连续工具失败", "达到本轮")) else "complete",
                       stop_reason=stop, elapsed_ms=run.get("elapsed_ms", 0) + round((time.monotonic() - started) * 1000))
            run["revision"] += 1
            self.store.save(run)
            self.store.action(run["id"], "stopped", reason=stop, elapsed_ms=run["elapsed_ms"], coverage=coverage)

    def coverage(self, rid):
        with self.e.db() as con:
            counts = {r[0]: r[1] for r in con.execute("SELECT state,COUNT(*) FROM flow_members WHERE run_id=? GROUP BY state", (rid,))}
        total = sum(counts.values())
        return {**{key: 0 for key in ("unreviewed", "supported", "conflict", "unknown", "error", "excluded")}, **counts,
                "total": total, "processed": total - counts.get("unreviewed", 0) - counts.get("excluded", 0)}

    def pause(self, rid):
        with self.lock:
            run = self.store.get_run(rid)
            worker = self.workers.get(rid)
            if worker and worker[0].is_alive():
                worker[1].set()
                run.update(status="pausing", stop_reason="将在当前模型调用结束后暂停")
                self.store.save(run)
            return self.view(rid)

    def resume(self, rid, selected=None, regions=None, retry_failed=False):
        with self.lock:
            run = self.store.get_run(rid)
            if self.store.get_session(run["session_id"])["current_run"] != rid:
                raise ValueError("该任务已过期，请继续当前查询")
            if selected is not None:
                if not selected or len(selected) > 8 or len(self._members(run, ids=selected)) != len(set(selected)):
                    raise ValueError("请选当前任务范围内的 1–8 张照片")
                if set(selected) & set(run["query"]["excluded"]):
                    raise ValueError("指定照片已被你排除，请先重新开始搜索")
                run["manual_done"] = []
            if regions:
                if not set(regions) <= set(selected or []):
                    raise ValueError("局部必须对应本次指定的照片")
                if any(len(boxes) > 2 or not all(box_valid(box) for box in boxes) for boxes in regions.values()):
                    raise ValueError("每张照片最多指定两个有效局部")
            active = self.workers.get(rid)
            if active and active[0].is_alive():
                raise ValueError("当前任务正在执行或结束中，请稍后继续")
            if retry_failed:
                with self.e.db() as con:
                    con.execute("UPDATE flow_members SET state='unreviewed',evidence=NULL WHERE run_id=? AND state='error'", (rid,))
            self._launch(run, selected, regions)
            return self.view(rid)

    def view(self, rid, offset=0, limit=60, filter_by="all", expected_ranking=None):
        run = self.store.get_run(rid)
        if expected_ranking is not None and expected_ranking != run["revision"]:
            raise ValueError("排序已更新，请从第一页重新载入")
        sql = "SELECT * FROM flow_members WHERE run_id=? AND state!='excluded'"
        params = [rid]
        if run["query"]["mode"] != "all":
            sql += " AND candidate=1"
        if filter_by != "all":
            if filter_by not in {"supported", "conflict", "unknown", "error", "unreviewed"}:
                raise ValueError("未知结果筛选")
            sql += " AND state=?"
            params.append(filter_by)
        with self.e.db() as con:
            count = con.execute("SELECT COUNT(*) FROM (" + sql + ")", params).fetchone()[0]
            rows = con.execute(sql + " ORDER BY priority,ordinal LIMIT ? OFFSET ?", params + [limit, offset]).fetchall()
        results = []
        for row in rows:
            item = json.loads(row["metadata"])
            item.update(state=row["state"], evidence=json.loads(row["evidence"]) if row["evidence"] else None,
                        thumbnail="/api/assets/" + row["asset_id"] + "/thumbnail",
                        sources=run["sources"].get(row["asset_id"], []), rank=row["priority"] + 1)
            results.append(item)
        return {"id": rid, "session": run["query"], "status": run["status"], "stage": run["stage"],
                "revision": run["revision"], "results": results, "result_total": count,
                "candidate_count": len(run["base"]), "coverage": self.coverage(rid),
                "warnings": list(dict.fromkeys(run["warnings"]))[-12:], "stop_reason": run["stop_reason"],
                "elapsed_ms": run.get("elapsed_ms", 0), "offset": offset, "limit": limit,
                "first_preview_ms": run.get("first_preview_ms"),
                "config": run["config"]}
