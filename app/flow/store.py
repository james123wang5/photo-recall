"""Additive SQLite schema. All inference happens outside database transactions.

A run belongs to exactly one query revision. Replacing a session invalidates old
writers by compare-and-swap, even when an old model call cannot be interrupted.
"""
import json
import time
from .domain import DEFAULTS


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class FlowStore:
    def __init__(self, engine):
        self.e = engine
        with engine.db() as con:
            con.executescript("""
            CREATE TABLE IF NOT EXISTS flow_sessions (
              id TEXT PRIMARY KEY, library_id TEXT NOT NULL, revision INTEGER NOT NULL,
              current_run TEXT, payload TEXT NOT NULL, updated REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS flow_sessions_library ON flow_sessions(library_id,updated);
            CREATE TABLE IF NOT EXISTS flow_runs (
              id TEXT PRIMARY KEY, session_id TEXT NOT NULL, query_revision INTEGER NOT NULL,
              payload TEXT NOT NULL, updated REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS flow_actions (
              id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
              kind TEXT NOT NULL, payload TEXT NOT NULL, created REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS flow_actions_run ON flow_actions(run_id,id);
            CREATE TABLE IF NOT EXISTS flow_members (
              run_id TEXT NOT NULL, asset_id TEXT NOT NULL, fingerprint TEXT,
              metadata TEXT NOT NULL, ordinal INTEGER NOT NULL, priority INTEGER NOT NULL,
              candidate INTEGER NOT NULL DEFAULT 0, state TEXT NOT NULL DEFAULT 'unreviewed',
              evidence TEXT, PRIMARY KEY(run_id,asset_id));
            CREATE INDEX IF NOT EXISTS flow_members_order ON flow_members(run_id,priority,ordinal);
            CREATE INDEX IF NOT EXISTS flow_members_state ON flow_members(run_id,state);
            CREATE TABLE IF NOT EXISTS flow_evidence (
              cache_key TEXT PRIMARY KEY, asset_id TEXT NOT NULL, fingerprint TEXT NOT NULL,
              payload TEXT NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS flow_vectors (
              asset_id TEXT NOT NULL, fingerprint TEXT NOT NULL, version TEXT NOT NULL,
              region TEXT NOT NULL, vector BLOB NOT NULL, dimension INTEGER NOT NULL,
              PRIMARY KEY(asset_id,version,region));
            CREATE TABLE IF NOT EXISTS flow_index_state (
              asset_id TEXT NOT NULL, component TEXT NOT NULL, fingerprint TEXT,
              version TEXT, status TEXT NOT NULL, error TEXT, updated REAL NOT NULL,
              PRIMARY KEY(asset_id,component));
            CREATE TABLE IF NOT EXISTS flow_people (
              id TEXT PRIMARY KEY, name TEXT NOT NULL DEFAULT '');
            CREATE TABLE IF NOT EXISTS flow_faces (
              asset_id TEXT NOT NULL, fingerprint TEXT NOT NULL, face_id TEXT NOT NULL,
              person_id TEXT NOT NULL, vector BLOB NOT NULL, box TEXT NOT NULL, version TEXT NOT NULL,
              PRIMARY KEY(asset_id,face_id));
            CREATE TABLE IF NOT EXISTS flow_config (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL);
            """)
            # A process restart cannot imply a task has completed.
            for row in con.execute("SELECT id,payload FROM flow_runs").fetchall():
                run = json.loads(row["payload"])
                if run.get("status") in {"running", "queued", "pausing"}:
                    run.update(status="paused", stop_reason="服务重启，进度已保存，可以继续")
                    con.execute("UPDATE flow_runs SET payload=? WHERE id=?", (encode(run), row["id"]))

    def config(self):
        with self.e.db() as con:
            row = con.execute("SELECT payload FROM flow_config WHERE id=1").fetchone()
        return {**DEFAULTS, **(json.loads(row[0]) if row else {})}

    def get_session(self, sid):
        with self.e.db() as con:
            row = con.execute("SELECT payload FROM flow_sessions WHERE id=?", (sid,)).fetchone()
        if not row:
            raise ValueError("搜索会话不存在")
        return json.loads(row[0])

    def get_run(self, rid):
        with self.e.db() as con:
            row = con.execute("SELECT payload FROM flow_runs WHERE id=?", (rid,)).fetchone()
        if not row:
            raise ValueError("搜索任务不存在")
        return json.loads(row[0])

    def replace(self, session, run, expected_revision, assets):
        with self.e.db() as con:
            con.execute("BEGIN IMMEDIATE")
            old = con.execute("SELECT revision FROM flow_sessions WHERE id=?", (session["id"],)).fetchone()
            if (old and old[0] != expected_revision) or (not old and expected_revision != 0):
                raise ValueError("搜索状态已更新，请刷新后重试")
            con.execute("INSERT OR REPLACE INTO flow_sessions VALUES (?,?,?,?,?,?)", (
                session["id"], session["library_id"], session["revision"], run["id"], encode(session), time.time()))
            con.execute("INSERT INTO flow_runs VALUES (?,?,?,?,?)", (
                run["id"], session["id"], session["revision"], encode(run), time.time()))
            con.executemany("INSERT INTO flow_members(run_id,asset_id,fingerprint,metadata,ordinal,priority) VALUES (?,?,?,?,?,?)",
                [(run["id"], a["id"], a["fingerprint"], encode({k: a.get(k) for k in (
                    "id", "name", "path", "width", "height", "captured", "fingerprint", "size", "mtime_ns")}), i, 1000000 + i)
                 for i, a in enumerate(assets)])

    def save(self, run):
        with self.e.db() as con:
            con.execute("BEGIN IMMEDIATE")
            current = con.execute("SELECT current_run,revision FROM flow_sessions WHERE id=?", (run["session_id"],)).fetchone()
            if not current or current[0] != run["id"] or current[1] != run["query"]["revision"]:
                return False
            con.execute("UPDATE flow_runs SET payload=?,updated=? WHERE id=?", (encode(run), time.time(), run["id"]))
            # The parsed query and its revision become visible together.
            con.execute("UPDATE flow_sessions SET payload=?,updated=? WHERE id=?", (
                encode(run["query"]), time.time(), run["session_id"]))
        return True

    def action(self, rid, kind, **payload):
        with self.e.db() as con:
            con.execute("INSERT INTO flow_actions(run_id,kind,payload,created) VALUES (?,?,?,?)", (rid, kind, encode(payload), time.time()))

    def actions(self, rid, after=0):
        with self.e.db() as con:
            return [{"id": r["id"], "kind": r["kind"], "created": r["created"], **json.loads(r["payload"])}
                    for r in con.execute("SELECT * FROM flow_actions WHERE run_id=? AND id>? ORDER BY id LIMIT 200", (rid, after))]

    def cached(self, key):
        with self.e.db() as con:
            row = con.execute("SELECT payload FROM flow_evidence WHERE cache_key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def cache(self, key, asset, payload):
        with self.e.db() as con:
            con.execute("INSERT OR REPLACE INTO flow_evidence VALUES (?,?,?,?,?)", (
                key, asset["id"], asset["fingerprint"], encode(payload), time.time()))

    def index_state(self, asset, component, version, status, error=None):
        with self.e.db() as con:
            con.execute("INSERT OR REPLACE INTO flow_index_state VALUES (?,?,?,?,?,?,?)", (
                asset["id"], component, asset["fingerprint"], version, status, error, time.time()))
