"""
Memory (Step 5 of the framework).

Two layers, mapped to the framework's taxonomy:

  - Working memory (in-RAM)   : current conversation + notes   = "cache memory"
  - Persistent cache (SQLite) : URL -> cleaned text            = "file system memory"

The persistent cache is critical: web fetches are slow and flaky, and
re-running the same query (during eval, debugging, or just iteration)
should be cheap. SQLite is used because it's in the Python stdlib, no
server required.

We deliberately do NOT use a vector database for this agent. The full
research is short-lived and fits in context. Vector search becomes
relevant in use case #5 (Document Q&A), not here.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_DB = Path.home() / ".research_agent" / "cache.db"


@dataclass
class WorkingMemory:
    """Per-run scratchpad. Lives only for one research session."""
    topic: str = ""
    messages: list[dict] = field(default_factory=list)
    notes: list[dict] = field(default_factory=list)
    fetched_urls: set[str] = field(default_factory=set)
    searched_queries: set[str] = field(default_factory=set)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    iterations: int = 0
    trace: list[dict] = field(default_factory=list)

    def log_step(self, kind: str, payload: dict) -> None:
        """Append to the trace - this is your Step 7 observability."""
        self.trace.append({"ts": time.time(), "kind": kind, **payload})

    def cited_sources(self) -> list[str]:
        """Distinct source URLs across all notes, in order first seen."""
        seen, out = set(), []
        for n in self.notes:
            u = n["source_url"]
            if u not in seen:
                seen.add(u)
                out.append(u)
        return out


class PersistentCache:
    """SQLite-backed page cache. URLs -> cleaned text + metadata."""

    def __init__(self, db_path: Path | str = DEFAULT_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS pages (
                    url_hash TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    content TEXT NOT NULL,
                    fetched_at REAL NOT NULL
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    completed_at REAL,
                    iterations INTEGER,
                    cost_usd REAL,
                    summary TEXT,
                    trace_json TEXT
                )
            """)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _hash(url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()[:16]

    def get_page(self, url: str, max_age_seconds: int = 86400) -> str | None:
        """Return cached content if fresh enough, else None."""
        with self._conn() as c:
            row = c.execute(
                "SELECT content, fetched_at FROM pages WHERE url_hash = ?",
                (self._hash(url),),
            ).fetchone()
        if not row:
            return None
        content, fetched_at = row
        if time.time() - fetched_at > max_age_seconds:
            return None
        return content

    def put_page(self, url: str, content: str) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO pages (url_hash, url, content, fetched_at) VALUES (?, ?, ?, ?)",
                (self._hash(url), url, content, time.time()),
            )

    def record_run(self, mem: WorkingMemory, summary: str) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO runs (topic, started_at, completed_at, iterations, cost_usd, summary, trace_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    mem.topic,
                    mem.trace[0]["ts"] if mem.trace else time.time(),
                    time.time(),
                    mem.iterations,
                    mem.total_cost_usd,
                    summary,
                    json.dumps(mem.trace, default=str),
                ),
            )
            return cur.lastrowid
