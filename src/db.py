from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

from config import get_db_timeout_seconds
from session_parser import NormalizedEvent


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY,
  session_key TEXT UNIQUE NOT NULL,
  source_path TEXT NOT NULL,
  started_at TEXT,
  thread_updated_at TEXT,
  thread_name TEXT,
  cwd TEXT,
  project_name TEXT,
  repo_name TEXT,
  model TEXT,
  raw_format TEXT,
  source_mtime_ns INTEGER,
  source_size INTEGER,
  source_sha256 TEXT,
  indexed_event_count INTEGER DEFAULT 0,
  last_event_hash TEXT,
  next_chunk_index INTEGER DEFAULT 0,
  tail_events_json TEXT DEFAULT '[]',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  last_indexed_at TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY,
  session_id INTEGER NOT NULL,
  chunk_index INTEGER NOT NULL,
  role TEXT,
  event_type TEXT,
  content TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  token_estimate INTEGER,
  start_event_index INTEGER,
  end_event_index INTEGER,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(session_id) REFERENCES sessions(id),
  UNIQUE(session_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS embeddings (
  chunk_id INTEGER PRIMARY KEY,
  embedding_json TEXT NOT NULL,
  embedding_dim INTEGER NOT NULL,
  FOREIGN KEY(chunk_id) REFERENCES chunks(id)
);

CREATE TABLE IF NOT EXISTS session_summaries (
  session_id INTEGER PRIMARY KEY,
  summary TEXT NOT NULL,
  embedding_json TEXT,
  summary_event_count INTEGER DEFAULT 0,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_name);
CREATE INDEX IF NOT EXISTS idx_sessions_repo ON sessions(repo_name);
CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_id);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);
"""


class MemoryDB:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, timeout=get_db_timeout_seconds())
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def upsert_session(
        self,
        *,
        session_key: str,
        source_path: str,
        started_at: str | None,
        thread_updated_at: str | None,
        thread_name: str | None,
        cwd: str | None,
        project_name: str | None,
        repo_name: str | None,
        model: str | None,
        raw_format: str | None,
        source_mtime_ns: int | None,
        source_size: int | None,
        source_sha256: str | None,
    ) -> int:
        self.conn.execute(
            """
            INSERT INTO sessions (
              session_key,
              source_path,
              started_at,
              thread_updated_at,
              thread_name,
              cwd,
              project_name,
              repo_name,
              model,
              raw_format,
              source_mtime_ns,
              source_size,
              source_sha256,
              updated_at,
              last_indexed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(session_key) DO UPDATE SET
              source_path=excluded.source_path,
              started_at=excluded.started_at,
              thread_updated_at=excluded.thread_updated_at,
              thread_name=excluded.thread_name,
              cwd=excluded.cwd,
              project_name=excluded.project_name,
              repo_name=excluded.repo_name,
              model=excluded.model,
              raw_format=excluded.raw_format,
              source_mtime_ns=excluded.source_mtime_ns,
              source_size=excluded.source_size,
              source_sha256=excluded.source_sha256,
              updated_at=CURRENT_TIMESTAMP,
              last_indexed_at=CURRENT_TIMESTAMP
            """,
            (
                session_key,
                source_path,
                started_at,
                thread_updated_at,
                thread_name,
                cwd,
                project_name,
                repo_name,
                model,
                raw_format,
                source_mtime_ns,
                source_size,
                source_sha256,
            ),
        )

        row = self.conn.execute(
            "SELECT id FROM sessions WHERE session_key = ?",
            (session_key,),
        ).fetchone()
        assert row is not None
        return int(row["id"])

    def get_session_state(self, session_key: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT
              id,
              session_key,
              source_mtime_ns,
              source_size,
              source_sha256,
              indexed_event_count,
              last_event_hash,
              next_chunk_index,
              tail_events_json
            FROM sessions
            WHERE session_key = ?
            """,
            (session_key,),
        ).fetchone()

        return dict(row) if row else None

    def get_session_summary_state(self, session_id: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT summary, summary_event_count
            FROM session_summaries
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()

        return dict(row) if row else None

    def upsert_session_summary(
        self,
        session_id: int,
        summary: str,
        summary_event_count: int,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO session_summaries (session_id, summary, summary_event_count, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(session_id) DO UPDATE SET
              summary=excluded.summary,
              summary_event_count=excluded.summary_event_count,
              updated_at=CURRENT_TIMESTAMP
            """,
            (session_id, summary, summary_event_count),
        )

    def upsert_session_summary_embedding(self, session_id: int, embedding: list[float]) -> None:
        payload = json.dumps(embedding)

        self.conn.execute(
            """
            UPDATE session_summaries
            SET embedding_json = ?, updated_at = CURRENT_TIMESTAMP
            WHERE session_id = ?
            """,
            (payload, session_id),
        )

    def update_session_progress(
        self,
        *,
        session_id: int,
        indexed_event_count: int,
        last_event_hash: str | None,
        next_chunk_index: int,
        tail_events: list[NormalizedEvent],
        source_mtime_ns: int | None,
        source_size: int | None,
        source_sha256: str | None,
    ) -> None:
        tail_payload = json.dumps([event.to_dict() for event in tail_events])

        self.conn.execute(
            """
            UPDATE sessions
            SET
              indexed_event_count = ?,
              last_event_hash = ?,
              next_chunk_index = ?,
              tail_events_json = ?,
              source_mtime_ns = ?,
              source_size = ?,
              source_sha256 = ?,
              updated_at = CURRENT_TIMESTAMP,
              last_indexed_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                indexed_event_count,
                last_event_hash,
                next_chunk_index,
                tail_payload,
                source_mtime_ns,
                source_size,
                source_sha256,
                session_id,
            ),
        )

    def upsert_chunk(
        self,
        *,
        session_id: int,
        chunk_index: int,
        role: str,
        event_type: str,
        content: str,
        token_estimate: int,
        start_event_index: int,
        end_event_index: int,
    ) -> tuple[int, bool]:
        content_hash = self.hash_chunk(
            session_id=session_id,
            chunk_index=chunk_index,
            content=content,
        )

        existing = self.conn.execute(
            """
            SELECT id, content_hash
            FROM chunks
            WHERE session_id = ? AND chunk_index = ?
            """,
            (session_id, chunk_index),
        ).fetchone()

        if existing and existing["content_hash"] == content_hash:
            return int(existing["id"]), False

        self.conn.execute(
            """
            INSERT INTO chunks (
              session_id,
              chunk_index,
              role,
              event_type,
              content,
              content_hash,
              token_estimate,
              start_event_index,
              end_event_index
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, chunk_index) DO UPDATE SET
              role=excluded.role,
              event_type=excluded.event_type,
              content=excluded.content,
              content_hash=excluded.content_hash,
              token_estimate=excluded.token_estimate,
              start_event_index=excluded.start_event_index,
              end_event_index=excluded.end_event_index
            """,
            (
                session_id,
                chunk_index,
                role,
                event_type,
                content,
                content_hash,
                token_estimate,
                start_event_index,
                end_event_index,
            ),
        )

        row = self.conn.execute(
            """
            SELECT id
            FROM chunks
            WHERE session_id = ? AND chunk_index = ?
            """,
            (session_id, chunk_index),
        ).fetchone()

        assert row is not None
        chunk_id = int(row["id"])

        if existing:
            self.conn.execute(
                "DELETE FROM embeddings WHERE chunk_id = ?",
                (chunk_id,),
            )

        return chunk_id, True

    def upsert_embedding(self, *, chunk_id: int, embedding: list[float]) -> None:
        payload = json.dumps(embedding)

        self.conn.execute(
            """
            INSERT INTO embeddings (chunk_id, embedding_json, embedding_dim)
            VALUES (?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
              embedding_json=excluded.embedding_json,
              embedding_dim=excluded.embedding_dim
            """,
            (chunk_id, payload, len(embedding)),
        )

    def search_session_summaries(
        self,
        query_embedding: list[float],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT
              s.session_key,
              s.thread_name,
              s.project_name,
              s.repo_name,
              s.started_at,
              ss.summary,
              ss.embedding_json
            FROM session_summaries ss
            JOIN sessions s ON s.id = ss.session_id
            WHERE ss.embedding_json IS NOT NULL
            """
        ).fetchall()

        scored: list[dict[str, Any]] = []

        for row in rows:
            embedding = json.loads(row["embedding_json"])
            score = cosine_similarity(query_embedding, embedding)

            scored.append(
                {
                    "session_key": row["session_key"],
                    "thread_name": row["thread_name"],
                    "project_name": row["project_name"],
                    "repo_name": row["repo_name"],
                    "started_at": row["started_at"],
                    "summary": row["summary"],
                    "score": score,
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def search(
        self,
        *,
        query_embedding: list[float],
        limit: int = 5,
        project_name: str | None = None,
        repo_name: str | None = None,
        session_key: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = """
        SELECT
          c.id AS chunk_id,
          c.chunk_index,
          c.role,
          c.event_type,
          c.content,
          c.start_event_index,
          c.end_event_index,
          s.session_key,
          s.thread_name,
          s.project_name,
          s.repo_name,
          s.started_at,
          s.cwd,
          e.embedding_json
        FROM chunks c
        JOIN sessions s ON s.id = c.session_id
        JOIN embeddings e ON e.chunk_id = c.id
        WHERE 1=1
        """
        params: list[Any] = []

        if project_name:
            sql += " AND s.project_name = ?"
            params.append(project_name)

        if repo_name:
            sql += " AND s.repo_name = ?"
            params.append(repo_name)

        if session_key:
            sql += " AND s.session_key = ?"
            params.append(session_key)

        rows = self.conn.execute(sql, params).fetchall()

        scored: list[dict[str, Any]] = []

        for row in rows:
            embedding = json.loads(row["embedding_json"])
            score = cosine_similarity(query_embedding, embedding)

            scored.append(
                {
                    "chunk_id": int(row["chunk_id"]),
                    "session_key": row["session_key"],
                    "thread_name": row["thread_name"],
                    "project_name": row["project_name"],
                    "repo_name": row["repo_name"],
                    "started_at": row["started_at"],
                    "cwd": row["cwd"],
                    "chunk_index": int(row["chunk_index"]),
                    "start_event_index": row["start_event_index"],
                    "end_event_index": row["end_event_index"],
                    "role": row["role"],
                    "event_type": row["event_type"],
                    "content": row["content"],
                    "score": score,
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def fetch_chunk(self, chunk_id: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT
              c.id AS chunk_id,
              c.chunk_index,
              c.role,
              c.event_type,
              c.content,
              c.start_event_index,
              c.end_event_index,
              s.session_key,
              s.thread_name,
              s.project_name,
              s.repo_name,
              s.started_at,
              s.cwd,
              s.source_path
            FROM chunks c
            JOIN sessions s ON s.id = c.session_id
            WHERE c.id = ?
            """,
            (chunk_id,),
        ).fetchone()

        return dict(row) if row else None

    def list_recent_sessions(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT
              session_key,
              thread_name,
              started_at,
              thread_updated_at,
              project_name,
              repo_name,
              cwd,
              source_path,
              indexed_event_count,
              updated_at,
              last_indexed_at
            FROM sessions
            ORDER BY COALESCE(started_at, thread_updated_at, updated_at) DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return [dict(r) for r in rows]

    def commit(self) -> None:
        self.conn.commit()

    @staticmethod
    def hash_chunk(*, session_id: int, chunk_index: int, content: str) -> str:
        payload = f"{session_id}:{chunk_index}:{content}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Embedding dimension mismatch: {len(a)} != {len(b)}")

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)
