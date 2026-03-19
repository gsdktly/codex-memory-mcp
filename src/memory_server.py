from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from config import APP_NAME, get_db_path, setup_logging
from db import MemoryDB
from embeddings import Embedder

logger = setup_logging()
DB_PATH = get_db_path()

mcp = FastMCP(APP_NAME)


def _get_db() -> MemoryDB:
    return MemoryDB(DB_PATH)


def _get_embedder() -> Embedder:
    return Embedder()


@mcp.tool()
def search_codex_sessions(
    query: str,
    limit: int = 5,
    project_name: str | None = None,
    repo_name: str | None = None,
    session_key: str | None = None,
) -> str:
    """Search the local Codex memory index for relevant prior session chunks."""

    db = _get_db()

    try:

        query_embedding = _get_embedder().embed_text(query)

        rows = db.search(
            query_embedding=query_embedding,
            limit=max(1, min(limit, 20)),
            project_name=project_name,
            repo_name=repo_name,
            session_key=session_key,
        )

        if not rows:
            return "No matching session chunks found."

        blocks: list[str] = []

        for row in rows:

            preview = row["content"]

            if len(preview) > 1200:
                preview = preview[:1200].rstrip() + "\n..."

            blocks.append(
                "\n".join(
                    [
                        f"chunk_id: {row['chunk_id']}",
                        f"score: {row['score']:.4f}",
                        f"session_key: {row['session_key']}",
                        f"project_name: {row['project_name']}",
                        f"repo_name: {row['repo_name']}",
                        f"started_at: {row['started_at']}",
                        f"role: {row['role']}",
                        f"event_type: {row['event_type']}",
                        "content:",
                        preview,
                    ]
                )
            )

        return "\n\n---\n\n".join(blocks)

    finally:
        db.close()


@mcp.tool()
def fetch_session_chunk(chunk_id: int) -> str:
    """Fetch the full text and metadata for a specific chunk."""

    db = _get_db()

    try:

        row = db.fetch_chunk(chunk_id)

        if not row:
            return f"No chunk found for chunk_id={chunk_id}."

        return "\n".join(
            [
                f"chunk_id: {row['chunk_id']}",
                f"session_key: {row['session_key']}",
                f"project_name: {row['project_name']}",
                f"repo_name: {row['repo_name']}",
                f"started_at: {row['started_at']}",
                f"cwd: {row['cwd']}",
                f"source_path: {row['source_path']}",
                f"role: {row['role']}",
                f"event_type: {row['event_type']}",
                f"chunk_index: {row['chunk_index']}",
                "content:",
                row["content"],
            ]
        )

    finally:
        db.close()


@mcp.tool()
def list_recent_sessions(limit: int = 10) -> str:
    """List recently indexed Codex sessions."""

    db = _get_db()

    try:

        rows = db.list_recent_sessions(limit=max(1, min(limit, 50)))

        if not rows:
            return "No sessions indexed yet."

        return "\n\n".join(
            "\n".join(
                [
                    f"session_key: {row['session_key']}",
                    f"started_at: {row['started_at']}",
                    f"project_name: {row['project_name']}",
                    f"repo_name: {row['repo_name']}",
                    f"cwd: {row['cwd']}",
                    f"source_path: {row['source_path']}",
                    f"updated_at: {row['updated_at']}",
                ]
            )
            for row in rows
        )

    finally:
        db.close()


@mcp.tool()
def search_session_summaries(query: str, limit: int = 5) -> str:
    """Search semantic summaries of past Codex sessions."""

    db = _get_db()

    try:

        query_embedding = _get_embedder().embed_text(query)

        rows = db.search_session_summaries(
            query_embedding=query_embedding,
            limit=max(1, min(limit, 20)),
        )

        if not rows:
            return "No matching session summaries found."

        blocks: list[str] = []

        for row in rows:

            summary = row["summary"]

            if len(summary) > 1200:
                summary = summary[:1200].rstrip() + "\n..."

            blocks.append(
                "\n".join(
                    [
                        f"score: {row['score']:.4f}",
                        f"session_key: {row['session_key']}",
                        f"project_name: {row['project_name']}",
                        f"repo_name: {row['repo_name']}",
                        f"started_at: {row['started_at']}",
                        "summary:",
                        summary,
                    ]
                )
            )

        return "\n\n---\n\n".join(blocks)

    finally:
        db.close()


if __name__ == "__main__":
    logger.info("Starting %s with db=%s", APP_NAME, DB_PATH)
    mcp.run(transport="stdio")
