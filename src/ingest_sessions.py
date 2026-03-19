from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from session_summarizer import summarize_session, update_session_summary
from chunker import build_incremental_chunks
from db import MemoryDB
from embeddings import Embedder
from session_parser import (
    NormalizedEvent,
    SessionParseError,
    load_session_index,
    parse_session_file,
)



def discover_session_files(root: Path) -> list[Path]:
    files: list[Path] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if path.name.startswith("."):
            continue

        if path.name.startswith("rollout-"):
            files.append(path)

    return sorted(files)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def file_fingerprint(path: Path) -> tuple[int, int, str]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size, file_sha256(path)


def load_tail_events(tail_events_json: str | None) -> list[NormalizedEvent]:
    if not tail_events_json:
        return []

    try:
        raw_items = __import__("json").loads(tail_events_json)
    except Exception:
        return []

    if not isinstance(raw_items, list):
        return []

    tail_events: list[NormalizedEvent] = []

    for item in raw_items:
        if isinstance(item, dict):
            try:
                tail_events.append(NormalizedEvent.from_dict(item))
            except Exception:
                continue

    return tail_events


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Incrementally ingest Codex session files into a local SQLite memory index."
    )

    parser.add_argument(
        "sessions_dir",
        help="Directory to scan, e.g. ~/.codex/sessions",
    )

    parser.add_argument(
        "--db",
        default="./data/codex_memory.sqlite",
        help="SQLite database path",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of session files to ingest",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )

    args = parser.parse_args()

    root = Path(args.sessions_dir).expanduser().resolve()

    if not root.exists() or not root.is_dir():
        print(f"Sessions directory not found: {root}", file=sys.stderr)
        return 1

    codex_home = root.parent
    session_index_path = codex_home / "session_index.jsonl"
    session_index = load_session_index(session_index_path)

    if args.verbose:
        print(f"[index] loaded {len(session_index)} session index records")

    files = discover_session_files(root)

    if args.limit is not None:
        files = files[: args.limit]

    if args.verbose:
        print(f"[discover] found {len(files)} session files")

    db = MemoryDB(args.db)
    embedder = Embedder()

    parsed_count = 0
    chunk_count = 0
    embedded_count = 0
    failed_count = 0
    skipped_unchanged_count = 0

    MAX_BATCH_SIZE = 8
    MAX_BATCH_CHARS = 24000
    MAX_CHUNK_CHARS = 6000
    SKIP_CHUNK_CHARS = 50000

    try:
        for path in files:
            try:
                if args.verbose:
                    print(f"[parse] {path}")

                parsed = parse_session_file(
                    path,
                    session_index=session_index,
                )

                source_mtime_ns, source_size, source_sha256 = file_fingerprint(path)

                session_id = db.upsert_session(
                    session_key=parsed.session_key,
                    source_path=parsed.source_path,
                    started_at=parsed.started_at,
                    thread_updated_at=parsed.updated_at,
                    thread_name=parsed.thread_name,
                    cwd=parsed.cwd,
                    project_name=parsed.project_name,
                    repo_name=parsed.repo_name,
                    model=parsed.model,
                    raw_format=parsed.raw_format,
                    source_mtime_ns=source_mtime_ns,
                    source_size=source_size,
                    source_sha256=source_sha256,
                )

                state = db.get_session_state(parsed.session_key)
                if state is None:
                    raise RuntimeError(f"Failed to load session state for {parsed.session_key}")

                indexed_event_count = int(state["indexed_event_count"] or 0)
                last_event_hash = state["last_event_hash"]
                next_chunk_index = int(state["next_chunk_index"] or 0)
                tail_events = load_tail_events(state["tail_events_json"])

                if indexed_event_count > len(parsed.events):
                    raise RuntimeError(
                        f"Non-append mutation detected for {path.name}: "
                        f"stored indexed_event_count={indexed_event_count} but parsed only {len(parsed.events)} events. "
                        f"Run a full rebuild for this DB."
                    )

                if indexed_event_count > 0:
                    actual_previous_hash = parsed.events[indexed_event_count - 1].event_hash
                    if last_event_hash and actual_previous_hash != last_event_hash:
                        raise RuntimeError(
                            f"Non-append mutation detected for {path.name}: "
                            f"stored last_event_hash does not match parsed prefix. "
                            f"Run a full rebuild for this DB."
                        )

                new_events = parsed.events[indexed_event_count:]

                if args.verbose:
                    print(
                        f"[events] {path.name} total_events={len(parsed.events)} "
                        f"new_events={len(new_events)} thread={parsed.thread_name}"
                    )

                if not new_events:
                    skipped_unchanged_count += 1
                    if args.verbose:
                        print(f"[skip-no-new-events] {path.name}")
                    parsed_count += 1
                    db.commit()
                    continue

                summary_state = db.get_session_summary_state(session_id)

                if summary_state and int(summary_state["summary_event_count"] or 0) == indexed_event_count:
                    summary = update_session_summary(
                        previous_summary=str(summary_state["summary"]),
                        new_events=new_events,
                    )
                    summary_event_count = indexed_event_count + len(new_events)
                else:
                    summary = summarize_session(parsed.events)
                    summary_event_count = len(parsed.events)

                if args.verbose:
                    print(f"[summary] {path.name} updated")

                db.upsert_session_summary(
                    session_id=session_id,
                    summary=summary,
                    summary_event_count=summary_event_count,
                )

                summary_embedding = embedder.embed_text(summary)
                db.upsert_session_summary_embedding(
                    session_id=session_id,
                    embedding=summary_embedding,
                )

                chunks, new_tail_events = build_incremental_chunks(
                    session=parsed,
                    new_events=new_events,
                    start_chunk_index=next_chunk_index,
                    seed_events=tail_events,
                )

                if args.verbose:
                    print(
                        f"[chunks] {path.name} finalized_chunks={len(chunks)} "
                        f"tail_events={len(new_tail_events)}"
                    )

                safe_chunks = []

                for chunk in chunks:
                    content = chunk.content.strip()

                    if not content:
                        continue

                    if len(content) > SKIP_CHUNK_CHARS:
                        if args.verbose:
                            print(
                                f"[skip-chunk] {path.name} "
                                f"chunk={chunk.chunk_index} chars={len(content)}"
                            )
                        continue

                    if len(content) > MAX_CHUNK_CHARS:
                        if args.verbose:
                            print(
                                f"[truncate-chunk] {path.name} "
                                f"chunk={chunk.chunk_index} {len(content)} -> {MAX_CHUNK_CHARS}"
                            )
                        content = content[:MAX_CHUNK_CHARS]

                    chunk.content = content
                    chunk.token_estimate = max(1, len(content) // 4)
                    safe_chunks.append(chunk)

                texts_to_embed: list[str] = []
                chunk_ids_to_embed: list[int] = []

                for chunk in safe_chunks:
                    chunk_id, changed = db.upsert_chunk(
                        session_id=session_id,
                        chunk_index=chunk.chunk_index,
                        role=chunk.role,
                        event_type=chunk.event_type,
                        content=chunk.content,
                        token_estimate=chunk.token_estimate,
                        start_event_index=chunk.start_event_index,
                        end_event_index=chunk.end_event_index,
                    )

                    chunk_count += 1

                    if changed:
                        texts_to_embed.append(chunk.content)
                        chunk_ids_to_embed.append(chunk_id)

                if texts_to_embed and args.verbose:
                    print(
                        f"[debug] {path.name} "
                        f"chunks_to_embed={len(texts_to_embed)} "
                        f"max_chars={max(len(t) for t in texts_to_embed)} "
                        f"total_chars={sum(len(t) for t in texts_to_embed)}"
                    )

                if texts_to_embed:
                    batch_texts: list[str] = []
                    batch_chunk_ids: list[int] = []
                    batch_chars = 0

                    for chunk_id, text in zip(chunk_ids_to_embed, texts_to_embed):
                        text = text.strip()

                        if not text:
                            continue

                        text_len = len(text)

                        should_flush = (
                            batch_texts
                            and (
                                len(batch_texts) >= MAX_BATCH_SIZE
                                or batch_chars + text_len > MAX_BATCH_CHARS
                            )
                        )

                        if should_flush:
                            if args.verbose:
                                print(
                                    f"[embed-batch] {path.name} "
                                    f"batch_size={len(batch_texts)} batch_chars={batch_chars}"
                                )

                            embeddings = embedder.embed_many(batch_texts)

                            for out_chunk_id, embedding in zip(batch_chunk_ids, embeddings):
                                db.upsert_embedding(
                                    chunk_id=out_chunk_id,
                                    embedding=embedding,
                                )
                                embedded_count += 1

                            batch_texts = []
                            batch_chunk_ids = []
                            batch_chars = 0

                        batch_texts.append(text)
                        batch_chunk_ids.append(chunk_id)
                        batch_chars += text_len

                    if batch_texts:
                        if args.verbose:
                            print(
                                f"[embed-batch] {path.name} "
                                f"batch_size={len(batch_texts)} batch_chars={batch_chars}"
                            )

                        embeddings = embedder.embed_many(batch_texts)

                        for out_chunk_id, embedding in zip(batch_chunk_ids, embeddings):
                            db.upsert_embedding(
                                chunk_id=out_chunk_id,
                                embedding=embedding,
                            )
                            embedded_count += 1

                db.update_session_progress(
                    session_id=session_id,
                    indexed_event_count=len(parsed.events),
                    last_event_hash=parsed.events[-1].event_hash,
                    next_chunk_index=next_chunk_index + len(safe_chunks),
                    tail_events=new_tail_events,
                    source_mtime_ns=source_mtime_ns,
                    source_size=source_size,
                    source_sha256=source_sha256,
                )

                parsed_count += 1
                db.commit()

            except SessionParseError as exc:
                failed_count += 1
                print(f"[skip] {path}: {exc}", file=sys.stderr)

            except Exception as exc:
                failed_count += 1
                print(f"[error] {path}: {exc}", file=sys.stderr)

        print(
            f"Ingested sessions={parsed_count} "
            f"chunks_seen={chunk_count} "
            f"newly_embedded={embedded_count} "
            f"failed={failed_count} "
            f"skipped_unchanged={skipped_unchanged_count} "
            f"db={args.db}"
        )

        return 0

    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
