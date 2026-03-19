from __future__ import annotations

from dataclasses import dataclass

from session_parser import ParsedSession, NormalizedEvent


@dataclass
class Chunk:
    chunk_index: int
    role: str
    event_type: str
    content: str
    token_estimate: int
    start_event_index: int
    end_event_index: int


MAX_CHARS = 3500
TARGET_CHARS = 1800


def build_chunks(session: ParsedSession) -> list[Chunk]:
    chunks, tail_events = build_incremental_chunks(
        session=session,
        new_events=session.events,
        start_chunk_index=0,
        seed_events=None,
    )

    if tail_events:
        chunks.append(_finalize_chunk(session, len(chunks), tail_events))

    return chunks


def build_incremental_chunks(
    session: ParsedSession,
    new_events: list[NormalizedEvent],
    start_chunk_index: int = 0,
    seed_events: list[NormalizedEvent] | None = None,
) -> tuple[list[Chunk], list[NormalizedEvent]]:
    chunks: list[Chunk] = []

    current_events: list[NormalizedEvent] = list(seed_events or [])
    current_chars = sum(len(_render_event(session, event)) for event in current_events)
    next_chunk_index = start_chunk_index

    def flush_current() -> None:
        nonlocal current_events, current_chars, next_chunk_index
        if not current_events:
            return
        chunks.append(_finalize_chunk(session, next_chunk_index, current_events))
        next_chunk_index += 1
        current_events = []
        current_chars = 0

    for event in new_events:
        event_text = _render_event(session, event)
        event_len = len(event_text)

        if event_len > MAX_CHARS:
            flush_current()
            split_chunks = _split_long_event(session, event, next_chunk_index)
            chunks.extend(split_chunks)
            next_chunk_index += len(split_chunks)
            continue

        if current_events and current_chars + event_len > MAX_CHARS:
            flush_current()

        current_events.append(event)
        current_chars += event_len

        if current_chars >= TARGET_CHARS:
            flush_current()

    tail_events = current_events
    return chunks, tail_events


def _finalize_chunk(
    session: ParsedSession,
    chunk_index: int,
    events: list[NormalizedEvent],
) -> Chunk:
    content = "\n\n".join(_render_event(session, event) for event in events).strip()
    role = events[0].role if len({e.role for e in events}) == 1 else "mixed"
    event_type = events[0].event_type if len({e.event_type for e in events}) == 1 else "mixed"

    return Chunk(
        chunk_index=chunk_index,
        role=role,
        event_type=event_type,
        content=content,
        token_estimate=max(1, len(content) // 4),
        start_event_index=events[0].index,
        end_event_index=events[-1].index,
    )


def _split_long_event(
    session: ParsedSession,
    event: NormalizedEvent,
    chunk_index_start: int,
) -> list[Chunk]:
    content = _render_event(session, event)
    pieces: list[str] = []

    while content:
        if len(content) <= MAX_CHARS:
            pieces.append(content)
            break

        split_at = content.rfind("\n", 0, MAX_CHARS)
        if split_at < MAX_CHARS // 2:
            split_at = content.rfind(" ", 0, MAX_CHARS)
        if split_at < MAX_CHARS // 2:
            split_at = MAX_CHARS

        pieces.append(content[:split_at].strip())
        content = content[split_at:].strip()

    chunks: list[Chunk] = []

    for offset, piece in enumerate(pieces):
        chunks.append(
            Chunk(
                chunk_index=chunk_index_start + offset,
                role=event.role,
                event_type=event.event_type,
                content=piece,
                token_estimate=max(1, len(piece) // 4),
                start_event_index=event.index,
                end_event_index=event.index,
            )
        )

    return chunks


def _render_event(session: ParsedSession, event: NormalizedEvent) -> str:
    content = event.content.strip()

    if len(content) > MAX_CHARS:
        content = content[:MAX_CHARS]

    lines = [
        f"Session: {session.session_key}",
        f"Thread: {session.thread_name or 'unknown'}",
        f"Project: {session.project_name or 'unknown'}",
        f"Repo: {session.repo_name or 'unknown'}",
    ]

    if session.started_at:
        lines.append(f"Started: {session.started_at}")

    if session.updated_at:
        lines.append(f"Updated: {session.updated_at}")

    if session.cwd:
        lines.append(f"CWD: {session.cwd}")

    if event.timestamp:
        lines.append(f"Event Time: {event.timestamp}")

    lines.extend(
        [
            f"Turn: {event.index}",
            f"Role: {event.role}",
            f"Event: {event.event_type}",
            "Content:",
            content,
        ]
    )

    return "\n".join(lines).strip()