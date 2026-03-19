from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class NormalizedEvent:
    index: int
    role: str
    event_type: str
    content: str
    timestamp: str | None = None
    event_hash: str | None = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "role": self.role,
            "event_type": self.event_type,
            "content": self.content,
            "timestamp": self.timestamp,
            "event_hash": self.event_hash,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NormalizedEvent":
        return cls(
            index=int(data["index"]),
            role=str(data["role"]),
            event_type=str(data["event_type"]),
            content=str(data["content"]),
            timestamp=data.get("timestamp"),
            event_hash=data.get("event_hash"),
        )


@dataclass
class ParsedSession:
    session_key: str
    source_path: str
    started_at: str | None
    updated_at: str | None
    thread_name: str | None
    cwd: str | None
    project_name: str | None
    repo_name: str | None
    model: str | None
    events: list[NormalizedEvent]
    raw_format: str


class SessionParseError(RuntimeError):
    pass


def load_session_index(path: str | Path) -> dict[str, dict[str, str]]:
    p = Path(path).expanduser()

    if not p.exists():
        return {}

    index: dict[str, dict[str, str]] = {}

    with p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            session_id = row.get("id")
            if not session_id:
                continue

            index[session_id] = {
                "thread_name": row.get("thread_name"),
                "updated_at": row.get("updated_at"),
            }

    return index


def parse_session_file(
    path: str | Path,
    session_index: dict[str, dict[str, str]] | None = None,
) -> ParsedSession:
    p = Path(path)

    raw_lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    session_index = session_index or {}

    session_id: str | None = None
    started_at: str | None = None
    cwd: str | None = None
    model: str | None = None
    project_name: str | None = None
    repo_name: str | None = None

    events: list[NormalizedEvent] = []
    event_idx = 0

    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line:
            continue

        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        item_type = item.get("type")
        payload = item.get("payload") or {}
        timestamp = item.get("timestamp")

        if item_type == "session_meta":
            session_id = payload.get("id") or session_id
            started_at = payload.get("timestamp") or started_at
            cwd = payload.get("cwd") or cwd
            model = payload.get("model_provider") or model

            if cwd:
                project_name = Path(cwd).name
                repo_name = Path(cwd).name

            continue

        if item_type == "turn_context":
            cwd = payload.get("cwd") or cwd
            model = payload.get("model") or model

            if cwd:
                project_name = Path(cwd).name
                repo_name = Path(cwd).name

            continue

        if item_type == "response_item":
            payload_type = payload.get("type")

            if payload_type == "message":
                role = payload.get("role", "unknown")
                text = _extract_message_text(payload.get("content", []))

                if text:
                    events.append(
                        _make_event(
                            index=event_idx,
                            role=role,
                            event_type="message",
                            content=text,
                            timestamp=timestamp,
                        )
                    )
                    event_idx += 1

            elif payload_type == "function_call":
                summary = _summarize_function_call(payload)

                if summary:
                    events.append(
                        _make_event(
                            index=event_idx,
                            role="assistant",
                            event_type="function_call",
                            content=summary,
                            timestamp=timestamp,
                        )
                    )
                    event_idx += 1

            continue

        if item_type == "event_msg":
            payload_type = payload.get("type")

            if payload_type == "agent_message":
                text = payload.get("message", "").strip()

                if text:
                    events.append(
                        _make_event(
                            index=event_idx,
                            role="assistant",
                            event_type="agent_message",
                            content=text,
                            timestamp=timestamp,
                        )
                    )
                    event_idx += 1

            elif payload_type == "user_message":
                text = payload.get("message", "").strip()

                if text:
                    events.append(
                        _make_event(
                            index=event_idx,
                            role="user",
                            event_type="user_message",
                            content=text,
                            timestamp=timestamp,
                        )
                    )
                    event_idx += 1

            continue

    session_key = session_id or p.stem
    meta = session_index.get(session_key, {})

    thread_name = meta.get("thread_name")
    updated_at = meta.get("updated_at")

    if not events:
        raise SessionParseError(f"No usable events found: {p}")

    return ParsedSession(
        session_key=session_key,
        source_path=str(p),
        started_at=started_at,
        updated_at=updated_at,
        thread_name=thread_name,
        cwd=cwd,
        project_name=project_name,
        repo_name=repo_name,
        model=model,
        events=events,
        raw_format="jsonl",
    )


def _make_event(
    *,
    index: int,
    role: str,
    event_type: str,
    content: str,
    timestamp: str | None,
) -> NormalizedEvent:
    event_hash = _stable_event_hash(
        index=index,
        role=role,
        event_type=event_type,
        content=content,
        timestamp=timestamp,
    )

    return NormalizedEvent(
        index=index,
        role=role,
        event_type=event_type,
        content=content,
        timestamp=timestamp,
        event_hash=event_hash,
    )


def _stable_event_hash(
    *,
    index: int,
    role: str,
    event_type: str,
    content: str,
    timestamp: str | None,
) -> str:
    payload = json.dumps(
        {
            "index": index,
            "role": role,
            "event_type": event_type,
            "content": content,
            "timestamp": timestamp,
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def _extract_message_text(content_items: list[dict]) -> str:
    parts: list[str] = []

    for item in content_items:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        text = item.get("text")

        if item_type in {"input_text", "output_text"} and isinstance(text, str):
            cleaned = text.strip()
            if cleaned:
                parts.append(cleaned)

    return "\n\n".join(parts).strip()


def _summarize_function_call(payload: dict) -> str:
    name = payload.get("name")
    arguments = payload.get("arguments")

    lines: list[str] = []

    if name:
        lines.append(f"Tool call: {name}")

    if isinstance(arguments, str) and arguments.strip():
        lines.append(arguments[:500])

    return "\n".join(lines).strip()