from __future__ import annotations

from openai import OpenAI

from config import (
    get_openai_api_key,
    get_openai_max_retries,
    get_openai_timeout_seconds,
    get_summary_max_output_tokens,
    get_summary_model,
)
from session_parser import NormalizedEvent

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = get_openai_api_key()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in the environment or .env file.")

        _client = OpenAI(
            api_key=api_key,
            timeout=get_openai_timeout_seconds(),
            max_retries=get_openai_max_retries(),
        )

    return _client


def summarize_session(events: list[NormalizedEvent]) -> str:
    transcript = _events_to_transcript(events, limit=80)

    prompt = f"""
You are summarizing an AI engineering session.

Produce a concise technical summary capturing:

- problem being solved
- debugging steps
- tools used
- key conclusions
- files/modules involved

Transcript:

{transcript}
"""

    resp = _get_client().responses.create(
        model=get_summary_model(),
        input=prompt,
        max_output_tokens=get_summary_max_output_tokens(),
    )

    return resp.output_text.strip()


def update_session_summary(
    previous_summary: str,
    new_events: list[NormalizedEvent],
) -> str:
    new_transcript = _events_to_transcript(new_events, limit=60)

    prompt = f"""
You are updating an existing summary of an AI engineering session.

Existing summary:
{previous_summary}

New session activity:
{new_transcript}

Update the summary so it remains concise and technically useful.
Preserve prior conclusions that still matter and add only genuinely new information:
- new debugging steps
- newly used tools
- changed conclusions
- new files/modules involved
- important outcomes or remaining blockers
"""

    resp = _get_client().responses.create(
        model=get_summary_model(),
        input=prompt,
        max_output_tokens=get_summary_max_output_tokens(),
    )

    return resp.output_text.strip()


def _events_to_transcript(events: list[NormalizedEvent], limit: int) -> str:
    blocks: list[str] = []

    for event in events[-limit:]:
        blocks.append(f"{event.role}: {event.content[:400]}")

    return "\n".join(blocks)
