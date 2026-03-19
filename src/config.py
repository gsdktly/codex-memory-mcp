from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

APP_NAME = "codex-memory-index"


def _get_env_int(name: str, default: int, minimum: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        value = default
    else:
        try:
            value = int(raw)
        except ValueError:
            value = default

    if minimum is not None and value < minimum:
        return minimum

    return value


def setup_logging() -> logging.Logger:
    level_name = os.getenv("CODEX_MEMORY_LOG_LEVEL", "INFO").upper().strip()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    return logging.getLogger(APP_NAME)


def get_db_path() -> Path:
    return Path(os.getenv("CODEX_MEMORY_DB", "./data/codex_memory.sqlite")).expanduser().resolve()


def get_db_timeout_seconds() -> float:
    timeout_ms = _get_env_int("CODEX_MEMORY_DB_TIMEOUT_MS", 5000, minimum=0)
    return timeout_ms / 1000.0


def get_openai_api_key() -> str | None:
    return os.getenv("OPENAI_API_KEY")


def get_openai_model() -> str:
    return os.getenv("CODEX_MEMORY_OPENAI_MODEL", "text-embedding-3-small")


def get_openai_timeout_seconds() -> float:
    timeout_ms = _get_env_int("CODEX_MEMORY_OPENAI_TIMEOUT_MS", 30000, minimum=1000)
    return timeout_ms / 1000.0


def get_openai_max_retries() -> int:
    return _get_env_int("CODEX_MEMORY_OPENAI_MAX_RETRIES", 2, minimum=0)


def get_summary_model() -> str:
    return os.getenv("CODEX_MEMORY_SUMMARY_MODEL", "gpt-4.1-mini")


def get_summary_max_output_tokens() -> int:
    return _get_env_int("CODEX_MEMORY_SUMMARY_MAX_OUTPUT_TOKENS", 400, minimum=32)
