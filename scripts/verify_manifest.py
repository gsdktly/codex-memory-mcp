#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from generate_manifest import _build_manifest  # noqa: E402


def _strip_generated_at(payload: dict) -> dict:
    cleaned = dict(payload)
    cleaned.pop("generated_at", None)
    return cleaned


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = repo_root / "tool_manifest.json"

    if not manifest_path.exists():
        print(f"Missing {manifest_path}. Run scripts/generate_manifest.py first.")
        return 2

    current_manifest = _strip_generated_at(json.loads(manifest_path.read_text()))

    expected_manifest = _strip_generated_at(asyncio.run(_build_manifest()))

    if current_manifest != expected_manifest:
        print("tool_manifest.json is out of date. Run scripts/generate_manifest.py.")
        return 1

    print("tool_manifest.json is up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
