#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path


async def _build_manifest() -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    import sys

    sys.path.insert(0, str(src_path))

    from memory_server import APP_NAME, mcp  # noqa: WPS433

    tools = await mcp.list_tools()
    tool_items = []

    for tool in tools:
        tool_items.append(
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
        )

    return {
        "server": APP_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tools": tool_items,
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    output_path = repo_root / "tool_manifest.json"

    manifest = asyncio.run(_build_manifest())
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
