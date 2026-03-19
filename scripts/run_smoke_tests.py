#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import anyio
from mcp import types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

DEFAULT_TIMEOUT_MS = 30000
DEFAULT_DELAY_AFTER_MS = 200


async def run_single_test(
    *,
    server_path: Path,
    repo_root: Path,
    method: str,
    params: dict[str, Any],
    timeout_ms: int,
    db_path: Path,
) -> tuple[bool, str]:
    env = dict(os.environ)
    env.setdefault("CODEX_MEMORY_DB", str(db_path))
    env.setdefault("PYTHONUNBUFFERED", "1")

    server = StdioServerParameters(
        command=sys.executable,
        args=["-u", str(server_path)],
        env=env,
        cwd=repo_root,
    )

    async with stdio_client(server) as (read_stream, write_stream):
        session = ClientSession(read_stream, write_stream)

        async with session:
            with anyio.fail_after(timeout_ms / 1000.0):
                await session.initialize()
                await session.send_notification(
                    types.ClientNotification(types.InitializedNotification())
                )

                if method == "tools/list":
                    await session.list_tools()
                    return True, ""

                if method == "tools/call":
                    tool_name = params.get("name")
                    if not tool_name:
                        return False, "Missing params.name for tools/call"

                    arguments = params.get("arguments") or {}
                    await session.call_tool(tool_name, arguments)
                    return True, ""

                return False, f"Unsupported method: {method}"


async def run_smoke_suite(manifest_path: Path, server_path: Path, repo_root: Path) -> int:
    manifest = json.loads(manifest_path.read_text())

    defaults = manifest.get("defaults") or {}
    timeout_default = int(defaults.get("timeout_ms", DEFAULT_TIMEOUT_MS))
    delay_default = int(defaults.get("delay_after_ms", DEFAULT_DELAY_AFTER_MS))

    tests = manifest.get("tests")
    if not isinstance(tests, list) or not tests:
        print("Manifest has no tests.")
        return 2

    print(f"Smoke manifest: {manifest_path}")
    print(f"Tests: {len(tests)}")
    print("Mode: sequential, one isolated server process per test")

    results: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="codex-memory-smoke-") as tmp_dir:
        db_path = Path(tmp_dir) / "codex_memory.sqlite"

        for idx, test in enumerate(tests):
            name = test.get("name", f"test-{idx + 1}")
            method = test.get("method")
            params = test.get("params") or {}
            timeout_ms = int(test.get("timeout_ms", timeout_default))
            delay_after = int(test.get("delay_after_ms", delay_default))

            label = f"{idx + 1}/{len(tests)} {name}"
            sys.stdout.write(f"RUN  {label} ... ")
            sys.stdout.flush()

            start = time.time()
            ok = True
            reason = ""

            try:
                ok, reason = await run_single_test(
                    server_path=server_path,
                    repo_root=repo_root,
                    method=method,
                    params=params,
                    timeout_ms=timeout_ms,
                    db_path=db_path,
                )
            except TimeoutError:
                ok = False
                reason = f"timeout after {timeout_ms}ms"
            except Exception as exc:  # noqa: BLE001
                ok = False
                reason = str(exc)

            duration_ms = int((time.time() - start) * 1000)

            if ok:
                sys.stdout.write(f"PASS ({duration_ms}ms)\n")
            else:
                sys.stdout.write(f"FAIL ({duration_ms}ms) :: {reason}\n")

            results.append({
                "name": name,
                "ok": ok,
                "duration_ms": duration_ms,
                "reason": reason,
            })

            if delay_after > 0 and idx < len(tests) - 1:
                await anyio.sleep(delay_after / 1000.0)

    pass_count = sum(1 for r in results if r["ok"])
    fail_count = len(results) - pass_count

    print("\nSummary")
    print(f"pass={pass_count} fail={fail_count} total={len(results)}")

    for r in results:
        if not r["ok"]:
            print(f"FAIL {r['name']}: {r['reason']}")

    return 0 if fail_count == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run read-only smoke tests for the MCP server.")
    parser.add_argument(
        "manifest",
        nargs="?",
        default=str(Path(__file__).resolve().parents[1] / "smoke" / "read_only_smoke_tests.json"),
        help="Path to smoke test manifest JSON",
    )
    parser.add_argument(
        "--server",
        default=str(Path(__file__).resolve().parents[1] / "src" / "memory_server.py"),
        help="Path to MCP server entrypoint",
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    server_path = Path(args.server).resolve()
    repo_root = Path(__file__).resolve().parents[1]

    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return 2

    if not server_path.exists():
        print(f"Server entrypoint not found: {server_path}")
        return 2

    return anyio.run(run_smoke_suite, manifest_path, server_path, repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
