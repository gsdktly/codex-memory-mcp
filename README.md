# Codex Memory MCP

Local MCP stdio server that indexes Codex session history into SQLite so prior work is searchable without a hosted vector store.

Session content stays local. The default embedder calls the OpenAI API for embeddings.

## What It Includes

- `src/session_parser.py`: defensive parser for Codex session files
- `src/chunker.py`: message-aware chunking for memory records
- `src/db.py`: SQLite schema, upserts, cosine-similarity search
- `src/embeddings.py`: embedding wrapper using OpenAI Python SDK
- `src/session_summarizer.py`: summary + summary embedding generation
- `src/ingest_sessions.py`: incremental ingest pipeline
- `src/memory_server.py`: MCP server exposing the memory tools

## Tools

- `search_codex_sessions(query, limit=5, project_name=None, repo_name=None, session_key=None)`
- `fetch_session_chunk(chunk_id)`
- `list_recent_sessions(limit=10)`
- `search_session_summaries(query, limit=5)`

## Requirements

- Python 3.11+ recommended
- OpenAI API key (for embeddings and summaries)

## Install

```bash
cd /path/to/codex-memory-mcp
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Environment Variables

The server loads `.env` in the repo root.

Required:

- `OPENAI_API_KEY`

Optional (defaults shown):

- `CODEX_MEMORY_DB=./data/codex_memory.sqlite`
- `CODEX_MEMORY_DB_TIMEOUT_MS=5000`
- `CODEX_MEMORY_OPENAI_MODEL=text-embedding-3-small`
- `CODEX_MEMORY_OPENAI_TIMEOUT_MS=30000`
- `CODEX_MEMORY_OPENAI_MAX_RETRIES=2`
- `CODEX_MEMORY_SUMMARY_MODEL=gpt-4.1-mini`
- `CODEX_MEMORY_SUMMARY_MAX_OUTPUT_TOKENS=400`
- `CODEX_MEMORY_LOG_LEVEL=INFO`

## Ingest Local Codex Sessions

```bash
python src/ingest_sessions.py ~/.codex/sessions --db ./data/codex_memory.sqlite --verbose
```

## Run The MCP Server

```bash
python src/memory_server.py
```

## Register With Codex

Example `~/.codex/config.toml` entry:

```toml
[[mcp_servers]]
name = "codex-memory-index"
command = "python"
args = ["/absolute/path/to/codex-memory-mcp/src/memory_server.py"]

[mcp_servers.env]
OPENAI_API_KEY = "your_openai_key"
CODEX_MEMORY_DB = "/absolute/path/to/codex-memory-mcp/data/codex_memory.sqlite"
```

Also see `examples/codex-config.toml` and `examples/mcp.json`.

## Tool Manifest

Generate a manifest from the server code:

```bash
python scripts/generate_manifest.py
```

Verify the checked-in manifest matches current tools:

```bash
python scripts/verify_manifest.py
```

The generated file is `tool_manifest.json`.

## Smoke Tests (Read-Only)

```bash
python scripts/run_smoke_tests.py
```

Use a custom manifest:

```bash
python scripts/run_smoke_tests.py /absolute/path/to/manifest.json
```

## Notes And Caveats

- The ingest pipeline is append-only and detects non-append mutations. If a session file changes in the middle, rebuild the DB.
- Similarity search currently loads embeddings into Python for cosine scoring. For large archives, consider sqlite-vec or ANN.
- Embeddings and summaries are generated via OpenAI by default. Replace `src/embeddings.py` and `src/session_summarizer.py` to go fully local.

## Versioning

Version is stored in `VERSION`. Release notes are tracked in `CHANGELOG.md`.
