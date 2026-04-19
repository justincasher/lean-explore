# Getting Started

This guide takes you from zero to your first search in under five minutes.
LeanExplore has two paths:

1. **Remote API**: a small pure-Python install that talks to the hosted
   search service. Best for getting started.
2. **Local backend**: runs the full search pipeline on your machine. Larger
   install, no network calls after the initial data fetch.

## Requirements

- Python 3.10 or newer (3.11, 3.12 also supported)
- `pip` (or `uv`, `poetry`, etc.)
- For the local backend: roughly 1 GB of disk for data files and a few GB for
  the first-run model download (embedding + reranker models from Hugging Face)

## Option 1: Remote API (recommended to start)

### 1. Install

```bash
pip install lean-explore
```

This installs the CLI, the `ApiClient`, and the MCP server, roughly 50 MB
of pure-Python and C-extension dependencies. No PyTorch.

### 2. Get an API key

Sign up and generate a key at <https://www.leanexplore.com>. Then export it:

```bash
export LEANEXPLORE_API_KEY="your-key-here"
```

You can also add it to your shell profile (`~/.zshrc`, `~/.bashrc`) so it
persists between sessions.

### 3. Run a search

```bash
lean-explore search "prime number divisibility"
lean-explore search "List.map" --limit 10
lean-explore search "fundamental theorem of arithmetic" --package Mathlib
```

The first argument is the query. It can be a Lean declaration name, a partial
name, or a natural-language description. The search engine handles both at
once; you don't need to pick a mode.

### 4. (Optional) Run the MCP server

If you want to give Claude, Cursor, or another MCP client access to
LeanExplore:

```bash
lean-explore mcp serve --backend api
```

See [MCP Server](./mcp-server.md) for client configuration.

## Option 2: Local backend

Use this when you need offline search, want more control over parameters, or
want to avoid per-query network latency.

### 1. Install with the `[local]` extra

```bash
pip install lean-explore[local]
```

This adds PyTorch and `sentence-transformers`. Expect roughly 1–2 GB of
Python dependencies.

### 2. Fetch the data

```bash
lean-explore data fetch
```

This downloads the latest prebuilt database, FAISS index, and BM25 indices to
`~/.lean_explore/cache/<version>/`. The total download is on the order of
1 GB.

### 3. Run the MCP server with the local backend

```bash
lean-explore mcp serve --backend local
```

The first run will also download the embedding and reranker models from
Hugging Face (`Qwen/Qwen3-Embedding-0.6B` and `Qwen/Qwen3-Reranker-0.6B`),
caching them under `~/.cache/huggingface/`. Subsequent runs are fast.

The `lean-explore search` CLI currently uses the remote API. For programmatic
local search, see [Local Search Backend](./local-backend.md) and use
`SearchEngine` / `Service` directly.

## What's indexed

LeanExplore currently covers:

- Batteries
- CSLib
- FLT (Fermat's Last Theorem)
- FormalConjectures
- Init
- Lean (core)
- Mathlib
- PhysLean
- Std

You can filter any search to a subset with `--package` (CLI) or the
`packages=[...]` argument (API/MCP).

## Next steps

- [CLI Reference](./cli.md): every command and flag.
- [MCP Server](./mcp-server.md): wire LeanExplore into Claude or Cursor.
- [API Client](./api-client.md): use `ApiClient` from Python.
- [Configuration](./configuration.md): environment variables and data layout.
