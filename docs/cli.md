# CLI Reference

The `lean-explore` command is installed alongside the Python package. It is
organized as a small command tree:

```
lean-explore
├── search               # Query Lean declarations
├── mcp
│   └── serve            # Run the MCP server (api or local backend)
└── data
    ├── fetch            # Download the data toolchain
    └── clean            # Delete all cached data
```

Run `lean-explore --help` or `lean-explore <command> --help` for built-in
help at any time.

## `lean-explore search`

Search the Lean Explore index and print results to your terminal.

```bash
lean-explore search QUERY [OPTIONS]
```

### Arguments

| Argument | Description |
|---|---|
| `QUERY` | The search query. Can be a declaration name (`List.map`, `Nat.Prime`) or natural-language text (`"continuous function on a compact set"`). Required. |

### Options

| Flag | Default | Description |
|---|---|---|
| `--limit`, `-n` | `5` | Number of results to display. |
| `--package`, `-p` | (all) | Filter by package. Repeatable: `-p Mathlib -p Std`. |

### Requirements

`lean-explore search` uses the remote API. You must have `LEANEXPLORE_API_KEY`
set in your environment:

```bash
export LEANEXPLORE_API_KEY="your-key-here"
```

### Examples

```bash
lean-explore search "List.map"
lean-explore search "prime number divisibility" --limit 10
lean-explore search "fundamental theorem of calculus" -p Mathlib
lean-explore search "continuous" -p Mathlib -p Std -n 20
```

## `lean-explore mcp serve`

Launch the Model Context Protocol server so an MCP client (Claude, Cursor,
etc.) can call LeanExplore's search tools. The server speaks MCP over stdio,
so you normally do not run it manually; your MCP client launches it. Running
it directly is mostly useful for debugging.

```bash
lean-explore mcp serve [OPTIONS]
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--backend`, `-b` | `api` | Backend to use: `api` or `local`. |
| `--api-key` | (none) | API key for the `api` backend. Overrides `LEANEXPLORE_API_KEY`. |

### Backends

- **`api`**: Delegates every query to the hosted LeanExplore API. Requires an
  API key (via env var or `--api-key`).
- **`local`**: Runs the full hybrid search pipeline on-device. Requires
  `pip install lean-explore[local]` and `lean-explore data fetch`.

### Examples

```bash
# Remote API (most users)
lean-explore mcp serve --backend api

# Remote API with an inline key
lean-explore mcp serve --backend api --api-key sk-...

# Local, fully offline backend
lean-explore mcp serve --backend local
```

For MCP tool schemas and client configuration, see
[MCP Server](./mcp-server.md).

## `lean-explore data fetch`

Download the prebuilt data toolchain used by the local search backend. This
pulls the SQLite database, the FAISS semantic index, and the BM25 indices from
remote storage into `~/.lean_explore/cache/<version>/`.

```bash
lean-explore data fetch [OPTIONS]
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--version`, `-v` | latest | Specific version to install (e.g., `20260127_103630`). |

### Behavior

1. Fetches the current version tag (or uses the one you passed).
2. Downloads all required files to `~/.lean_explore/cache/<version>/`:
   - `lean_explore.db`
   - `informalization_faiss.index` and `informalization_faiss_ids_map.json`
   - `bm25_ids_map.json`
   - `bm25_name_raw/` and `bm25_name_spaced/` directories
3. Writes `~/.lean_explore/active_version` so future commands resolve the
   correct cache directory.
4. Removes any older cached versions to reclaim disk space.

Already-present files are skipped, so re-running `fetch` after an interrupted
download will resume rather than start over.

### Examples

```bash
# Install the latest version
lean-explore data fetch

# Pin a specific version
lean-explore data fetch --version 20260127_103630
```

## `lean-explore data clean`

Remove every cached data toolchain and clear the active-version pointer.
Useful if you want to reset state or free disk.

```bash
lean-explore data clean
```

This command prompts for confirmation before deleting anything. It does not
touch downloaded model weights; those live under `~/.cache/huggingface/`.

## Exit codes

All CLI commands follow standard conventions:

- `0`: success
- non-zero: an error occurred (missing API key, failed download, etc.).
  An error message is printed to stderr.

## See also

- [Getting Started](./getting-started.md)
- [MCP Server](./mcp-server.md)
- [Configuration](./configuration.md)
