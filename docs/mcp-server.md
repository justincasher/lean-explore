# MCP Server

LeanExplore ships with a [Model Context Protocol](https://modelcontextprotocol.io)
server that exposes Lean search as a set of tools. Any MCP-compatible client
(Claude, Cursor, Zed, Continue, etc.) can launch it and let its model search
Mathlib and other Lean packages during a conversation.

This page covers:

- [Running the server](#running-the-server)
- [Connecting from MCP clients](#connecting-from-mcp-clients)
- [The tools](#the-tools) and their schemas
- [Recommended agent workflow](#recommended-agent-workflow)

## Hosted plugin (Claude Code and Codex)

The recommended setup is the repository's `lean-explore` plugin. It points at
`https://www.leanexplore.com/mcp`; no local executable, account, API key,
browser authorization, or model data is required.

The hosted endpoint allows 30 POST requests per client IP in any 60-second
window. MCP initialization and tool-discovery requests count toward that
total, and clients behind the same NAT or proxy share the limit. When the
limit is reached, the server returns HTTP 429 with a `Retry-After` header.

Claude Code:

```text
/plugin marketplace add justincasher/lean-explore
/plugin install lean-explore@lean-explore
/reload-plugins
```

Codex:

```bash
codex plugin marketplace add https://github.com/justincasher/lean-explore
codex plugin add lean-explore@lean-explore
```

The stdio server described below remains available for existing configurations
and for the fully local backend.

## Running the server

The server speaks MCP over stdio; your client launches it as a subprocess.
You rarely invoke it directly except for debugging.

### Remote API backend (default)

```bash
lean-explore mcp serve --backend api
```

No account or API key is required. The old option remains accepted as a no-op:

```bash
lean-explore mcp serve --backend api --api-key sk-...
```

### Local backend

```bash
lean-explore mcp serve --backend local
```

Requires `pip install lean-explore[local]` and a prior
`lean-explore data fetch`. The first run also downloads the embedding and
reranker models from Hugging Face.

If required data files are missing, the server exits immediately with a
message pointing at `~/.lean_explore/cache/<version>/`.

## Connecting from MCP clients

### Claude Desktop / Claude Code

Add an entry to your MCP settings file (location depends on client; Claude
Desktop uses `~/Library/Application Support/Claude/claude_desktop_config.json`
on macOS):

```json
{
  "mcpServers": {
    "lean-explore": {
      "command": "lean-explore",
      "args": ["mcp", "serve", "--backend", "api"]
    }
  }
}
```

For the local backend, swap `"api"` for `"local"` and drop the `env` block:

```json
{
  "mcpServers": {
    "lean-explore": {
      "command": "lean-explore",
      "args": ["mcp", "serve", "--backend", "local"]
    }
  }
}
```

### Cursor, Zed, Continue, and others

Any client that accepts a command + args will work. Point it at the
`lean-explore` executable with the same arguments shown above.

### Troubleshooting

- **"Essential data files for the local backend are missing"**: run
  `lean-explore data fetch` first.
- **Tools do not appear in the client**: check the client's MCP logs. The
  server logs to stderr; increasing verbosity with
  `python -m lean_explore.mcp.server --backend local --log-level DEBUG`
  helps diagnose startup issues.

## The tools

The server registers seven current tools: `search_summary` and six per-field
getters. It also retains the legacy full-result `search` tool as a deprecated
compatibility alias. IDs returned from `search_summary` can be passed to the
per-field getters to fetch exactly the field you need, which keeps token usage
low.

All tools are annotated as read-only, non-destructive, and idempotent. They
operate only on LeanExplore's indexed corpus and cannot modify Lean packages or
external systems.

### `search_summary`: the preferred starting point

Returns only `id`, `name`, and a short description per hit. Use this first,
then fetch details for the handful of entries you care about.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `query` | string | *required* | Declaration name or natural-language text. |
| `limit` | int | `10` | Maximum results to return. |
| `rerank_top` | int or null | `50` | Candidates to rerank with the cross-encoder. Set `0` or `null` to skip. Only used by the local backend. |
| `packages` | list of strings | `null` | Filter by package name (e.g., `["Mathlib", "Std"]`). |

**Returns**

```json
{
  "query": "prime number divisibility",
  "results": [
    {"id": 12345, "name": "Nat.Prime.dvd_mul", "description": "Divisibility of a product by a prime"}
  ],
  "count": 1,
  "processing_time_ms": 84
}
```

### `search`: deprecated compatibility tool

Do not use this tool in new integrations. It remains registered so existing
MCP clients do not break. It has the same parameters as `search_summary`, but
returns every field for every result and can consume substantially more
context. Use `search_summary` followed by the per-field getters instead.

### Per-field getters

All six take a single parameter `declaration_id` (integer) and return `null`
if the id does not exist. The id comes from a prior `search_summary` result.

| Tool | Returns |
|---|---|
| `get_source_code` | `{id, name, source_text}`: the Lean source. |
| `get_source_link` | `{id, name, source_link}`: GitHub URL to the source. |
| `get_docstring` | `{id, name, docstring}`: the doc comment (may be null). |
| `get_description` | `{id, name, informalization}`: AI-generated natural-language description. |
| `get_module` | `{id, name, module}`: module path (e.g., `Mathlib.Data.List.Basic`). |
| `get_dependencies` | `{id, name, dependencies}`: JSON array of declaration names this depends on. |

## Recommended agent workflow

The server's built-in instructions ask models to follow this pattern:

1. **Browse with `search_summary`** for low token cost.
2. **Fetch only what you need** with the per-field getters (`get_source_code`,
   `get_docstring`, `get_description`, `get_module`, `get_dependencies`,
   `get_source_link`).

The deprecated `search` tool is excluded from this workflow and exists only
for backwards compatibility.

This keeps context short while still giving the model access to full source
code and dependency chains when it matters.

## Query styles

Both styles work through the same endpoints; the hybrid retriever decides
internally which signal applies:

- **By name**: `List.map`, `Nat.Prime`, `CategoryTheory.Functor.map`.
- **By meaning**: `"continuous function on a compact set"`, `"sum of a
  geometric series"`, `"a group homomorphism preserving multiplication"`.

You do not need to specify which mode you want.

## See also

- [CLI Reference](./cli.md) for `lean-explore mcp serve` options.
- [Local Search Backend](./local-backend.md) for how hybrid retrieval works
  under the hood.
- [Configuration](./configuration.md) for environment variables and cache
  paths.
