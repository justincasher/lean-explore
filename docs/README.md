# LeanExplore Documentation

LeanExplore is a search engine for Lean 4 declarations — theorems, definitions,
lemmas, instances, and more — from Mathlib and other major Lean packages. It
provides hybrid retrieval that matches both by declaration name (e.g.,
`List.map`, `Nat.Prime`) and by informal natural-language meaning (e.g.,
"continuous function on a compact set").

This documentation covers how to install, configure, and use LeanExplore from
the CLI, as a Python library, and as an MCP server for AI assistants.

## Contents

| Page | Description |
|---|---|
| [Getting Started](./getting-started.md) | Install the package, fetch data, and run your first search. |
| [CLI Reference](./cli.md) | Every `lean-explore` command, flag, and option, with examples. |
| [MCP Server](./mcp-server.md) | Run the MCP server and connect it to Claude, Cursor, and other MCP clients. Full reference for every tool exposed. |
| [API Client](./api-client.md) | Use `ApiClient` to query the remote LeanExplore API from Python. |
| [Local Search Backend](./local-backend.md) | How the on-device BM25 + FAISS + cross-encoder pipeline works. |
| [Configuration](./configuration.md) | Environment variables, cache paths, and data layout on disk. |
| [Data Models](./data-models.md) | `SearchResult`, `SearchResponse`, and related types returned from search. |
| [Extraction Pipeline](./extraction-pipeline.md) | Rebuild the dataset from Lean source — contributor-focused. |

## Which backend should I use?

LeanExplore has two backends and you pick one per task:

| | **Remote API** | **Local backend** |
|---|---|---|
| Install | `pip install lean-explore` | `pip install lean-explore[local]` |
| Requires | API key | ~1 GB of data + a few GB of model weights |
| Network | Required per query | Only for initial data fetch |
| Use when | You want zero setup | You want offline, private, or tunable search |

If you don't know yet, start with the remote API — it's the fastest path to a
working search. You can switch to local later without changing your code.

## Links

- Website: <https://www.leanexplore.com>
- Repository: <https://github.com/justincasher/lean-explore>
- Paper: <https://arxiv.org/abs/2506.11085>
- Contributing: [../CONTRIBUTING.md](../CONTRIBUTING.md)
