# Local Search Backend

The local backend runs LeanExplore's full hybrid retrieval pipeline on your
machine. Once the data is fetched, no network calls are needed — queries
execute entirely locally.

This page explains:

- [When to use it](#when-to-use-it)
- [Installation and data setup](#installation-and-data-setup)
- [How hybrid retrieval works](#how-hybrid-retrieval-works)
- [Using it from Python](#using-it-from-python)

## When to use it

- **Offline use** — no network per query after the initial data fetch.
- **Privacy** — queries never leave your machine.
- **Tuning** — you can adjust candidate pool sizes and rerank depth.
- **Integration** — embed `SearchEngine` or `Service` directly in your
  application or data pipeline.

For quick exploration or low-setup use, prefer the
[remote API client](./api-client.md) instead.

## Installation and data setup

```bash
pip install lean-explore[local]
lean-explore data fetch
```

This adds PyTorch and `sentence-transformers`, then downloads the prebuilt
database and indices into `~/.lean_explore/cache/<version>/`. The first
search will additionally download the embedding model
(`Qwen/Qwen3-Embedding-0.6B`) and reranker model
(`Qwen/Qwen3-Reranker-0.6B`) from Hugging Face, cached under
`~/.cache/huggingface/`.

GPU is optional but used automatically when available.

## How hybrid retrieval works

A query flows through five stages:

1. **BM25 lexical search on names.** Two BM25 indices score the query against
   declaration names — one using raw tokenization (entire dotted name as a
   single token), one using spaced tokenization (splitting on dots,
   underscores, and camelCase). This catches both exact and fuzzy name
   matches (e.g., `list.map` → `List.map`).
2. **FAISS semantic search on informalizations.** Each declaration has an
   AI-generated natural-language description (its *informalization*). The
   query is embedded with the Qwen embedding model and the top FAISS
   neighbours are retrieved. This catches "what does this thing mean" style
   queries.
3. **Reciprocal Rank Fusion.** The candidate lists are merged by summing
   `1/rank` across the BM25 and FAISS results, producing a single fused
   ranking.
4. **Dependency boosting.** Candidates that are depended on by top-ranked
   results get a small score boost, surfacing foundational lemmas.
5. **Cross-encoder reranking.** The top N candidates (default 25–50) are
   rescored pairwise with the Qwen reranker, which looks at the query and
   candidate text together. The final top results are returned.

Auto-generated declarations (e.g., `.mk` constructors) are filtered from the
final list.

Switches available on every search call:

- `limit` — how many results to return (default 20–50).
- `rerank_top` — how deep to rerank. Larger is slower but more precise.
  Setting it to `0` or `None` skips reranking entirely.
- `packages` — restrict to a subset of packages.

## Using it from Python

Two layers are available:

- **`Service`** — the convenient wrapper. Returns a `SearchResponse` with
  timing metadata.
- **`SearchEngine`** — the lower-level engine. Returns a plain
  `list[SearchResult]` and exposes all retrieval knobs.

### `Service`

```python
import asyncio
from lean_explore.search import Service

async def main():
    service = Service()  # spins up a default SearchEngine

    response = await service.search(
        query="continuous function on a compact set",
        limit=10,
        rerank_top=50,
        packages=["Mathlib"],
    )
    for result in response.results:
        print(result.name, "-", result.module)

    # Look up one declaration by id
    first = response.results[0]
    same = await service.get_by_id(first.id)

asyncio.run(main())
```

### `SearchEngine`

```python
import asyncio
from lean_explore.search import SearchEngine

async def main():
    engine = SearchEngine()

    results = await engine.search(
        query="List.map",
        limit=25,
        faiss_k=1000,     # how many FAISS candidates to pull
        bm25_k=1000,      # how many BM25 candidates to pull
        rerank_top=25,    # depth of cross-encoder reranking
        packages=["Mathlib", "Std"],
    )
    for result in results:
        print(result.name)

asyncio.run(main())
```

Key constructor options (all optional; sensible defaults):

| Parameter | Default | Purpose |
|---|---|---|
| `db_url` | configured URL | SQLAlchemy database URL. |
| `embedding_model_name` | `Qwen/Qwen3-Embedding-0.6B` | Sentence-transformer model for queries. |
| `reranker_model_name` | `Qwen/Qwen3-Reranker-0.6B` | Cross-encoder used for final reranking. |
| `use_local_data` | `False` | `True` reads from `DATA_DIRECTORY` (extraction output). `False` reads from `CACHE_DIRECTORY` (downloaded data). |

For the full set of parameters, see the class docstring in
`src/lean_explore/search/engine.py`.

## Tuning tips

- **Small queries, fast responses** — set `rerank_top=0` to skip the
  cross-encoder. You lose some precision but gain significant speed.
- **Recall-heavy queries** — bump `faiss_k` and `bm25_k` to widen the
  candidate pool before reranking.
- **Memory-constrained machines** — lower `LEAN_EXPLORE_EMBEDDING_BATCH_SIZE`
  and `LEAN_EXPLORE_RERANKER_BATCH_SIZE`. See [Configuration](./configuration.md).

## See also

- [Data Models](./data-models.md) for the shape of `SearchResult`.
- [Configuration](./configuration.md) for env vars that control batch sizes
  and cache paths.
- [MCP Server](./mcp-server.md) — the local backend powers
  `lean-explore mcp serve --backend local`.
