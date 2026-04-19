# API Client

`lean_explore.api.ApiClient` is an async HTTP client for the hosted
LeanExplore API. It ships with the base package — no PyTorch, no local
indices, no data download required.

## Install and authenticate

```bash
pip install lean-explore
```

Get an API key from <https://www.leanexplore.com> and set it as an
environment variable:

```bash
export LEANEXPLORE_API_KEY="your-key-here"
```

Or pass it explicitly to the client constructor.

## Quick start

```python
import asyncio
from lean_explore.api import ApiClient

async def main():
    client = ApiClient()  # reads LEANEXPLORE_API_KEY

    response = await client.search("prime number divisibility", limit=5)
    for result in response.results:
        print(f"{result.name}  ({result.module})")

asyncio.run(main())
```

The `search` and `get_by_id` methods are both `async`, so call them from
inside an async function or via `asyncio.run`.

## `ApiClient`

```python
ApiClient(api_key: str | None = None, timeout: float = 10.0)
```

| Parameter | Default | Description |
|---|---|---|
| `api_key` | `None` | API key. Falls back to `LEANEXPLORE_API_KEY` env var. Raises `ValueError` if neither is provided. |
| `timeout` | `10.0` | HTTP timeout in seconds for every request. |

The client hits `https://www.leanexplore.com/api/v2` by default.

### `search()`

```python
async def search(
    query: str,
    limit: int = 20,
    rerank_top: int | None = None,
    packages: list[str] | None = None,
) -> SearchResponse
```

| Parameter | Default | Description |
|---|---|---|
| `query` | *required* | Declaration name or natural-language text. |
| `limit` | `20` | Maximum results to return. |
| `rerank_top` | `None` | Ignored by the API backend (reranking is server-side). Accepted for interface parity with the local `Service`. |
| `packages` | `None` | Filter by package, e.g., `["Mathlib", "Std"]`. |

Returns a [`SearchResponse`](./data-models.md#searchresponse) with `query`,
`results`, `count`, and `processing_time_ms`.

Raises:

- `httpx.HTTPStatusError` — the API returned a non-2xx status.
- `httpx.RequestError` — network, DNS, or timeout failure.

### `get_by_id()`

```python
async def get_by_id(declaration_id: int) -> SearchResult | None
```

Fetches a single declaration by its numeric id (ids come from `search`
results). Returns a [`SearchResult`](./data-models.md#searchresult) on hit,
or `None` on a 404. Other non-2xx statuses raise
`httpx.HTTPStatusError`.

## Full example

```python
import asyncio
from lean_explore.api import ApiClient

async def main():
    client = ApiClient(api_key="sk-...", timeout=15.0)

    response = await client.search(
        query="continuous function on a compact set",
        limit=10,
        packages=["Mathlib"],
    )
    print(f"{response.count} results in {response.processing_time_ms} ms")

    # Grab the first result's full source
    if response.results:
        first = response.results[0]
        print(f"\n{first.name}")
        print(first.source_link)
        print(first.source_text)

        # Round-trip by id
        same = await client.get_by_id(first.id)
        assert same is not None and same.id == first.id

asyncio.run(main())
```

## Notes

- The client creates a fresh `httpx.AsyncClient` per call. If you make many
  rapid requests, consider wrapping them in `asyncio.gather(...)` — they run
  concurrently without extra setup.
- Package names follow the casing shown in results (e.g., `Mathlib`, `Std`,
  `PhysLean`).
- For offline use or custom tuning, see [Local Search Backend](./local-backend.md)
  and use `lean_explore.search.Service` instead.

## See also

- [Data Models](./data-models.md) — field reference for `SearchResult` and
  `SearchResponse`.
- [Configuration](./configuration.md) — environment variables including
  `LEANEXPLORE_API_KEY`.
