# Data Models

All public search results are Pydantic v2 models, defined in
`lean_explore.models`. Both the remote `ApiClient` and the local `Service`
return the same types, so code written against one works against the other.

## Imports

```python
from lean_explore.models import (
    SearchResult,
    SearchResponse,
    SearchResultSummary,
    SearchSummaryResponse,
    extract_bold_description,
)
```

## `SearchResult`

A single Lean declaration returned from a search.

| Field | Type | Description |
|---|---|---|
| `id` | `int` | Primary key identifier. |
| `name` | `str` | Fully qualified Lean name (e.g., `Nat.add`). |
| `module` | `str` | Module path (e.g., `Mathlib.Data.List.Basic`). |
| `docstring` | `str \| None` | Doc comment from the Lean source. May be `None`. |
| `source_text` | `str` | Full Lean source code for the declaration. |
| `source_link` | `str` | GitHub URL to the source. |
| `dependencies` | `str \| None` | JSON-encoded array of declaration names this depends on. Parse with `json.loads` if you need the list. |
| `informalization` | `str \| None` | AI-generated natural-language description. Typically starts with a bold header: `**Title.** Rest...`. |

Example:

```python
for result in response.results:
    print(result.name)
    print("  module:", result.module)
    print("  link:  ", result.source_link)
    if result.docstring:
        print("  doc:   ", result.docstring.splitlines()[0])
```

## `SearchResponse`

Envelope returned by `ApiClient.search()` and `Service.search()`.

| Field | Type | Description |
|---|---|---|
| `query` | `str` | The original query string. |
| `results` | `list[SearchResult]` | Hits, ordered best-first. |
| `count` | `int` | `len(results)`. |
| `processing_time_ms` | `int \| None` | Server/engine latency in milliseconds. |

## `SearchResultSummary`

Slim form used by the MCP `search_summary` tool. Contains only enough to let
a caller decide which ids to drill into.

| Field | Type | Description |
|---|---|---|
| `id` | `int` | Declaration id (feed into per-field getters). |
| `name` | `str` | Fully qualified Lean name. |
| `description` | `str \| None` | The bold header extracted from the informalization. |

## `SearchSummaryResponse`

Envelope for slim results.

| Field | Type | Description |
|---|---|---|
| `query` | `str` | The original query string. |
| `results` | `list[SearchResultSummary]` | Slim hits, ordered best-first. |
| `count` | `int` | `len(results)`. |
| `processing_time_ms` | `int \| None` | Latency in milliseconds. |

## `extract_bold_description`

```python
extract_bold_description(informalization: str | None) -> str | None
```

Pulls the `**Bold Title.**` prefix out of an informalization. Returns `None`
if the input is `None` or does not start with a bold header. This is the
helper used to build `SearchResultSummary.description`.

## ORM model (advanced)

If you are working directly against the SQLite database (e.g., in the
extraction pipeline), `lean_explore.models.Declaration` is the SQLAlchemy
ORM model that backs the `declarations` table. Most users should stick to
the Pydantic models above.

## See also

- [API Client](./api-client.md) — returns `SearchResponse` / `SearchResult`.
- [Local Search Backend](./local-backend.md) — same return types via
  `Service`.
- [MCP Server](./mcp-server.md) — tool return payloads serialize these
  models to JSON.
