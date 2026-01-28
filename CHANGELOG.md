# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

N/A

## [1.0.0] - 2025-01-27

Complete architectural rewrite. The external interface remains similar, but the
entire codebase has been rebuilt from scratch with a new data model, search
algorithm, and local-first architecture.

### Changed
- **Hybrid search**: Combines BM25 lexical search with FAISS semantic vector search (previously FAISS-only)
- **Cross-encoder reranking**: Uses sentence transformers for improved result quality
- **Nightly updates**: Data toolchain fetched from remote manifest with SHA256 verification
- **Expanded package support**: Now indexes 9 packages (Batteries, CSLib, FLT, FormalConjectures, Init, Lean, Mathlib, PhysLean, Std)
- New data model: `Declaration` replaces `StatementGroup`
- New field names: `name`, `module`, `source_text`, `source_link`, `informalization`
- Simplified API: `SearchEngine`, `Service`, `SearchResult`, `SearchResponse`
- Remote API endpoints: `/declarations/{id}` replaces `/statement_groups/{id}`

### Added
- LLM-generated natural language descriptions (informalizations) for declarations
- New extraction pipeline for processing doc-gen4 output

## [0.3.0] - 2025-06-09

### Added
- Implemented batch processing for `search`, `get_by_id`, and `get_dependencies` methods across the stack, allowing them to accept lists of requests for greater efficiency.
- The **API Client** (`lean_explore.api.client`) now sends batch requests concurrently using `asyncio.gather` to reduce network latency.
- The **Local Service** (`lean_explore.local.service`) was updated to process lists of requests serially against the local database and FAISS index.
- The **MCP Tools** (`lean_explore.mcp.tools`) now expose this batch functionality and provide list-based responses.
- The **AI Agent** instructions (`lean_explore.cli.agent`) were updated to explicitly guide the model to use batch calls for more efficient tool use.

## [0.2.2] - 2025-06-06

### Changed
- Updated minimum Python requirement to `>=3.10`.