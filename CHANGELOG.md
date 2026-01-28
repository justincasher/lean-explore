# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

N/A

## [1.0.0] - 2025-01-27

Complete architectural rewrite with a new extraction pipeline that enables
nightly data updates and dynamic package indexing.

### Added
- **Extraction pipeline**: Automated pipeline for processing doc-gen4 output, enabling nightly data refreshes
- **Cross-encoder reranking**: Uses sentence transformers for improved search result quality

### Changed
- **Expanded package support**: Now indexes 9 packages (Batteries, CSLib, FLT, FormalConjectures, Init, Lean, Mathlib, PhysLean, Std)
- New data model: `Declaration` replaces `StatementGroup`
- New field names: `name`, `module`, `source_text`, `source_link`, `informalization`
- Simplified API: `SearchEngine`, `Service`, `SearchResult`, `SearchResponse`
- Remote API endpoints: `/declarations/{id}` replaces `/statement_groups/{id}`

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