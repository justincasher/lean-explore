# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

N/A

## [1.0.0] - 2025-01-27

### Changed
- Complete rewrite of the search engine with hybrid BM25 + semantic search
- New local search backend using FAISS for vector similarity search
- Cross-encoder reranking for improved result quality
- Simplified API with `SearchEngine` and `Service` classes
- MCP server for AI assistant integration
- CLI commands for data management (`lean-explore data fetch`, `lean-explore data clean`)
- Extraction pipeline for processing doc-gen4 output
- Support for multiple Lean packages (Mathlib, PhysLean, FLT, etc.)

### Added
- TypedDict definitions for improved type safety
- Nightly data updates via remote manifest
- Informalization generation using LLMs
- Embedding generation using sentence transformers

### Removed
- Legacy API client (replaced with local-first architecture)
- Old batch processing methods

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