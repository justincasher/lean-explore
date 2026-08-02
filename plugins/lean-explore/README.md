# LeanExplore plugin

Search Lean 4 declarations from Claude Code or Codex through LeanExplore's
hosted public MCP server. There is no sign-in, browser authorization, or API
key to create, copy, or store, and the plugin does not download the local
search index.

The MCP endpoint is `https://www.leanexplore.com/mcp`.

The hosted endpoint allows 30 POST requests per client IP in any 60-second
window. MCP initialization and tool-discovery requests count toward the same
limit, and clients sharing a public IP share the budget. Limited requests
receive HTTP 429 with a `Retry-After` header.

The Codex marketplace schema requires an authentication timing policy, so the
marketplace entry uses `ON_INSTALL`. This is lifecycle metadata only: the
plugin declares no credentials, and the server sends no authentication
challenge.
