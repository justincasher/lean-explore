import Lake
open Lake DSL

package «extractor» where
  -- add package configuration options here

require «doc-gen4» from git
  "https://github.com/leanprover/doc-gen4" @ "v4.23.0"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.23.0"
