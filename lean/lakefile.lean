import Lake
open Lake DSL

package «extractor» where
  -- add package configuration options here

lean_lib «LeanExtract» where
  -- Library that imports Mathlib and PhysLean for documentation generation
  roots := #[`LeanExtract]

require «doc-gen4» from git
  "https://github.com/leanprover/doc-gen4" @ "v4.24.0"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.24.0"

require PhysLean from git
  "https://github.com/HEPLean/PhysLean" @ "v4.24.0"
