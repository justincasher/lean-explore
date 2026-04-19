import Lake
open Lake DSL

package «mathlib-extractor» where
  -- Workspace for extracting mathlib documentation

lean_lib «MathExtract» where
  roots := #[`MathExtract]

require «doc-gen4» from git
  "https://github.com/leanprover/doc-gen4" @ "main"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"
