import Lake
open Lake DSL

package «cslib-extractor» where
  -- Workspace for extracting cslib documentation

lean_lib «CslibExtract» where
  roots := #[`CslibExtract]

require «doc-gen4» from git
  "https://github.com/leanprover/doc-gen4" @ "main"

require Cslib from git
  "https://github.com/leanprover/cslib" @ "main"
