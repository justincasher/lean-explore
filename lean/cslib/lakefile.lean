import Lake
open Lake DSL

package «cslib-extractor» where
  -- Workspace for extracting cslib documentation

lean_lib «CslibExtract» where
  roots := #[`CslibExtract]

require Cslib from git
  "https://github.com/leanprover/cslib" @ "main"

require «doc-gen4» from git
  "https://github.com/leanprover/doc-gen4" @ "v4.28.0-rc1"
