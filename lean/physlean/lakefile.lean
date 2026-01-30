import Lake
open Lake DSL

package «physlean-extractor» where
  -- Workspace for extracting PhysLean documentation

lean_lib «PhysExtract» where
  roots := #[`PhysExtract]

require PhysLean from git
  "https://github.com/HEPLean/PhysLean"

require «doc-gen4» from git
  "https://github.com/leanprover/doc-gen4" @ "v4.26.0"
