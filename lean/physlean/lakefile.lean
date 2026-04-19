import Lake
open Lake DSL

package «physlean-extractor» where
  -- Workspace for extracting PhysLean documentation

@[default_target]
lean_lib «PhysExtract» where
  roots := #[`PhysExtract]

require «doc-gen4» from git
  "https://github.com/leanprover/doc-gen4" @ "v4.29.0"

require PhysLean from git
  "https://github.com/HEPLean/PhysLean"
