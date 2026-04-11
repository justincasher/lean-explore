import Lake
open Lake DSL

package «flt-extractor» where
  -- Workspace for extracting FLT documentation

lean_lib «FLTExtract» where
  roots := #[`FLTExtract]

require «doc-gen4» from git
  "https://github.com/leanprover/doc-gen4" @ "main"

require FLT from git
  "https://github.com/ImperialCollegeLondon/FLT" @ "main"
