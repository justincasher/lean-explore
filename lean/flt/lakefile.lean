import Lake
open Lake DSL

package «flt-extractor» where
  -- Workspace for extracting FLT documentation

lean_lib «FLTExtract» where
  roots := #[`FLTExtract]

require FLT from git
  "https://github.com/ImperialCollegeLondon/FLT" @ "main"

require «doc-gen4» from git
  "https://github.com/leanprover/doc-gen4" @ "v4.27.0"
