import Lake
open Lake DSL

package «formal-conjectures-extractor» where
  -- Workspace for extracting formal-conjectures documentation

require «doc-gen4» from git
  "https://github.com/leanprover/doc-gen4" @ "main"

require «formal_conjectures» from git
  "https://github.com/google-deepmind/formal-conjectures" @ "main"
