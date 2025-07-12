import Lake
open Lake DSL

package «extractor» where
  -- add package configuration options here

/- Fixed all of the packages to use Lean v4.19.0 -/

-- Require Aesop (a tableaux-based automated theorem prover)
require aesop from git
  "https://github.com/leanprover-community/aesop.git" @ "v4.21.0"

-- Require Batteries (a general-purpose utility library)
require batteries from git
  "https://github.com/leanprover-community/batteries" @ "v4.21.0"

-- Require Duper (a superposition-based automated theorem prover)
-- require Duper from git
--  "https://github.com/leanprover-community/duper.git" @ "v0.0.25"

-- Require FLT (a library for formalizing FLT)
-- require FLT from git 
--  "https://github.com/ImperialCollegeLondon/FLT.git" @ "v4.19.0"

-- Require ImportGraph (transitive dep, now made direct to override version)
require importGraph from git
  "https://github.com/leanprover-community/import-graph" @ "v4.21.0"

-- Require Mathlib (the comprehensive library of mathematics in Lean)
require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.21.0"

-- Require Pantograph (a library for automated theorem proving)
-- require pantograph from git
--  "https://github.com/lenianiva/Pantograph.git" @ "v0.3.1"

-- Require Plausible (transitive dep, now made direct to override version)
require plausible from git
  "https://github.com/leanprover-community/plausible.git" @ "v4.21.0"

-- Require PhysLean (a library for physics-related concepts in Lean)
require PhysLean from git
  "https://github.com/HEPLean/PhysLean.git" @ "v4.21.0"

-- Require ProofWidgets4 (for interactive proof widgets)
require proofwidgets from git
  "https://github.com/leanprover-community/ProofWidgets4.git" @ "v0.0.62"

-- Require quote4 (transitive dep, now made direct to override version)
require Qq from git
  "https://github.com/leanprover-community/quote4" @ "v4.21.0"

-- Require SciLean (for scientific computing and machine learning)
-- require scilean from git
--  "https://github.com/lecopivo/SciLean.git" @ "master"

-- Define the executable target for metadata/dependency extraction
@[default_target]
lean_exe «extractDeclarations» where
  root := `ExtractDeclarations
  supportInterpreter := true

-- Define the executable target for syntax snippet extraction
lean_exe «extractAST» where
  root := `ExtractAST
  supportInterpreter := true
