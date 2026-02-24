# Primary Model Artifacts Directory

This folder (`artifacts/`) contains all the dynamically generated outputs, raw test data caches, and run artifacts produced during local development and testing.

## Contents Overview
- `data/`: Temporary, clean, and raw data files synthesized during isolated workflows.
- `reports/`: Exported metric summaries, CSV sheets, and validation output.
  - Examples include outputs from `walk_forward`, `robustness`, and full `reproducibility` scripts.

## Version Control Policy
Most subfolders and files inside `artifacts/` are **not source code** and should heavily prioritize being git-ignored to prevent bloating repository states.
- **IGNORED (Recommended):** Large CSV files, raw matrices, generated HTML, specific intermediate states.
- **VERSIONED (If necessary):** Critical final manifests (e.g. `validation_manifest.json`) or static setup scripts, though these should ideally live outside the immediate heavy-data footprint.
