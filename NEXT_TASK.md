# Next Task

## Goal

Scaffold the secondary model structure without changing current primary-model behavior.

## Allowed scope

- Add conservative package structure under `src/metalabel/secondary/`
- Add placeholder modules, interfaces, and minimal documentation comments where needed
- Add non-invasive tests for importability or empty scaffolding only
- Add docs that explain intended secondary-model boundaries and dependencies on the primary baseline

## Do not change

- Do not change logic under `src/metalabel/primary/`
- Do not change current CLI command names or current primary workflow behavior
- Do not change `configs/primary.yaml` unless a future task explicitly requires secondary settings
- Do not overwrite current tracked outputs in `reports/results/`
- Do not alter the numerical behavior of the current primary baseline

## Deliverables

- A clear `secondary/` package skeleton with named modules for future development
- Safe placeholders for secondary-model inputs, labels/features, training, inference, and evaluation
- Imports and package structure that do not affect existing primary execution paths
- Minimal tests or checks, if added, that only validate the scaffold

## Done when

- The repo has a readable secondary-model skeleton
- Existing primary commands still behave the same based on code inspection and unchanged primary paths
- No primary logic, configs, tests, or tracked outputs were modified as part of the scaffolding task
