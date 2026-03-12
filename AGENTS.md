# AGENTS.md

This file gives coding agents and collaborators the minimum project-specific instructions needed to work safely in this repository.

## Session Startup Protocol

Before making any plan or edits, read these root files:

- `AGENTS.md`
- `PROJECT_STATE.md`
- `NEXT_TASK.md`
- `DECISIONS.md`

Then:

1. summarize the current project state
2. summarize the current task
3. identify which files are in scope
4. avoid editing outside the defined scope
5. treat the current primary model as the protected baseline unless the task explicitly says otherwise

## Task Lifecycle And Control Files

- The project planner is the human user and/or the ChatGPT Project context. Codex is the executor, not the strategic decision-maker, and must not silently redefine project direction beyond the currently approved task.
- Before planning or editing, always read `AGENTS.md`, `PROJECT_STATE.md`, `NEXT_TASK.md`, and `DECISIONS.md`.
- Treat `NEXT_TASK.md` as the currently approved assignment. At the end of every completed task, update it. If the task is complete and no new task has been explicitly approved, set it to an `Awaiting planning` state with the completed task, a short repo-status paragraph, 2-4 proposed bounded next tasks, and one recommended next task.
- Update `PROJECT_STATE.md` only when repo state materially changes, and also before final handoff when the completed task changed the real project state. Do not update it for trivial refactors, comment-only edits, or formatting-only changes.
- `DECISIONS.md` is append-only. Add an entry only for durable decisions that affect future work, and include `Date`, `Decision`, `Reason`, and `Impact`.
- Treat the current primary model as the protected baseline unless `NEXT_TASK.md` explicitly authorizes changing it.
- At end of task, report files changed, tests/checks run, whether `NEXT_TASK.md` was updated, whether `PROJECT_STATE.md` was updated, whether `DECISIONS.md` was updated, and any proposed next tasks if planning is required.

## Project Scope

- This repository contains the implemented primary model and the project structure for continued secondary-model development.
- Preserve the numerical behavior of the primary model unless the user explicitly asks for a model change.

## Environment Setup

Work from the repository root:

```powershell
cd "C:\Users\crist\OneDrive\Documents\meta-labeling-project"
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The main package entry point is:

```powershell
python -m metalabel.cli <command>
```

## Data Assumptions

Required raw input files in `data/raw/`:

- `spx.xlsx`
- `bcom.xlsx`
- `treasury_10y.xlsx`
- `corp_bonds.xlsx`

Important assumption:

- `treasury_10y` is treated as a yield-level input series.
- The code converts that yield level into a duration-based total return proxy during processing.

## Safe Commands

Safe validation and workflow commands from the repo root:

```powershell
python -m metalabel.cli run-self-tests
python -m pytest
python -m metalabel.cli prepare-data
python -m metalabel.cli data-qc
python -m metalabel.cli run-primary-v1
python -m metalabel.cli run-benchmarks
python -m metalabel.cli run-robustness
python -m metalabel.cli run-walk-forward
python -m metalabel.cli run-monthly-cv
python -m metalabel.cli run-all
```

Thin wrappers also exist in `scripts/`, but package entry points are preferred.

## Output Locations

- Raw inputs: `data/raw/`
- Cleaned canonical data: `data/clean/`
- Plots: `reports/assets/`
- CSV, HTML, markdown, and validation outputs: `reports/results/`

These folders are intentionally tracked because collaborators may want to inspect the current cleaned data and latest model results directly.

## Project Conventions

- Do not change strategy logic without documenting the change in the relevant repository docs and regenerating the affected outputs.
- Keep main runtime configuration in `configs/primary.yaml`.
- Preserve canonical CLI command names and canonical output file names unless the user explicitly asks for an interface change.
- Prefer package entry points such as `python -m metalabel.cli ...` over ad hoc one-off commands.
- Use clear commit messages in imperative mood with the format `<verb> <area> <change>`, such as `Update README setup instructions` or `Refactor primary signal pipeline`.
- Keep one main change per commit when practical. If model logic changes, say that explicitly in the commit message.
- Validate normal code changes with `python -m metalabel.cli run-self-tests` and `python -m pytest`.
- For model logic, reporting, or validation changes, also rerun the affected workflows and refresh the tracked outputs they produce.

## What Not To Modify Casually

- Do not modify `data/raw/` unless the user explicitly asks for data changes.
- Do not silently rename CLI commands, output file names, or folder layout.
- Do not move primary configuration out of `configs/primary.yaml` unless explicitly requested.
- Do not change the tracked result/report locations unless the user asks for a repository layout change.

## Validation Expectations

For documentation-only changes:

- no code run is required

For package/code changes:

```powershell
python -m metalabel.cli run-self-tests
python -m pytest
```

For changes that affect model logic, outputs, or reports:

```powershell
python -m metalabel.cli run-primary-v1
python -m metalabel.cli run-benchmarks
```

If validation workflows are touched, also rerun:

```powershell
python -m metalabel.cli run-robustness
python -m metalabel.cli run-walk-forward
python -m metalabel.cli run-monthly-cv
```

## Collaboration Notes

- Prefer small, explicit changes over broad rewrites.
- Keep README instructions aligned with the actual commands that work.
- If behavior changes, say so clearly and regenerate the tracked outputs that depend on that behavior.
- Keep commit history easy to scan by separating docs, code, config, and regenerated result updates whenever that separation is practical.
