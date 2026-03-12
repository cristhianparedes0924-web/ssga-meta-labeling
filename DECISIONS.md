# Decisions

## 2026-03-12

### Decision

Freeze the current primary model as the stable baseline.

### Reason

Based on the current repo state, the primary model already has implemented pipeline code, configuration, tests, and tracked result artifacts. It is the only fully implemented strategy path in the repository today.

### Impact

Future work should treat current primary behavior as the reference implementation. Any intentional baseline change should be explicit, documented, and followed by regeneration of affected outputs.

## 2026-03-12

### Decision

Continue new development in `secondary/` while minimizing risk to `primary/`.

### Reason

The repository already separates `src/metalabel/primary/` from `src/metalabel/secondary/`, and `secondary/` currently appears to be only a placeholder. Using that boundary reduces risk to the working primary pipeline.

### Impact

New experimentation should be added under `secondary/` first, with conservative interfaces and no silent changes to primary commands, configs, or reports. This keeps the baseline stable while opening a safe path for secondary-model work.
