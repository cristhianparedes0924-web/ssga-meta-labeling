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

## 2026-03-12

### Decision

Use actionable primary events as the first secondary observations, define `meta_label` from the next-period realized primary-strategy net return, and implement secondary dataset construction before any secondary-model training.

### Reason

This keeps the first secondary step aligned with actual primary behavior instead of a proxy target, limits leakage risk by separating decision-time features from `t+1` outcomes, and keeps the rollout incremental.

### Impact

The first secondary dataset is built from `BUY` and `SELL` primary events by default, `meta_target_return` reflects the realized next-period primary-strategy net return implied at the decision date, and model training remains out of scope until this dataset contract is stable.
