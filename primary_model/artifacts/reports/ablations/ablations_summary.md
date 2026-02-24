# Ablation Summary

- Acceptance passed: `True`
- Variants evaluated: `10`
- Best Sharpe variant: `full_equal_weight_aggregation` (`0.933794`)
- Recommended aggregation from evidence: `equal_weight` (dynamic - equal Sharpe: `-0.095935`)

## Acceptance Checks
- `contribution_ranking_clear_and_reproducible`: `True`
- `no_hidden_dependence_undocumented`: `True`
- `composite_logic_justified_by_evidence_not_assumption`: `True`

## Findings
- `dynamic_underperforms_equal_weight`: `True`
- `single_indicator_outperforms_baseline`: `True`

## Outputs
- `\\wsl.localhost\Ubuntu\home\cristhian\Projects\ssga-meta-labeling\primary_model\artifacts\reports\ablations\ablations_variant_summary.csv`
- `\\wsl.localhost\Ubuntu\home\cristhian\Projects\ssga-meta-labeling\primary_model\artifacts\reports\ablations\ablations_leave_one_out.csv`
- `\\wsl.localhost\Ubuntu\home\cristhian\Projects\ssga-meta-labeling\primary_model\artifacts\reports\ablations\ablations_single_indicator.csv`
- `\\wsl.localhost\Ubuntu\home\cristhian\Projects\ssga-meta-labeling\primary_model\artifacts\reports\ablations\ablations_ranked_by_sharpe.csv`
- `\\wsl.localhost\Ubuntu\home\cristhian\Projects\ssga-meta-labeling\primary_model\artifacts\reports\ablations\ablations_assessment.json`