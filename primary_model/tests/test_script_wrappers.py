"""
Tests for script wrappers and orchestration entrypoints.
Validates that research modules expose exactly the entrypoints the CLI scripts expect.
"""

import sys

def test_research_entrypoints_exist():
    """Verify that research scripts still expose the expected run_experiment calls."""
    import primary_model.research.walk_forward
    import primary_model.research.robustness
    import primary_model.research.validation_suite
    import primary_model.research.m1_baseline
    import primary_model.research.signal_validation
    import primary_model.research.ablations
    import primary_model.research.decision_diagnostics
    import primary_model.research.m1_readiness
    
    assert callable(primary_model.research.walk_forward.run_experiment)
    assert callable(primary_model.research.robustness.run_experiment)
    assert callable(primary_model.research.validation_suite.run_experiment)
    assert callable(primary_model.research.m1_baseline.run_experiment)
    assert callable(primary_model.research.signal_validation.run_experiment)
    assert callable(primary_model.research.ablations.run_experiment)
    assert callable(primary_model.research.decision_diagnostics.run_experiment)
    assert callable(primary_model.research.m1_readiness.run_experiment)

def test_cli_wrappers_import_safe():
    """Verify that the thin CLI wrappers can be imported without executing."""
    try:
        import scripts.run_walk_forward as run_walk_forward
        import scripts.run_robustness as run_robustness
        import scripts.run_validation_suite as run_validation_suite
        import scripts.run_m1_baseline as run_m1_baseline
        import scripts.run_signal_validation as run_signal_validation
        import scripts.run_ablations as run_ablations
        import scripts.run_decision_diagnostics as run_decision_diagnostics
        import scripts.run_m1_readiness as run_m1_readiness
        import scripts.run_modes as run_modes
    except ImportError as e:
        assert False, f"CLI wrapper import failed: {e}"
    assert run_walk_forward is not None
    assert run_robustness is not None
    assert run_validation_suite is not None
    assert run_m1_baseline is not None
    assert run_signal_validation is not None
    assert run_ablations is not None
    assert run_decision_diagnostics is not None
    assert run_m1_readiness is not None
    assert run_modes is not None
        
def test_cli_parsers_valid():
    """Verify that the parser building functions inside the CLI wrappers successfully compile."""
    from scripts.run_walk_forward import parse_args as wf_args
    from scripts.run_robustness import parse_args as rob_args
    from scripts.run_validation_suite import parse_args as val_args
    from scripts.run_m1_baseline import parse_args as m1_args
    from scripts.run_signal_validation import parse_args as sig_args
    from scripts.run_ablations import parse_args as ab_args
    from scripts.run_decision_diagnostics import parse_args as dec_args
    from scripts.run_m1_readiness import parse_args as ready_args
    
    # We replace sys.argv temporarily to avoid arg parsing errors halting pytest
    old_argv = sys.argv
    sys.argv = ['script_name']
    
    try:
        assert wf_args() is not None
        assert rob_args() is not None
        assert val_args() is not None
        assert m1_args() is not None
        assert sig_args() is not None
        assert ab_args() is not None
        assert dec_args() is not None
        assert ready_args() is not None
    finally:
        sys.argv = old_argv
