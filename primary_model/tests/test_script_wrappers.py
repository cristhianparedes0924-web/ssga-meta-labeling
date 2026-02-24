"""
Tests for script wrappers and orchestration entrypoints.
Validates that research modules expose exactly the entrypoints the CLI scripts expect.
"""

import sys
from pathlib import Path

def test_research_entrypoints_exist():
    """Verify that research scripts still expose the expected run_experiment calls."""
    import primary_model.research.walk_forward
    import primary_model.research.robustness
    import primary_model.research.validation_suite
    
    assert callable(primary_model.research.walk_forward.run_experiment)
    assert callable(primary_model.research.robustness.run_experiment)
    assert callable(primary_model.research.validation_suite.run_experiment)

def test_cli_wrappers_import_safe():
    """Verify that the thin CLI wrappers can be imported without executing."""
    try:
        import scripts.run_walk_forward
        import scripts.run_robustness
        import scripts.run_validation_suite
        import scripts.run_modes
    except ImportError as e:
        assert False, f"CLI wrapper import failed: {e}"
        
def test_cli_parsers_valid():
    """Verify that the parser building functions inside the CLI wrappers successfully compile."""
    from scripts.run_walk_forward import parse_args as wf_args
    from scripts.run_robustness import parse_args as rob_args
    from scripts.run_validation_suite import parse_args as val_args
    
    # We replace sys.argv temporarily to avoid arg parsing errors halting pytest
    old_argv = sys.argv
    sys.argv = ['script_name']
    
    try:
        assert wf_args() is not None
        assert rob_args() is not None
        assert val_args() is not None
    finally:
        sys.argv = old_argv
