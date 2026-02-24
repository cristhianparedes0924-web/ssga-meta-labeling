"""
Tests for architecture scaffolds.
Validates that new packages are import-safe and that their paths cleanly resolve.
"""

from pathlib import Path

def test_utils_paths_resolve():
    """Verify that the utils package exposes and resolves paths properly."""
    from primary_model.utils.paths import get_project_root, get_artifacts_root
    
    project_root = get_project_root()
    assert project_root.exists() and project_root.is_dir()
    
    artifacts_root = get_artifacts_root()
    assert artifacts_root.name == "artifacts"

def test_config_loader_safe():
    """Verify config_loader loads without errors and provides deep_merge_dicts."""
    from primary_model.utils.config_loader import deep_merge_dicts
    
    base = {"a": {"b": 1}, "c": 3}
    override = {"a": {"b": 2, "d": 4}}
    result = deep_merge_dicts(base, override)
    
    assert result == {"a": {"b": 2, "d": 4}, "c": 3}

def test_analytics_scaffolds_import():
    """Verify analytics package modules import cleanly."""
    import primary_model.analytics.performance
    import primary_model.analytics.diagnostics
    import primary_model.analytics.plots
    
    assert hasattr(primary_model.analytics.performance, "perf_table")
    assert hasattr(primary_model.analytics.diagnostics, "strategy_return_table")

def test_benchmarks_scaffolds_import():
    """Verify benchmarks package modules import cleanly."""
    import primary_model.benchmarks.static
    import primary_model.benchmarks.trend
    import primary_model.benchmarks.tactical
    import primary_model.benchmarks.evaluate
    
    assert hasattr(primary_model.benchmarks.static, "weights_equal_weight")

def test_research_scaffolds_import():
    """Verify research package modules import cleanly."""
    import primary_model.research.walk_forward
    import primary_model.research.robustness
    import primary_model.research.validation_suite
    import primary_model.research.ablations
    import primary_model.research.sensitivity
    import primary_model.research.subperiods
    
    assert hasattr(primary_model.research.walk_forward, "run_experiment")

def test_scripts_common_import():
    """Verify scripts common loaded properly."""
    from scripts._common import get_config_parser_args
    assert callable(get_config_parser_args)
