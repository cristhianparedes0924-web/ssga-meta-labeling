"""
Tests for native research entrypoints.
Validates that the execution logic properly processes standard python invocations.
"""

from pathlib import Path
import pytest

from primary_model.research.walk_forward import run_experiment as wf_run
from primary_model.research.robustness import run_experiment as rob_run
from primary_model.research.validation_suite import run_experiment as val_run
from primary_model.research.validation_suite import _run_and_log_native


def test_validation_suite_native_runner_captures_errors(tmp_path):
    """Ensure our native orchestrator catches, dumps stack traces, and does not crash out early."""
    
    def bad_function():
        raise ValueError("Simulated crash")
        
    log_file = tmp_path / "crash.log"
    success = _run_and_log_native("bad_step", bad_function, log_file)
    
    assert success is False
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "ValueError: Simulated crash" in content
    assert "exit_code=1" in content


def test_validation_suite_native_runner_success(tmp_path):
    """Ensure successful calls propagate cleanly."""
    
    def good_function(a, b):
        return a + b
        
    log_file = tmp_path / "success.log"
    success = _run_and_log_native("good_step", good_function, log_file, 2, 3)
    
    assert success is True
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "exit_code=0" in content
