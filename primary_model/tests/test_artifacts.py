"""
Tests for generalized artifact creation utilities.
"""

from pathlib import Path

import pandas as pd
import pytest

from primary_model.utils.artifacts import write_dataframe, write_markdown_protocol


def test_write_dataframe_creates_parents(tmp_path):
    """Ensure write_dataframe auto-creates missing intermediate directories."""
    deep_path = tmp_path / "some" / "nested" / "dir" / "data.csv"
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    
    assert not deep_path.parent.exists()
    
    write_dataframe(df, deep_path, index=False)
    
    assert deep_path.parent.exists()
    assert deep_path.exists()
    
    # Verify content
    recovered = pd.read_csv(deep_path)
    assert len(recovered) == 2
    assert "a" in recovered.columns


def test_write_markdown_protocol_creates_parents(tmp_path):
    """Ensure write_markdown_protocol auto-creates missing intermediate directories."""
    deep_path = tmp_path / "another" / "layer" / "report.md"
    lines = ["# Title", "", "Some content."]
    
    assert not deep_path.parent.exists()
    
    write_markdown_protocol(lines, deep_path)
    
    assert deep_path.parent.exists()
    assert deep_path.exists()
    
    # Verify content
    content = deep_path.read_text(encoding="utf-8")
    assert content == "# Title\n\nSome content."
