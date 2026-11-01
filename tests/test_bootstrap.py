"""Safety checks for the fresh-checkout supplement bootstrap; labeled test files only."""

import importlib.util
import json
from pathlib import Path

import pytest


def module():
    script = Path(__file__).parents[1] / "tools/bootstrap_supplements.py"
    assert script.is_file(), "Portable supplement bootstrap is missing"
    spec = importlib.util.spec_from_file_location("supplement_bootstrap", script)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_snapshot_never_resets_a_changed_protected_file(tmp_path):
    m = module()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "source.txt").write_text("Labeled unit input")
    output = tmp_path / "supplement"
    output.mkdir()
    m.ensure_snapshot(tmp_path, output, [inputs], [])
    original = (output / "protected-before.json").read_bytes()
    (inputs / "source.txt").write_text("Changed unit input")
    with pytest.raises(ValueError, match="changed"):
        m.ensure_snapshot(tmp_path, output, [inputs], [])
    assert (output / "protected-before.json").read_bytes() == original


def test_snapshot_does_not_include_its_own_output_or_bytecode(tmp_path):
    m = module()
    (tmp_path / "kept.txt").write_text("Labeled unit input")
    output = tmp_path / "supplement"
    output.mkdir()
    (output / "generated.txt").write_text("Labeled derived output")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "unit.pyc").write_bytes(b"test")
    m.ensure_snapshot(tmp_path, output, [tmp_path], [])
    snapshot = json.loads((output / "protected-before.json").read_text())
    assert list(snapshot) == ["kept.txt"]


def test_existing_marker_must_verify_without_rebuilding(tmp_path):
    m = module()
    (tmp_path / "counts.csv").write_text("Labeled unit counts")
    marker = {"status": "complete", "artifacts": {"counts.csv": m.sha256(tmp_path / "counts.csv")}}
    (tmp_path / "completed.json").write_text(json.dumps(marker))
    assert m.verify_completed(tmp_path)
    (tmp_path / "counts.csv").write_text("Changed unit counts")
    with pytest.raises(ValueError, match="changed"):
        m.verify_completed(tmp_path)
