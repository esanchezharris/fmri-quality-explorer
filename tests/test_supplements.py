"""Artifact-only supplement checks; synthetic observations are never added to study outputs."""

import io
import json
import shutil
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "investigations/eight-adult-coverage"


def supplements():
    assert (ROOT / "fmri_study_check/supplements.py").is_file(), "Supplement loader is missing"
    from fmri_study_check import supplements as module

    return module


def test_unfinished_supplement_cannot_be_presented_as_finished(tmp_path):
    with pytest.raises(ValueError, match="completed"):
        supplements().load_supplement(tmp_path)


def test_supplement_rejects_artifact_paths_outside_its_directory(tmp_path):
    (tmp_path / "completed.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "data_kind": "real_supplement",
                "status": "complete",
                "artifacts": {"../outside.csv": "bad"},
            }
        )
    )
    with pytest.raises(ValueError, match="artifact"):
        supplements().load_supplement(tmp_path)


def test_completed_real_supplement_exports_no_source_images_and_detects_tampering(tmp_path):
    module = supplements()
    if not (AUDIT / "completed.json").exists():
        pytest.skip("Real coverage audit not yet completed; no substitute observations")
    result = module.load_supplement(AUDIT)
    assert set(result["participant_summary"].participant_id) == {
        f"sub-pixar{i}" for i in range(123, 131)
    }
    with zipfile.ZipFile(io.BytesIO(module.supplement_archive(AUDIT))) as archive:
        assert {
            "report.html",
            "participant_summary.csv",
            "region_coverage.csv",
            "provenance.json",
        }.issubset(archive.namelist())
        assert not any(
            "downloads/" in name or name.endswith(".nii.gz") for name in archive.namelist()
        )
    marker = json.loads((AUDIT / "completed.json").read_text())
    (tmp_path / "completed.json").write_text(
        json.dumps({**marker, "artifacts": {"report.html": marker["artifacts"]["report.html"]}})
    )
    (tmp_path / "report.html").write_text("Deliberately changed test artifact")
    with pytest.raises(ValueError, match="artifact"):
        module.load_supplement(tmp_path)


def test_app_supplement_navigation_is_read_only_and_keeps_primary_status(monkeypatch):
    if not (AUDIT / "completed.json").exists():
        pytest.skip("Real supplement not ready; no substitute observations")
    from streamlit.testing.v1 import AppTest

    import fmri_study_check.pipeline as pipeline

    def forbidden(*args, **kwargs):
        raise AssertionError("Supplement navigation invoked scientific computation")

    for name in ("run_study", "fetch_adults", "extract_regions"):
        monkeypatch.setattr(pipeline, name, forbidden)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert "Coverage Supplement" in app.radio(key="view").options
    app.radio(key="view").set_value("Coverage Supplement").run()
    assert not app.exception
    assert any("separate" in item.value.lower() for item in app.info)
    for participant in ("sub-pixar123", "sub-pixar124"):
        app.selectbox(key="scan").set_value(participant + "_task-pixar").run()
        assert not app.exception
        assert any("coverage" in item.value.lower() for item in app.warning)
    app.radio(key="view").set_value("Analysis Comparison").run()
    assert not app.exception
    assert any("unavailable" in item.value.lower() for item in app.warning)


def test_warm_navigation_rejects_changed_derived_artifact(tmp_path, monkeypatch):
    """A cached UI must not hide an on-disk integrity failure after its first read."""
    module = supplements()
    if not (AUDIT / "completed.json").exists():
        pytest.skip("Real coverage audit not ready; no substitute observations")
    from streamlit.testing.v1 import AppTest

    marker = json.loads((AUDIT / "completed.json").read_text())
    for relative in ["completed.json", *marker["artifacts"]]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(AUDIT / relative, target)
    original_loader = module.load_supplement
    monkeypatch.setattr(module, "load_supplement", lambda directory: original_loader(tmp_path))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    app.radio(key="view").set_value("Coverage Supplement").run()
    assert not app.exception and not app.error
    (tmp_path / "report.html").write_text("Deliberately modified copied test artifact")
    app.selectbox(key="scan").set_value("sub-pixar124_task-pixar").run()
    assert not app.exception
    assert any("artifact" in item.value.lower() for item in app.error)


def test_failed_rerun_preserves_history_exports_only_current_evidence_and_hides_stale_figure(
    tmp_path, monkeypatch
):
    """Exercise a missing-asset rerun on copied real outputs; never alter an observation."""
    module = supplements()
    if not (AUDIT / "completed.json").exists():
        pytest.skip("Real coverage audit not ready; no substitute observations")
    import importlib.util
    import sys

    from streamlit.testing.v1 import AppTest

    monkeypatch.syspath_prepend(str(AUDIT))
    marker = json.loads((AUDIT / "completed.json").read_text())
    for relative in ["completed.json", *marker["artifacts"]]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(AUDIT / relative, target)
    assets = json.loads((tmp_path / "download_manifest.json").read_text())
    for record in assets:
        if record["participant_id"] == "sub-pixar123" and record["role"] == "spm8_mask":
            record["status"] = "unavailable"  # Test-only acquisition status, not source data.
    (tmp_path / "download_manifest.json").write_text(json.dumps(assets))
    spec = importlib.util.spec_from_file_location("copied_coverage_audit", AUDIT / "audit.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    monkeypatch.setattr(audit, "OUT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["audit.py"])

    def forbidden(*args, **kwargs):
        raise AssertionError("Verified cached scans unexpectedly repeated image analysis")

    monkeypatch.setattr(audit, "inspect_scan", forbidden)
    import verify_independently

    monkeypatch.setattr(verify_independently, "OUT", tmp_path)
    assert audit.main() is True  # Partial status, with seven real scans retained.
    result = module.load_supplement(tmp_path)
    assert result["status"] == "partial"
    assert len(result["participant_summary"]) == 7
    active = tmp_path / "scans/sub-pixar123"
    assert [p.name for p in active.iterdir()] == ["failure.json"]
    assert list((tmp_path / "attempts").glob("sub-pixar123-*/completed.json"))
    with zipfile.ZipFile(io.BytesIO(module.supplement_archive(tmp_path))) as archive:
        failed_files = [
            name for name in archive.namelist() if name.startswith("scans/sub-pixar123/")
        ]
        assert failed_files == ["scans/sub-pixar123/failure.json"]
        assert "scans/sub-pixar124/region_coverage.csv" in archive.namelist()
    original_loader = module.load_supplement
    monkeypatch.setattr(module, "load_supplement", lambda directory: original_loader(tmp_path))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    app.radio(key="view").set_value("Coverage Supplement").run()
    assert not app.exception and not app.error
    assert any(
        "coverage audit unavailable for sub-pixar123" in item.value.lower() for item in app.warning
    )
    assert len(app.get("image")) == 1  # Heatmap only; the existing real scan figure is suppressed.
