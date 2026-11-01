"""Artifact-only interface tests; never invoke acquisition or scientific computation."""

import io
import zipfile
from pathlib import Path

import pytest

from fmri_study_check.ui import export_archive, load_bundle

ROOT = Path(__file__).resolve().parents[1]


def test_app_loader_requires_a_finished_bundle(tmp_path):
    with pytest.raises(ValueError, match="completed"):
        load_bundle(tmp_path)


def test_artifact_zip_contains_csv_and_methods_without_source_images():
    directory = ROOT / "outputs" / "one"
    if not (directory / "completed.json").is_file():
        pytest.skip("Real bundle not acquired; no synthetic observations substituted")
    result = load_bundle(directory)
    assert result["manifest"].participant_id.tolist() == ["sub-pixar123"]
    with zipfile.ZipFile(io.BytesIO(export_archive(directory))) as archive:
        assert {
            "manifest.csv",
            "frame_exclusions.csv",
            "settings.json",
            "provenance.json",
        }.issubset(archive.namelist())
        assert not any(name.endswith(".nii.gz") for name in archive.namelist())
        assert all("fixture" not in name for name in archive.namelist())


def test_four_views_render_real_bundle_and_unavailable_scan_without_analysis(monkeypatch):
    directory = ROOT / "outputs" / "eight"
    if not (directory / "completed.json").is_file():
        pytest.skip("Eight-real-adult workflow has not finished")
    from streamlit.testing.v1 import AppTest

    import fmri_study_check.pipeline as pipeline

    def forbidden(*args, **kwargs):
        raise AssertionError("UI invoked expensive analysis")

    monkeypatch.setattr(pipeline, "run_study", forbidden)
    monkeypatch.setattr(pipeline, "fetch_adults", forbidden)
    monkeypatch.setattr(pipeline, "extract_regions", forbidden)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "fMRI Quality Explorer"
    for view in ["Scan Inspection", "Analysis Comparison", "Report", "Overview"]:
        app.radio(key="view").set_value(view).run()
        assert not app.exception
    app.radio(key="view").set_value("Analysis Comparison").run()
    app.selectbox(key="scan").set_value("sub-pixar124_task-pixar").run()
    assert not app.exception
    assert any("unavailable" in x.value.lower() for x in app.warning)
    assert any("14" in x.value for x in app.warning)
