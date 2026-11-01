"""Explorer checks: labeled unit arrays never enter real study artifacts."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DISPLAY = ROOT / "data/derived/coverage_explorer"


def preparation():
    assert (ROOT / "fmri_study_check/explorer_prepare.py").is_file(), (
        "Display preparation is missing"
    )
    from fmri_study_check import explorer_prepare

    return explorer_prepare


def viewer():
    assert (ROOT / "fmri_study_check/explorer.py").is_file(), "Artifact-only viewer is missing"
    from fmri_study_check import explorer

    return explorer


def test_support_strata_keep_unknown_and_empty_denominators_explicit():
    module = preparation()
    roi = np.array([True, True, True, True, False])
    x = np.array([-2, -2, 2, 2, 0])
    included = np.array([True, False, True, False, True])
    unknown = np.array([False, True, False, False, False])
    result = module.support_summary(roi, included, x, unknown)
    assert result["whole"]["total"] == 4
    assert result["whole"]["included"] == 2
    assert result["whole"]["fraction"] is None
    assert result["whole"]["lower"] == 0.5 and result["whole"]["upper"] == 0.75
    assert result["left"]["unknown"] == 1 and result["left"]["fraction"] is None
    assert result["right"]["fraction"] == 0.5
    assert result["midline"]["total"] == 0 and result["midline"]["fraction"] is None


def test_regional_means_preserve_order_scaling_and_exact_constant_signals():
    values = np.array([[1, 2, 3], [3, 2, 1], [7, 7, 7], [0, 2, 4]], dtype=np.int16).reshape(
        4, 1, 1, 3
    )
    labels = np.array([2, 2, 1, 0]).reshape(4, 1, 1)
    result = preparation().regional_series(values, labels, [1, 2], slope=2.0, intercept=5.0)
    assert result["1"] == [19.0, 19.0, 19.0]
    assert result["2"] == [9.0, 9.0, 9.0]
    values[0, 0, 0, 1] = 6
    changed = preparation().regional_series(values, labels, [2], slope=2.0, intercept=5.0)
    assert changed["2"] == [9.0, 13.0, 9.0]


def test_source_assets_join_by_id_and_reject_duplicate_or_mismatched_filename():
    records = [
        {"participant_id": p, "role": "spm8_mask", "source_filename": p + "_analysis_mask.nii.gz"}
        for p in ("sub-pixar124", "sub-pixar123")
    ]
    joined = preparation().source_index(records)
    assert (
        joined["sub-pixar123", "spm8_mask"]["source_filename"]
        == "sub-pixar123_analysis_mask.nii.gz"
    )
    with pytest.raises(ValueError, match="Duplicate"):
        preparation().source_index(records + records[:1])
    records[0]["source_filename"] = "sub-pixar123_analysis_mask.nii.gz"
    with pytest.raises(ValueError, match="participant"):
        preparation().source_index(records)


def sample_frames():
    return pd.DataFrame(
        {
            "original_index": [0, 1, 2],
            "time_seconds": [0.0, 2.0, 4.0],
            "framewise_displacement": [0.0, 0.5, 0.5001],
            "fd_defined": [False, True, True],
            "baseline_keep": [False, True, True],
            "censored_keep": [False, True, False],
            "baseline_exclusion_reasons": ["initial_frame", None, None],
            "censored_exclusion_reasons": ["initial_frame", None, "fd_above_threshold"],
        }
    )


def test_original_frame_and_fd_denominators_use_defined_observations_and_strict_boundary():
    result = preparation().frame_payload(sample_frames(), 2.0)
    assert result["fd"] == [None, 0.5, 0.5001]
    assert result["fd_stored"] == [0.0, 0.5, 0.5001]
    assert result["n_fd_observed"] == 2 and result["high_fd_count"] == 1
    assert result["time_seconds"] == [0.0, 2.0, 4.0]
    assert result["censored_keep"] == [False, True, False]
    assert result["censored_reason"][2] == "fd_above_threshold"


def test_bad_time_alignment_or_undefined_later_fd_blocks_display_preparation():
    frames = sample_frames()
    frames.loc[1, "time_seconds"] = 9.0
    with pytest.raises(ValueError, match="time"):
        preparation().frame_payload(frames, 2.0)
    frames = sample_frames()
    frames.loc[2, "framewise_displacement"] = np.nan
    with pytest.raises(ValueError, match="FD"):
        preparation().frame_payload(frames, 2.0)


def test_censoring_membership_cannot_silently_disagree_with_fixed_rule():
    frames = sample_frames()
    frames.loc[1, "censored_keep"] = False
    with pytest.raises(ValueError, match="censor"):
        preparation().frame_payload(frames, 2.0)


@pytest.mark.parametrize("relative", ["data/source", "outputs/eight", "investigations", "data"])
def test_preparation_cannot_write_to_scientific_inputs_or_their_ancestors(tmp_path, relative):
    with pytest.raises(ValueError, match="isolated"):
        preparation().prepare_explorer(tmp_path, tmp_path / relative)
    assert not (tmp_path / relative / "completed.json").exists()


def test_viewer_requires_completed_assets_without_starting_preparation(tmp_path):
    with pytest.raises(ValueError, match="prepared"):
        viewer().load_explorer(tmp_path, "sub-pixar123")


def test_json_embedding_does_not_allow_labels_to_terminate_script_elements():
    encoded = viewer().script_json({"name": "</script><script>alert(1)</script>", "value": 0})
    assert "</script>" not in encoded
    assert json.loads(encoded)["name"] == "</script><script>alert(1)</script>"
    assert json.loads(encoded)["value"] == 0


def test_real_viewer_payload_keeps_lineages_flat_signal_and_denominators_separate():
    module = viewer()
    if not (DISPLAY / "completed.json").exists():
        pytest.skip("Real display assets have not been prepared")
    data = module.load_explorer(DISPLAY, "sub-pixar124")
    r = next(r for r in data["regions"] if r["id"] == 14)
    assert r["tutorial"]["constant"] and len(set(r["tutorial"]["series"])) == 1
    assert r["tutorial"]["support"]["whole"]["total"] == 82
    assert r["archive"]["support"]["whole"]["total"] == 647
    assert data["motion"]["n_fd_observed"] == 167
    assert data["motion"]["fd"][0] is None
    assert data["primary_status"] == "analysis_unavailable"
    html = module.render_explorer(DISPLAY, "sub-pixar124")
    assert "coverage-data" in html and "coverage-canvas" in html


def test_changed_display_artifact_is_rejected_even_with_unchanged_marker(tmp_path):
    module = viewer()
    if not (DISPLAY / "completed.json").exists():
        pytest.skip("Real display assets have not been prepared")
    shutil.copyfile(DISPLAY / "completed.json", tmp_path / "completed.json")
    for filename in ("provenance.json", "failures.json"):
        shutil.copyfile(DISPLAY / filename, tmp_path / filename)
    original = DISPLAY / "scans/sub-pixar123"
    shutil.copytree(original, tmp_path / "scans/sub-pixar123")
    module.load_explorer(tmp_path, "sub-pixar123")
    (tmp_path / "scans/sub-pixar123/tutorial/mean.nii.gz").write_bytes(b"Changed test artifact")
    with pytest.raises(ValueError, match="artifact"):
        module.load_explorer(tmp_path, "sub-pixar123")


def test_app_explorer_reads_prepared_assets_without_scientific_work(monkeypatch):
    if not (DISPLAY / "completed.json").exists():
        pytest.skip("No real display assets prepared")
    from streamlit.testing.v1 import AppTest

    import fmri_study_check.pipeline as pipeline

    def forbidden(*args, **kwargs):
        raise AssertionError("Explorer navigation invoked scientific work")

    for name in ("fetch_adults", "extract_regions", "run_study"):
        monkeypatch.setattr(pipeline, name, forbidden)
    monkeypatch.setattr(preparation(), "prepare_explorer", forbidden)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert "Coverage Explorer" in app.radio(key="view").options
    app.radio(key="view").set_value("Coverage Explorer").run()
    for pid in ("sub-pixar123", "sub-pixar124", "sub-pixar125"):
        app.selectbox(key="scan").set_value(pid + "_task-pixar").run()
        assert not app.exception and not app.error
    app.radio(key="view").set_value("Analysis Comparison").run()
    app.selectbox(key="scan").set_value("sub-pixar124_task-pixar").run()
    assert any("unavailable" in w.value.lower() for w in app.warning)


def test_display_is_bound_to_selected_primary_results_and_checkpoint(tmp_path):
    module = viewer()
    if not (DISPLAY / "completed.json").exists():
        pytest.skip("No real display assets prepared")
    for original in (ROOT / "outputs/one", ROOT / "outputs/eight"):
        module.load_explorer(DISPLAY, "sub-pixar123", primary_bundle=original)
    for name in ("manifest.csv", "retention.csv"):
        shutil.copyfile(ROOT / "outputs/eight" / name, tmp_path / name)
    scan = "sub-pixar123_task-pixar"
    (tmp_path / "scans" / scan).mkdir(parents=True)
    for name in ("frames.csv", "raw_timeseries.csv"):
        shutil.copyfile(
            ROOT / "outputs/eight/scans" / scan / name, tmp_path / "scans" / scan / name
        )
    module.load_explorer(DISPLAY, "sub-pixar123", primary_bundle=tmp_path)
    retention = pd.read_csv(tmp_path / "retention.csv")
    selected = retention.scan_key.eq(scan)
    retention.loc[selected, "retained_volumes"] -= 1
    retention.to_csv(tmp_path / "retention.csv", index=False)
    with pytest.raises(ValueError, match="stale"):
        module.load_explorer(DISPLAY, "sub-pixar123", primary_bundle=tmp_path)
    shutil.copyfile(ROOT / "outputs/eight/retention.csv", tmp_path / "retention.csv")
    manifest = pd.read_csv(tmp_path / "manifest.csv")
    manifest.loc[manifest.scan_key.eq(scan), "processing_status"] = "changed_test_status"
    manifest.to_csv(tmp_path / "manifest.csv", index=False)
    with pytest.raises(ValueError, match="stale"):
        module.load_explorer(DISPLAY, "sub-pixar123", primary_bundle=tmp_path)
