"""Generated image arrays are labeled test inputs, never study observations."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from fmri_study_check.pipeline import fetch_adults, inspect_scan, verify_bundle


def make_scan(tmp_path, n=5, rows=5, tr=2, units="sec"):
    image = tmp_path / "sub-fixture_task-test_desc-preproc_bold.nii.gz"
    table = tmp_path / "sub-fixture_task-test_desc-confounds_regressors.tsv"
    img = nib.Nifti1Image(np.ones((2, 2, 2, n), dtype=np.float32), np.eye(4))
    img.header.set_zooms((1, 1, 1, tr))
    img.header.set_xyzt_units("mm", units)
    nib.save(img, image)
    df = pd.DataFrame(
        {
            c: np.linspace(1, 2, rows)
            for c in [
                "trans_x",
                "trans_y",
                "trans_z",
                "rot_x",
                "rot_y",
                "rot_z",
                "csf",
                "white_matter",
                "cosine00",
                "framewise_displacement",
            ]
        }
    )
    df.to_csv(table, sep="\t", index=False)
    return dict(
        scan_key="sub-fixture_task-test",
        participant_id="sub-fixture",
        task="test",
        session=None,
        acquisition=None,
        run=None,
        echo=None,
        image_path=str(image),
        confounds_path=str(table),
        phenotype={"Age": 27},
    )


def test_alignment_mismatch_produces_actionable_issue(tmp_path):
    row, df, issues, mean = inspect_scan(make_scan(tmp_path, rows=4), 2)
    error = next(x for x in issues if x["rule"] == "image_confound_alignment")
    assert error["severity"] == "error"
    assert "5" in error["evidence"] and "4" in error["evidence"]
    assert error["file_location"].endswith(".tsv")
    assert error["possible_explanations"] and error["recommended_action"]
    assert row["processing_status"] == "integrity_blocked"


def test_unknown_time_units_use_recorded_dataset_fallback(tmp_path):
    row, _, issues, _ = inspect_scan(make_scan(tmp_path, tr=1, units="unknown"), 2)
    assert row["sampling_interval_seconds"] == 2
    assert row["header_time_zoom"] == 1
    assert row["duration_seconds"] == 10
    assert row["timing_source"] == "fetcher_metadata_fallback"
    assert any(x["rule"] == "missing_header_timing" and x["severity"] == "review" for x in issues)
    assert not [x for x in issues if x["severity"] == "error"]


def test_conflicting_valid_header_timing_blocks(tmp_path):
    row, _, issues, _ = inspect_scan(make_scan(tmp_path, tr=3), 2)
    assert row["processing_status"] == "integrity_blocked"
    assert any(x["rule"] == "timing_agreement" and x["severity"] == "error" for x in issues)


def test_duplicate_tsv_headers_are_detected_before_pandas_mangling(tmp_path):
    scan = make_scan(tmp_path)
    path = Path(scan["confounds_path"])
    path.write_text(path.read_text().replace("trans_y", "trans_x", 1))
    _, _, issues, _ = inspect_scan(scan, 2)
    assert any(x["rule"] == "unique_confound_columns" for x in issues)


def test_adult_alias_adapter_records_only_specific_validation_error(tmp_path):
    calls = []

    def fetcher(**kwargs):
        calls.append(kwargs)
        if kwargs["age_group"] == "adults":
            raise ValueError(
                "'age_group' must be one of ('both', 'child', 'adult').\n'adults' was provided."
            )
        return "actual-bunch"

    result, event = fetch_adults(1, tmp_path, fetcher=fetcher)
    assert result == "actual-bunch"
    assert [x["age_group"] for x in calls] == ["adults", "adult"]
    assert all(x["reduce_confounds"] is False for x in calls)
    assert event["requested_age_group"] == "adults"
    assert event["effective_age_group"] == "adult"
    assert event["compatibility_error"]

    def other_failure(**kwargs):
        raise ValueError("Unreadable downloaded participant file")

    with pytest.raises(ValueError, match="Unreadable"):
        fetch_adults(1, tmp_path, fetcher=other_failure)


def test_incomplete_bundle_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="completed"):
        verify_bundle(tmp_path, check_source_hashes=False)


def test_acquisition_failure_still_delivers_honest_report(tmp_path, monkeypatch):
    import fmri_study_check.pipeline as pipeline

    def failed_fetch(*args, **kwargs):
        raise ConnectionError("HTTP 503 from public source")

    monkeypatch.setattr(pipeline, "fetch_adults", failed_fetch)
    output = tmp_path / "out"
    assert not pipeline.run_study(1, tmp_path / "source", output, tmp_path / "cache")
    assert "HTTP 503" in (output / "report.html").read_text()
    assert (output / "failure.json").is_file()
    assert pipeline.verify_bundle(output, check_source_hashes=False)["status"] == "partial"


def test_corrupted_real_excerpt_is_isolated_from_study_report(tmp_path):
    from fmri_study_check.fixtures import create_corrupted_fixture

    record = make_scan(tmp_path, n=8, rows=8)
    fixture_dir = tmp_path / "fixtures" / "corrupted"
    result = create_corrupted_fixture(record, fixture_dir, 2)
    assert result["data_kind"] == "fixture"
    assert result["image_volumes"] == 5 and result["confound_rows"] == 4
    errors = pd.read_csv(fixture_dir / "issues.csv")
    assert errors.loc[errors.rule == "image_confound_alignment", "severity"].iloc[0] == "error"
    import json

    metadata = json.loads((fixture_dir / "fixture.json").read_text())
    assert metadata["source_image_sha256"]
    assert metadata["exclude_from_real_data_report"] is True


def test_rerun_preserves_prior_outputs_outside_current_bundle(tmp_path):
    from fmri_study_check.pipeline import prepare_output

    output = tmp_path / "outputs" / "one"
    output.mkdir(parents=True)
    (output / "report.html").write_text("Prior real report")
    (output / "scans").mkdir()
    (output / "scans" / "old.csv").write_text("prior values")
    archived = prepare_output(output)
    assert (archived / "report.html").read_text() == "Prior real report"
    assert (archived / "scans" / "old.csv").is_file()
    assert list(output.iterdir()) == []


@pytest.mark.parametrize("destination_kind", ["source_parent", "source_ancestor", "linked_file"])
def test_fixture_rejects_writes_overlapping_preserved_source(tmp_path, destination_kind):
    from fmri_study_check.fixtures import create_corrupted_fixture
    from fmri_study_check.pipeline import sha256

    source = tmp_path / "source"
    source.mkdir()
    record = make_scan(source, n=8, rows=8)
    before = [sha256(record[field]) for field in ("image_path", "confounds_path")]
    destination = source if destination_kind == "source_parent" else tmp_path
    if destination_kind == "linked_file":
        destination = tmp_path / "fixture"
        destination.mkdir()
        (destination / Path(record["image_path"]).name).symlink_to(record["image_path"])
    with pytest.raises(ValueError, match="source"):
        create_corrupted_fixture(record, destination, 2)
    assert [sha256(record[field]) for field in ("image_path", "confounds_path")] == before


def test_missing_raw_cache_artifact_is_recomputed(tmp_path):
    from types import SimpleNamespace

    from fmri_study_check.pipeline import extract_regions, sha256

    rng = np.random.default_rng(4)
    image_path = tmp_path / "sub-test_task-fixture_bold.nii.gz"
    nib.save(
        nib.Nifti1Image(rng.normal(size=(2, 2, 2, 5)).astype(np.float32), np.eye(4)), image_path
    )
    labels = np.ones((2, 2, 2), dtype=np.int16)
    labels[1] = 2
    atlas = SimpleNamespace(
        maps=nib.Nifti1Image(labels, np.eye(4)),
        lut=pd.DataFrame({"index": [0, 1, 2], "name": ["Background", "One", "Two"]}),
    )
    first, regions, hit = extract_regions(
        image_path, atlas, tmp_path / "cache", sha256(image_path), "test-atlas"
    )
    assert not hit
    next((tmp_path / "cache").rglob("signals.npy")).unlink()
    recovered, recovered_regions, hit = extract_regions(
        image_path, atlas, tmp_path / "cache", sha256(image_path), "test-atlas"
    )
    assert not hit
    np.testing.assert_array_equal(first, recovered)
    pd.testing.assert_frame_equal(regions, recovered_regions)


def test_actual_transient_analysis_failure_bundle_recovers_on_same_command(tmp_path, monkeypatch):
    import json

    import fmri_study_check.pipeline as pipeline

    root = Path(__file__).resolve().parents[1]
    if not (root / "outputs" / "one" / "completed.json").exists():
        pytest.skip("Real source not acquired; no substitute observations")
    output, cache = tmp_path / "out", tmp_path / "cache"
    original = pipeline.extract_regions

    def temporary_failure(*args, **kwargs):
        raise OSError("Temporary regional extraction failure")

    monkeypatch.setattr(pipeline, "extract_regions", temporary_failure)
    assert not pipeline.run_study(1, root / "data" / "source", output, cache)
    assert "analysis_failure" in pd.read_csv(output / "issues.csv").rule.tolist()
    assert json.loads((output / "summary.json").read_text())["status"] == "partial"
    monkeypatch.setattr(pipeline, "extract_regions", original)
    assert pipeline.run_study(1, root / "data" / "source", output, cache)
    assert pipeline.verify_bundle(output)["all_analyses_complete"]
    assert list((tmp_path / ".history").glob("*/report.html"))
