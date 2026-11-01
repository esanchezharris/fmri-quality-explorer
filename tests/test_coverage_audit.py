"""Small labeled test arrays only; no test observations enter the coverage supplement."""

import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def metrics():
    path = ROOT / "investigations/eight-adult-coverage/metrics.py"
    assert path.is_file(), "Coverage measurement implementation is missing"
    spec = importlib.util.spec_from_file_location("coverage_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_varying_voxel_denominator_does_not_confuse_constant_regional_mean():
    # Catch counting background, dividing by all voxels, or equating a constant mean with no variation.
    labels = np.array([1, 1, 2, 2, 0]).reshape(5, 1, 1)
    values = np.array([[1, 2, 3], [9, 9, 9], [0, 1, 2], [2, 1, 0], [7, 7, 7]])
    rows, varying, means = metrics().region_profile(values.reshape(5, 1, 1, 3), labels, [1, 2, 3])
    one, two, absent = rows
    assert one["grid_voxels"] == 2 and one["grid_varying_voxels"] == 1
    assert one["grid_varying_fraction"] == 0.5
    assert not one["regional_mean_constant"]
    assert two["grid_varying_voxels"] == 2 and two["regional_mean_constant"]
    assert absent["grid_voxels"] == 0 and absent["grid_varying_fraction"] is None
    assert means[0].tolist() == [5.0, 5.5, 6.0]
    assert means[1].tolist() == [1.0, 1.0, 1.0]
    assert varying.sum() == 3


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_observations_block_coverage_measurement(bad):
    values = np.ones((2, 1, 1, 3))
    values[1, 0, 0, 2] = bad
    with pytest.raises(ValueError, match="finite"):
        metrics().region_profile(values, np.ones((2, 1, 1)), [1])


def test_full_atlas_denominator_keeps_out_of_grid_locations():
    # Native atlas has four voxels; a shifted two-voxel image includes only x=1 and x=2.
    atlas = nib.Nifti1Image(np.ones((4, 1, 1), dtype=np.int16), np.eye(4))
    affine = np.eye(4)
    affine[0, 3] = 1
    grid = nib.Nifti1Image(np.zeros((2, 1, 1), dtype=np.int16), affine)
    support = np.array([0, 1], dtype=np.uint8).reshape(2, 1, 1)
    row = metrics().reference_profile(atlas, grid, support, [1], support)[0]
    assert row["atlas_voxels"] == 4
    assert row["atlas_in_grid_voxels"] == 2
    assert row["atlas_varying_voxels"] == 1
    assert row["atlas_mask_voxels"] == 1
    assert row["atlas_mask_fraction"] == 0.25


def test_mask_grid_and_binary_validation_prevents_misaligned_overlap():
    labels = np.ones((2, 1, 1))
    values = np.ones((2, 1, 1, 3))
    for mask in [np.ones((3, 1, 1)), np.full((2, 1, 1), 0.5)]:
        with pytest.raises(ValueError, match="mask"):
            metrics().region_profile(values, labels, [1], mask)


def test_undefined_mask_locations_are_unknown_and_never_filled_as_excluded():
    # Catch silently treating a missing mask value as outside brain or dividing by observed rows only.
    labels = np.ones((2, 1, 1), dtype=np.int16)
    values = np.ones((2, 1, 1, 3))
    mask = np.array([1.0, np.nan]).reshape(2, 1, 1)
    row = metrics().region_profile(values, labels, [1], mask)[0][0]
    assert row["grid_mask_unknown_voxels"] == 1
    assert row["grid_mask_fraction"] is None
    assert row["grid_mask_fraction_lower"] == 0.5
    assert row["grid_mask_fraction_upper"] == 1.0
    atlas = nib.Nifti1Image(labels, np.eye(4))
    image = nib.Nifti1Image(np.zeros((2, 1, 1), dtype=np.int16), np.eye(4))
    reference = metrics().reference_profile(atlas, image, np.zeros((2, 1, 1)), [1], mask)[0]
    assert reference["atlas_mask_unknown_voxels"] == 1
    assert reference["atlas_mask_fraction"] is None
    assert reference["atlas_mask_fraction_lower"] == 0.5
    assert reference["atlas_mask_fraction_upper"] == 1.0


def test_asset_join_uses_identifiers_and_rejects_duplicates_or_mismatch():
    records = [
        {
            "participant_id": "sub-pixar124",
            "role": "mask",
            "source_filename": "sub-pixar124_mask.nii.gz",
        },
        {
            "participant_id": "sub-pixar123",
            "role": "mask",
            "source_filename": "sub-pixar123_mask.nii.gz",
        },
    ]
    result = metrics().join_assets(records, ["sub-pixar123", "sub-pixar124"])
    assert result["sub-pixar123"]["mask"]["source_filename"] == "sub-pixar123_mask.nii.gz"
    with pytest.raises(ValueError, match="Duplicate"):
        metrics().join_assets(records + records[:1], ["sub-pixar123", "sub-pixar124"])
    records[0]["source_filename"] = "sub-pixar123_mask.nii.gz"
    with pytest.raises(ValueError, match="participant"):
        metrics().join_assets(records, ["sub-pixar123", "sub-pixar124"])


def test_failed_scan_archives_prior_success_without_leaving_active_artifacts(tmp_path, monkeypatch):
    """Catch mixing old success CSVs/markers with a new unavailable scan in exports."""
    monkeypatch.syspath_prepend(str(ROOT / "investigations/eight-adult-coverage"))
    spec = importlib.util.spec_from_file_location(
        "coverage_audit", ROOT / "investigations/eight-adult-coverage/audit.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scan = tmp_path / "scans/sub-test"
    scan.mkdir(parents=True)
    (scan / "completed.json").write_text('{"status":"test_success"}')
    (scan / "region_coverage.csv").write_text("Labeled test artifact; not study observations")
    assert hasattr(module, "archive_scan_attempt"), "Failed-scan history handling is missing"
    archived = module.archive_scan_attempt(scan, tmp_path / "history")
    assert scan.is_dir() and not list(scan.iterdir())
    assert (archived / "completed.json").read_text() == '{"status":"test_success"}'
    assert (
        archived / "region_coverage.csv"
    ).read_text() == "Labeled test artifact; not study observations"
