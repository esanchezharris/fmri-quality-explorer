"""Labeled unit-test arrays only; never exported as study observations."""

import hashlib
import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


def methods():
    path = Path(__file__).with_name("hemisphere_audit.py")
    assert path.is_file(), "Hemisphere audit implementation is missing"
    spec = importlib.util.spec_from_file_location("hemisphere_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_coordinate_partition_records_midline_without_dropping_or_double_counting():
    masks = methods().hemisphere_masks(np.array([-4.0, -2.0, 0.0, 2.0, 4.0]))
    assert masks["left"].tolist() == [True, True, False, False, False]
    assert masks["midline"].tolist() == [False, False, True, False, False]
    assert masks["right"].tolist() == [False, False, False, True, True]
    assert np.stack(list(masks.values())).sum(axis=0).tolist() == [1, 1, 1, 1, 1]


def test_missing_mask_membership_stays_unknown_in_full_stratum_denominator():
    labels = np.array([14, 14, 14, 0])
    x = np.array([-2.0, 0.0, 2.0, 4.0])
    varying = np.array([True, False, True, True])
    mask = np.array([1.0, np.nan, 0.0, 1.0])
    rows = methods().stratify(labels, x, varying, 14, mask)
    left, midline, right = rows
    assert [r["hemisphere"] for r in rows] == ["left", "midline", "right"]
    assert left["voxels"] == 1 and left["mask_fraction"] == 1.0
    assert midline["mask_unknown_voxels"] == 1 and midline["mask_fraction"] is None
    assert midline["mask_fraction_lower"] == 0.0 and midline["mask_fraction_upper"] == 1.0
    assert right["mask_fraction"] == 0.0 and right["varying_fraction"] == 1.0
    assert sum(row["voxels"] for row in rows) == 3
    assert sum(row["varying_voxels"] for row in rows) == 2


def test_empty_stratum_is_unavailable_rather_than_zero_percent():
    rows = methods().stratify(np.array([37, 37]), np.array([-2.0, 2.0]), np.array([True, True]), 37)
    assert rows[1]["voxels"] == 0
    assert rows[1]["varying_fraction"] is None


def test_nonbinary_finite_mask_or_infinite_membership_is_rejected():
    for bad in (0.5, np.inf):
        with pytest.raises(ValueError, match="mask"):
            methods().stratify(
                np.array([14]), np.array([-2.0]), np.array([True]), 14, np.array([bad])
            )


def test_sampling_preserves_out_of_grid_locations_and_uses_half_up_nearest():
    source = nib.Nifti1Image(np.zeros((2, 1, 1), dtype=np.uint8), np.diag([2.0, 1.0, 1.0, 1.0]))
    target = nib.Nifti1Image(np.zeros((4, 1, 1), dtype=np.uint8), np.eye(4))
    values = np.array([1, 0], dtype=np.uint8).reshape(2, 1, 1)
    observed, domain = methods().sample_on_grid(values, source, target)
    assert observed.ravel().tolist() == [1, 0, 0, 0]
    assert domain.ravel().tolist() == [True, True, True, False]


def test_stale_independent_proof_cannot_approve_changed_derived_data(tmp_path):
    path = Path(__file__).with_name("finalize.py")
    assert path.is_file(), "Final proof validation is missing"
    spec = importlib.util.spec_from_file_location("hemisphere_finalize", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = tmp_path / "data.csv"
    data.write_text("Labeled test artifact, not study observations")
    expected = hashlib.sha256(data.read_bytes()).hexdigest()
    (tmp_path / "independent_verification.json").write_text(
        json.dumps({"status": "passed", "input_artifacts": {"data.csv": expected}})
    )
    module.verify_proof(tmp_path)
    data.write_text("Deliberately modified test artifact")
    with pytest.raises(ValueError, match="stale"):
        module.verify_proof(tmp_path)
