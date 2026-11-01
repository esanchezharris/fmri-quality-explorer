"""Small synthetic arrays here test rules; they are never study observations."""

import numpy as np
import pandas as pd
import pytest

from fmri_study_check.core import (
    Settings,
    clean_condition,
    compare_matrices,
    frame_table,
    join_sources,
    motion_summary,
    validate_confounds,
)


def confounds(n=80):
    rng = np.random.default_rng(12)
    df = pd.DataFrame(
        rng.normal(size=(n, 8)),
        columns=["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z", "csf", "white_matter"],
    )
    df["cosine00"] = np.cos(np.arange(n) * np.pi / n)
    df["framewise_displacement"] = np.r_[np.nan, np.full(n - 1, 0.1)]
    return df


def test_joins_use_entities_not_input_order():
    joined = join_sources(
        [
            "/d/sub-b_task-pixar_space-MNI_desc-preproc_bold.nii.gz",
            "/d/sub-a_task-pixar_space-MNI_desc-preproc_bold.nii.gz",
        ],
        [
            "/d/sub-a_task-pixar_desc-confounds_regressors.tsv",
            "/d/sub-b_task-pixar_desc-confounds_regressors.tsv",
        ],
        pd.DataFrame({"participant_id": ["sub-a", "sub-b"], "Child_Adult": ["adult", "adult"]}),
    )
    assert joined[0]["participant_id"] == "sub-a"
    assert "sub-a_task" in joined[0]["confounds_path"]
    assert joined[0]["run"] is None


@pytest.mark.parametrize(
    "func,conf,phenotype",
    [
        (["sub-a_task-pixar_bold.nii.gz"] * 2, ["sub-a_task-pixar_confounds.tsv"], ["sub-a"]),
        (["sub-a_task-pixar_bold.nii.gz"], ["sub-b_task-pixar_confounds.tsv"], ["sub-a"]),
        (["sub-a_task-pixar_bold.nii.gz"], ["sub-a_task-pixar_confounds.tsv"], ["sub-b"]),
    ],
)
def test_ambiguous_or_mismatched_sources_are_rejected(func, conf, phenotype):
    with pytest.raises(ValueError):
        join_sources(
            func, conf, pd.DataFrame({"participant_id": phenotype, "Child_Adult": "adult"})
        )


def test_initial_missing_fd_is_expected_but_later_missingness_blocks():
    df = confounds()
    df["trans_x_derivative1"] = np.r_[np.nan, np.ones(len(df) - 1)]
    issues = validate_confounds(df, "scan", "/source.tsv")
    assert not [x for x in issues if x["severity"] == "error"]
    df.loc[4, "framewise_displacement"] = np.nan
    issues = validate_confounds(df, "scan", "/source.tsv")
    assert any(x["rule"] == "unexpected_missingness" and x["severity"] == "error" for x in issues)
    assert any("4" in str(x["rows"]) for x in issues if x["rule"] == "unexpected_missingness")


@pytest.mark.parametrize(
    "mutation,rule",
    [
        (lambda df: df.drop(columns="trans_x"), "required_columns"),
        (lambda df: df.assign(rot_x="bad"), "numeric_values"),
        (lambda df: df.assign(csf=np.inf), "finite_values"),
        (lambda df: df.assign(framewise_displacement=-1.0), "nonnegative_fd"),
        (lambda df: df.drop(columns="cosine00"), "cosine_regressors"),
    ],
)
def test_invalid_analysis_inputs_block(mutation, rule):
    issues = validate_confounds(mutation(confounds()), "scan", "source.tsv")
    assert any(x["rule"] == rule and x["severity"] == "error" for x in issues)


def test_fd_denominators_threshold_boundary_and_exclusion_union():
    df = confounds(5)
    df["framewise_displacement"] = [np.nan, 0.5, 0.5001, 0.2, 1.0]
    df["non_steady_state_outlier00"] = [1, 0, 1, 0, 0]
    frames = frame_table(df, 2.0, Settings())
    assert frames.loc[frames.baseline_keep, "original_index"].tolist() == [1, 3, 4]
    assert frames.loc[frames.censored_keep, "original_index"].tolist() == [1, 3]
    assert frames.time_seconds.tolist() == [0, 2, 4, 6, 8]
    summary = motion_summary(frames)
    assert summary["n_fd_observed"] == 4
    assert summary["high_fd_count"] == 2
    assert summary["high_fd_fraction"] == 0.5
    assert summary["mean_fd_mm"] == pytest.approx(0.550025)
    assert "initial_frame" in frames.loc[0, "baseline_exclusion_reasons"]
    assert "non_steady_state" in frames.loc[2, "censored_exclusion_reasons"]


def test_sample_mask_matches_independent_regression_and_original_indices():
    rng = np.random.default_rng(41)
    c = confounds().drop(columns="framewise_displacement")
    signals = rng.normal(size=(80, 4)) + c.to_numpy() @ rng.normal(size=(9, 4))
    kept = np.array([i for i in range(1, 80) if i not in [9, 10, 14, 37]])
    result = clean_condition(signals, c, kept, 2.0, Settings())
    design = np.column_stack([np.ones(len(kept)), c.iloc[kept].to_numpy()])
    residual = signals[kept] - design @ np.linalg.lstsq(design, signals[kept], rcond=None)[0]
    expected = (residual - residual.mean(axis=0)) / residual.std(axis=0, ddof=1)
    assert result["status"] == "complete"
    np.testing.assert_allclose(result["signals"], expected, atol=1e-9)
    np.testing.assert_allclose(
        result["correlation"], np.corrcoef(expected, rowvar=False), atol=1e-9
    )
    assert result["retained_volumes"] == 75
    assert result["retained_minutes"] == 2.5
    assert result["residual_df"] == 65


def test_insufficient_time_and_rank_are_not_published():
    rng = np.random.default_rng(2)
    assert (
        clean_condition(
            rng.normal(size=(80, 3)), confounds().iloc[:, :9], np.arange(59), 2, Settings()
        )["status"]
        == "unavailable"
    )
    high_rank = pd.DataFrame(np.eye(80)[:, :65])
    result = clean_condition(rng.normal(size=(80, 3)), high_rank, np.arange(80), 2, Settings())
    assert result["status"] == "unavailable"
    assert result["residual_df"] == 14


@pytest.mark.parametrize("bad", ["constant", "nan"])
def test_invalid_regional_calculations_are_explained(bad):
    rng = np.random.default_rng(13)
    signals = rng.normal(size=(80, 3))
    signals[:, 1] = 1 if bad == "constant" else np.nan
    result = clean_condition(signals, confounds().iloc[:, :9], np.arange(1, 80), 2, Settings())
    assert result["status"] == "unavailable"
    assert result["reason"]


def test_matrix_changes_exclude_diagonal_and_handle_constant_edge_vectors():
    a = np.eye(3)
    b = np.array([[1, 0.2, -0.4], [0.2, 1, 0.6], [-0.4, 0.6, 1]])
    result = compare_matrices(a, b)
    assert result["mean_absolute_change"] == pytest.approx(0.4)
    assert result["mean_signed_change"] == pytest.approx(0.4 / 3)
    assert result["max_absolute_change"] == 0.6
    assert result["edge_similarity"] is None


def test_stored_initial_zero_is_not_counted_as_an_observed_fd():
    df = confounds(4)
    df["framewise_displacement"] = [0, 0.1, 0.2, 0.6]
    frames = frame_table(df, 2, Settings())
    result = motion_summary(frames)
    assert frames.loc[0, "framewise_displacement"] == 0
    assert not frames.loc[0, "fd_defined"]
    assert result["n_fd_observed"] == 3
    assert result["mean_fd_mm"] == pytest.approx(0.3)


def test_nonfinite_nuisance_explains_unavailable_result_instead_of_svd_failure():
    c = confounds().iloc[:, :9].to_numpy()
    c[4, 0] = np.nan
    result = clean_condition(
        np.random.default_rng(3).normal(size=(80, 3)), c, np.arange(1, 80), 2, Settings()
    )
    assert result["status"] == "unavailable"
    assert "Non-finite" in result["reason"]


def test_constant_region_identifies_the_affected_feature():
    signals = np.random.default_rng(3).normal(size=(80, 4))
    signals[:, 2] = 0
    result = clean_condition(signals, confounds().iloc[:, :9], np.arange(1, 80), 2, Settings())
    assert result["invalid_region_columns"] == [2]
