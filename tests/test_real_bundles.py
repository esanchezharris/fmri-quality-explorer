"""Acceptance against downloaded observations; skip explicitly when not acquired."""

from html.parser import HTMLParser
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from fmri_study_check.core import key_for
from fmri_study_check.pipeline import inspect_scan, verify_bundle

ROOT = Path(__file__).resolve().parents[1]


def available(name):
    path = ROOT / "outputs" / name
    if not (path / "completed.json").exists():
        pytest.skip(f"Real {name} bundle not acquired; no replacement observations")
    return path


def test_real_stages_preserve_first_participant_methods_and_results():
    one, eight = available("one"), available("eight")
    assert verify_bundle(one)["all_analyses_complete"]
    assert verify_bundle(eight)["acquired_scans"] == 8
    first = pd.read_csv(one / "manifest.csv").iloc[0]
    expanded = pd.read_csv(eight / "manifest.csv")
    match = expanded.loc[expanded.scan_key == first.scan_key].iloc[0]
    assert first.image_sha256 == match.image_sha256
    assert first.confounds_sha256 == match.confounds_sha256
    for condition in ("baseline", "censored"):
        a = pd.read_csv(
            one / "scans" / first.scan_key / f"correlation_{condition}.csv", index_col=0
        )
        b = pd.read_csv(
            eight / "scans" / first.scan_key / f"correlation_{condition}.csv", index_col=0
        )
        pd.testing.assert_frame_equal(a, b)


def test_real_masks_and_motion_denominators_independently_match_source():
    path = available("eight")
    manifest = pd.read_csv(path / "manifest.csv")
    motion = pd.read_csv(path / "motion_summary.csv")
    for record in manifest.itertuples():
        source = pd.read_csv(record.confounds_path, sep="\t")
        frames = pd.read_csv(path / "scans" / record.scan_key / "frames.csv")
        assert nib.load(record.image_path).shape[3] == len(source) == len(frames)
        initial_or_nss = np.arange(len(source)) == 0
        nss = source.filter(regex=r"^non_steady_state_outlier")
        initial_or_nss |= nss.eq(1).any(axis=1).to_numpy()
        expected_baseline = ~initial_or_nss
        expected_censored = expected_baseline & ~(source.framewise_displacement.to_numpy() > 0.5)
        np.testing.assert_array_equal(frames.baseline_keep, expected_baseline)
        np.testing.assert_array_equal(frames.censored_keep, expected_censored)
        observed = source.framewise_displacement.iloc[1:]
        observed = observed[np.isfinite(observed)]
        actual = motion.loc[
            (motion.scan_key == record.scan_key) & (motion.population == "all_acquired")
        ].iloc[0]
        assert actual.n_fd_observed == len(observed)
        assert actual.mean_fd_mm == pytest.approx(observed.mean())
        assert actual.high_fd_fraction == pytest.approx((observed > 0.5).sum() / len(observed))


class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.rows, self.row, self.cell, self.in_cell = [], None, None, "", False

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.rows = []
        if tag == "tr" and self.rows is not None:
            self.row = []
        if tag in {"td", "th"}:
            self.in_cell, self.cell = True, ""

    def handle_data(self, data):
        if self.in_cell:
            self.cell += data

    def handle_endtag(self, tag):
        if tag in {"td", "th"}:
            self.row.append(self.cell.strip())
            self.in_cell = False
        if tag == "tr" and self.rows is not None:
            self.rows.append(self.row)
        if tag == "table":
            self.tables.append(self.rows)
            self.rows = None


def test_investigator_html_table_rows_reconcile_with_exports():
    path = available("eight")
    parser = Tables()
    parser.feed((path / "report.html").read_text())
    for table, filename in zip(
        parser.tables,
        ["manifest", "retention", "comparison", "motion_summary", "issues"],
        strict=True,
    ):
        df = pd.read_csv(path / f"{filename}.csv")
        assert len(table) - 1 == len(df)
        headers = table[0]
        for field in ["scan_key", "retained_volumes", "n_fd_observed", "mean_absolute_change"]:
            if field not in headers:
                continue
            actual = [row[headers.index(field)] for row in table[1:]]
            if field == "scan_key":
                assert actual == df[field].tolist()
            else:
                np.testing.assert_allclose(
                    np.array(actual, dtype=float), df[field].to_numpy(), rtol=1e-4
                )


def test_shipped_corrupted_real_excerpt_detects_five_vs_four():
    path = ROOT / "tests" / "fixtures" / "corrupted"
    if not list(path.glob("*.nii.gz")) or not list(path.glob("*.tsv")):
        pytest.skip("Generate the separate real excerpt after acquiring the one-participant checkpoint")
    image = next(path.glob("*.nii.gz"))
    table = next(path.glob("*.tsv"))
    record = dict(
        scan_key=key_for(image),
        participant_id="sub-pixar123",
        image_path=str(image),
        confounds_path=str(table),
    )
    row, _, issues, _ = inspect_scan(record, 2)
    assert row["n_volumes"] == 5 and row["confound_rows"] == 4
    assert any(x["rule"] == "image_confound_alignment" and x["severity"] == "error" for x in issues)
