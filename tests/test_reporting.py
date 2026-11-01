import json

import numpy as np
import pandas as pd

from fmri_study_check.reporting import plot_matrices, write_report


def test_report_states_actual_failure_without_fabricated_results(tmp_path):
    (tmp_path / "summary.json").write_text(
        json.dumps(
            dict(
                status="partial",
                requested_n_subjects=1,
                n_subjects=1,
                acquired_scans=0,
                comparisons_complete=0,
            )
        )
    )
    (tmp_path / "provenance.json").write_text(
        json.dumps(
            dict(
                selected_participant_ids=[],
                dependencies={},
                sources={},
                created_at_utc="2026-10-31",
                dataset_citation="Dataset citation",
            )
        )
    )
    (tmp_path / "settings.json").write_text(
        json.dumps(dict(fd_threshold_mm=0.5, minimum_seconds=120, minimum_residual_df=20))
    )
    pd.DataFrame(columns=["scan_key", "data_kind", "processing_status"]).to_csv(
        tmp_path / "manifest.csv", index=False
    )
    pd.DataFrame(
        [
            dict(
                scan_key="setup",
                severity="error",
                stage="acquisition",
                rule="workflow_failure",
                evidence="HTTP 403 <script>bad</script>",
                file_location="source",
                column="",
                rows="[]",
                possible_explanations="Access restriction or endpoint failure",
                recommended_action="Check endpoint",
            )
        ]
    ).to_csv(tmp_path / "issues.csv", index=False)
    for name in ["retention", "motion_summary", "comparison"]:
        pd.DataFrame(columns=["scan_key"]).to_csv(tmp_path / f"{name}.csv", index=False)
    write_report(tmp_path)
    html = (tmp_path / "report.html").read_text()
    assert "HTTP 403" in html
    assert "&lt;script&gt;bad&lt;/script&gt;" in html
    assert "No real-data comparison could be produced" in html
    assert "biological accuracy" in html


def test_matrix_plot_does_not_mutate_numerical_diagonal(tmp_path):
    a = np.array([[1, 0.2], [0.2, 1]])
    b = np.array([[1, -0.3], [-0.3, 1]])
    plot_matrices(
        a,
        b,
        pd.DataFrame({"region_id": [1, 2], "region_name": ["One", "Two"]}),
        tmp_path / "matrix.png",
    )
    np.testing.assert_array_equal(np.diag(a), [1, 1])
    np.testing.assert_array_equal(np.diag(b), [1, 1])
    assert (tmp_path / "matrix.png").stat().st_size > 1000
