"""Export explicitly labeled, path-free excerpts of verified real results. Never run analysis."""

import argparse
import hashlib
import html
import json
from pathlib import Path

import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(study_root, destination):
    from fmri_study_check.pipeline import verify_bundle

    study_root, destination = study_root.resolve(), destination.resolve()
    protected = [study_root / p for p in ("data", "outputs", "investigations", ".cache")]
    if any(destination.is_relative_to(p) or p.is_relative_to(destination) for p in protected):
        raise ValueError("Public examples must be isolated from scientific inputs/results")
    primary = study_root / "outputs/eight"
    audit = study_root / "investigations/eight-adult-coverage"
    status = verify_bundle(primary)
    for directory in (audit, audit / "overnight-followups"):
        marker = json.loads((directory / "completed.json").read_text())
        for relative, expected in marker["artifacts"].items():
            path = (directory / relative).resolve()
            if not path.is_relative_to(directory) or digest(path) != expected:
                raise ValueError(f"Saved result artifact changed: {relative}")
    manifest = pd.read_csv(primary / "manifest.csv")
    motion = pd.read_csv(primary / "motion_summary.csv")
    motion = motion.loc[
        motion.population.eq("all_acquired"),
        ["scan_key", "n_fd_observed", "mean_fd_mm", "high_fd_count"],
    ]
    retained = pd.read_csv(primary / "retention.csv")
    wide = retained.pivot(
        index="scan_key", columns="condition", values=["retained_volumes", "retained_minutes"]
    )
    wide.columns = [f"{condition}_{measure}" for measure, condition in wide.columns]
    summary = (
        manifest[["participant_id", "scan_key", "processing_status", "n_volumes"]]
        .merge(motion, on="scan_key", validate="one_to_one")
        .merge(wide, on="scan_key", validate="one_to_one")
    )
    comparison = pd.read_csv(primary / "comparison.csv")
    summary = summary.merge(comparison, on="scan_key", validate="one_to_one", how="left")
    full_coverage = pd.read_csv(audit / "participant_summary.csv")
    coverage = full_coverage[
        [
            "participant_id",
            "primary_status",
            "tutorial_r14_grid_varying_voxels",
            "tutorial_r14_grid_voxels",
            "tutorial_r37_grid_varying_voxels",
            "tutorial_r37_grid_voxels",
            "spm8_r14_atlas_mask_voxels",
            "spm8_r14_atlas_voxels",
            "spm8_r37_atlas_mask_voxels",
            "spm8_r37_atlas_voxels",
            "spm8_mask_nonfinite_voxels",
        ]
    ]
    if set(summary.participant_id) != {f"sub-pixar{i}" for i in range(123, 131)} or set(
        coverage.participant_id
    ) != set(summary.participant_id):
        raise ValueError("Original participant join disagrees")
    hemisphere = pd.read_csv(audit / "overnight-followups/hemisphere_coverage.csv")
    # The accompanying prose is a specific recorded case, never a template for invented findings.
    c = full_coverage.set_index("participant_id")
    primary_status = summary.set_index("participant_id").processing_status
    h = hemisphere.set_index(["participant_id", "evidence", "region_id", "hemisphere"])
    if not (
        status["comparisons_complete"] == 7
        and primary_status.loc["sub-pixar123"] == "complete"
        and primary_status.loc["sub-pixar124"] == "analysis_unavailable"
        and c.loc["sub-pixar123", "spm8_r14_atlas_mask_voxels"] == 171
        and c.loc["sub-pixar123", "spm8_r14_atlas_voxels"] == 647
        and c.loc["sub-pixar123", "spm8_r37_atlas_mask_voxels"] == 50
        and c.loc["sub-pixar123", "spm8_r37_atlas_voxels"] == 610
        and c.loc["sub-pixar123", "tutorial_r37_grid_varying_voxels"] == 20
        and c.loc["sub-pixar123", "tutorial_r37_grid_voxels"] == 76
        and c.loc["sub-pixar124", "tutorial_r14_grid_varying_voxels"] == 0
        and c.loc["sub-pixar124", "tutorial_r14_grid_voxels"] == 82
        and c.loc["sub-pixar124", "spm8_r14_atlas_mask_voxels"] == 0
        and c.loc["sub-pixar124", "spm8_r14_atlas_voxels"] == 647
        and bool(c.loc["sub-pixar124", "tutorial_r14_regional_mean_constant"])
        and c.loc["sub-pixar124", "primary_status"] == "analysis_unavailable"
        and c.loc["sub-pixar130", "spm8_mask_nonfinite_voxels"] == 152
        and c.loc["sub-pixar130", "archived_mask_unknown_atlas_voxels"] == 0
        and summary.n_volumes.eq(168).all()
        and summary.n_fd_observed.eq(167).all()
        and manifest.sampling_interval_seconds.eq(2.0).all()
        and h.loc[("sub-pixar123", "spm8_full_atlas_2mm", 37, "left"), "mask_voxels"] == 49
        and h.loc[("sub-pixar123", "spm8_full_atlas_2mm", 37, "left"), "voxels"] == 316
        and h.loc[("sub-pixar123", "spm8_full_atlas_2mm", 37, "right"), "mask_voxels"] == 1
        and h.loc[("sub-pixar123", "spm8_full_atlas_2mm", 37, "right"), "voxels"] == 294
    ):
        raise ValueError(
            "Recorded findings changed; review the narrative before exporting a new public snapshot"
        )
    destination.mkdir(parents=True, exist_ok=True)
    summary.to_csv(destination / "results_summary.csv", index=False)
    coverage.to_csv(destination / "coverage_summary.csv", index=False)
    manifest[
        [
            "participant_id",
            "image_filename",
            "image_sha256",
            "confounds_filename",
            "confounds_sha256",
        ]
    ].to_csv(destination / "source_hashes.csv", index=False)
    hemisphere.to_csv(destination / "hemisphere_coverage.csv", index=False)
    settings = json.loads((primary / "settings.json").read_text())
    provenance = json.loads((primary / "provenance.json").read_text())
    inputs = [
        primary / name
        for name in (
            "manifest.csv",
            "retention.csv",
            "motion_summary.csv",
            "comparison.csv",
            "settings.json",
        )
    ]
    inputs += [
        audit / "participant_summary.csv",
        audit / "overnight-followups/hemisphere_coverage.csv",
    ]
    metadata = dict(
        data_kind="saved_real_result_excerpt",
        primary_status=status,
        dataset_citation=provenance["dataset_citation"],
        settings=settings,
        dependency_versions=provenance["dependencies"],
        source_artifact_hashes={str(p.relative_to(study_root)): digest(p) for p in inputs},
        transformation="Identifier-validated joins and selected columns; original scientific artifacts unchanged; local filesystem paths omitted",
    )
    serialized = json.dumps(metadata, indent=2, allow_nan=False)
    if "/Users/" in serialized or "@gmail.com" in serialized:
        raise ValueError("Local identity information must not enter the public example")
    (destination / "provenance.json").write_text(serialized + "\n")
    display = summary[
        [
            "participant_id",
            "processing_status",
            "baseline_retained_volumes",
            "censored_retained_volumes",
            "n_fd_observed",
            "mean_absolute_change",
        ]
    ].copy()
    display.columns = [
        "Participant",
        "Primary status",
        "Baseline volumes",
        "FD-censored volumes",
        "Defined FD values",
        "Mean absolute Δr",
    ]
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>fMRI Quality Explorer — saved findings</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;color:#19384a;max-width:1050px;margin:48px auto;padding:0 24px}}h1,h2{{line-height:1.2}}table{{border-collapse:collapse;font-size:14px;width:100%}}th,td{{padding:9px;border:1px solid #d6e3eb;text-align:left}}th{{background:#eaf1f6}}pre{{white-space:pre-wrap;font-size:12px}}a{{color:#176b8a}}.notice{{background:#fff3db;padding:16px}}img{{max-width:100%}}</style>
<h1>fMRI Quality Explorer</h1><p>Saved findings from a descriptive movie-watching demonstration using eight public adult scans. This is an excerpt of verified outputs, not a new analysis or a validation study.</p>
<h2>Question and methods</h2><p>How does censoring FD strictly above 0.5 mm change retained data and regional Pearson correlations? Both conditions exclude frame zero and supplied non-steady-state frames. The atlas and nuisance model are fixed. Project guardrails require 120 retained seconds and 20 nominal residual design degrees of freedom. <a href="../docs/METHODS.md">Full methods</a>.</p>
<h2>Actual primary results</h2><p>Seven paired matrices were available. Sub-pixar124's region 14 was constant, so its matrices were withheld. Defined-FD summaries use 167 observations of 168 acquired frames; the stored initial value is preserved locally but undefined for difference-based summaries. Retained time is volumes × 2 seconds.</p>{display.to_html(index=False, na_rep="Unavailable", float_format=lambda v: f"{v:.5g}", border=0)}
<h2>Coverage investigation</h2><p>For sub-pixar123, the independent SPM8 mask included 171/647 atlas centers in region 14 and 50/610 in region 37. The region-37 split was 49/316 left and 1/294 right. In the tutorial derivative, 20/76 region-37 centers varied over the run despite a computable regional matrix. Sub-pixar124 had 0/82 varying tutorial region-14 centers and 0/647 centers in the independent archived mask.</p>
<p class="notice">The tutorial variation proxy and independent SPM8 mask measure different things with different denominators. The matching 2019 fMRIPrep parent mask and QA remain unavailable. These observations establish a coverage concern; they do not isolate acquisition, masking, registration, resampling or quantization contributions. No primary result or exclusion was changed.</p>
<p>Sub-pixar130's archived mask had 152 undefined locations, all outside the selected cortical atlas. Unknown membership remains unknown. Coverage review bins are project conventions, not validated exclusions. Changed matrices do not prove biological accuracy.</p>
<h2>Exports and next steps</h2><p><a href="results_summary.csv">Primary summary CSV</a> · <a href="coverage_summary.csv">Coverage CSV</a> · <a href="hemisphere_coverage.csv">Hemisphere CSV</a> · <a href="source_hashes.csv">Source identities and hashes</a> · <a href="provenance.json">Settings and provenance</a>. Inspect the original source QA and resolve the matching-parent gap before proposing a research inclusion decision.</p>
<h2>Recorded settings</h2><pre>{html.escape(json.dumps(settings, indent=2))}</pre>
<h2>Attribution</h2><p>{html.escape(provenance["dataset_citation"])}. <a href="https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_development_fmri.html">Nilearn dataset documentation</a> · <a href="https://osf.io/wjtyq/">Tutorial preprocessing documentation</a>. Independent archived derivatives are a separate SPM8 lineage. Existing Nilearn methods support extraction and cleaning; this project adds integrity checks, coverage audits, explicit failure handling and reporting.</p></html>"""
    (destination / "report.html").write_text(document)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "examples"
    )
    arguments = parser.parse_args()
    export(arguments.study_root, arguments.output_dir)
