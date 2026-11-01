"""Artifact-only Streamlit inspection; run the Python CLI before opening this app."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from fmri_study_check.explorer import load_explorer, render_explorer, show_explorer
from fmri_study_check.supplements import load_supplement, supplement_archive
from fmri_study_check.ui import export_archive, file_digest, load_bundle

ROOT = Path(__file__).resolve().parent
VIEWS = [
    "Coverage Explorer",
    "Overview",
    "Scan Inspection",
    "Analysis Comparison",
    "Report",
    "Coverage Supplement",
]

st.set_page_config(page_title="fMRI Quality Explorer", page_icon="🧠", layout="wide")
st.markdown(
    """<style>
.block-container{max-width:1280px;padding-top:2.8rem;padding-bottom:3rem}
h1{font-weight:600!important;letter-spacing:-.035em}h2{letter-spacing:-.02em}
[data-testid="stSidebar"]{border-right:1px solid #d5e0e9}
[data-testid="stMetric"]{padding:12px 0;border-top:2px solid #b9cedd}
[data-testid="stMetricLabel"]{color:#476177}
[data-testid="stMetricValue"]{font-size:2rem}
[data-testid="stCaptionContainer"]{color:#476177}
a:focus-visible,button:focus-visible{outline:3px solid #b27818!important;outline-offset:2px}
</style>""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_bundle(path, signature):
    return load_bundle(path)


@st.cache_data(show_spinner=False)
def cached_archive(path, signature):
    return export_archive(path)


@st.cache_data(show_spinner=False)
def cached_supplement_archive(path, signature):
    return supplement_archive(path)


def display_table(df, **kwargs):
    st.dataframe(df, hide_index=True, width="stretch", **kwargs)


def image_if_available(path, caption=None):
    if path.is_file():
        st.image(str(path), width="stretch", caption=caption)


st.sidebar.title("Study workspace")
available = sorted(
    [p for p in (ROOT / "outputs").glob("*") if p.is_dir() and (p / "completed.json").is_file()],
    key=lambda p: (p.name != "eight", p.name != "one", p.name),
)
st.title("fMRI Quality Explorer")
st.caption("Inspect data integrity, brain coverage and motion sensitivity in public movie-watching fMRI")
if not available:
    st.info("No completed analysis bundle is available. Run the one-participant script first.")
    st.code("uv run python -m fmri_study_check run --n-subjects 1 --output-dir outputs/one")
    st.stop()
bundle_path = st.sidebar.selectbox(
    "Analysis bundle",
    available,
    format_func=lambda p: {"one": "One-participant checkpoint", "eight": "Eight adults"}.get(
        p.name, p.name
    ),
    key="bundle",
)
signature = file_digest(bundle_path / "completed.json")
try:
    bundle = cached_bundle(str(bundle_path), signature)
except (ValueError, OSError, KeyError) as exc:
    st.error(str(exc))
    st.code(f"uv run python -m fmri_study_check verify {bundle_path}")
    st.stop()
manifest, summary, settings = bundle["manifest"], bundle["summary"], bundle["settings"]
view = st.sidebar.radio("View", VIEWS, key="view")
scan_key = (
    st.sidebar.selectbox("Scan", manifest.scan_key.tolist(), key="scan") if len(manifest) else None
)
st.sidebar.divider()
st.sidebar.caption("Fixed comparison")
st.sidebar.markdown(
    "**Baseline + FD > 0.5 mm**\n\nCommon initial-frame policy. Minimum 120 seconds and 20 nominal residual df."
)
st.sidebar.caption("Views read saved results. Changing the view or scan does not run an analysis.")
if summary["status"] != "complete":
    st.warning(
        f"Results available for {summary['comparisons_complete']} of {summary['n_subjects']} requested scans. Unavailable results and integrity issues are retained for review."
    )


def for_scan(name):
    df = bundle[name]
    return df.loc[df.scan_key == scan_key] if scan_key and "scan_key" in df else df.iloc[:0]


def show_scan_issues():
    issues = for_scan("issues")
    if issues.empty:
        st.info("No issues recorded for this scan.")
    else:
        display_table(
            issues[
                [
                    "severity",
                    "rule",
                    "evidence",
                    "file_location",
                    "possible_explanations",
                    "recommended_action",
                ]
            ]
        )


if view == "Coverage Explorer":
    if scan_key is None:
        st.info("No scans are available for the Coverage Explorer.")
    else:
        participant = manifest.loc[manifest.scan_key == scan_key, "participant_id"].iloc[0]
        explorer_path = ROOT / "data" / "derived" / "coverage_explorer"
        try:
            payload = load_explorer(explorer_path, participant, primary_bundle=bundle_path)
            show_explorer(payload)
            with st.expander("Take this participant's explorer offline"):
                st.caption(
                    "Self-contained HTML with the same real display assets, charts and provenance. No analysis is run."
                )
                st.download_button(
                    "Download standalone fMRI Quality Explorer",
                    render_explorer(explorer_path, participant, primary_bundle=bundle_path),
                    file_name=f"{participant}-fmri-quality-explorer.html",
                    mime="text/html",
                )
        except (ValueError, OSError, KeyError) as exc:
            st.error(f"Coverage Explorer unavailable: {exc}")
            st.code("uv run python -B -m fmri_study_check.explorer_prepare")

elif view == "Overview":
    st.header("From source files to a reviewable result")
    st.write(
        "Inventory the scans, check image/confound alignment, and examine how one fixed motion rule changes retained data and regional correlations."
    )
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        cols = st.columns(3)
        cols[0].metric("Scans inventoried", f"{len(manifest)}/{summary['n_subjects']}")
        cols[1].metric("Paired comparisons", summary["comparisons_complete"])
        cols[2].metric("Blocking issues", int(bundle["issues"].severity.eq("error").sum()))
        st.markdown(
            "**Question**  \nHow sensitive are usable data and regional correlations to excluding frames with FD strictly above 0.5 mm?"
        )
        st.caption(
            "Movie responses, residual artifacts and unequal retained samples can affect the comparison. Matrix changes do not prove biological accuracy."
        )
    with right:
        if scan_key:
            image_if_available(
                bundle_path / "scans" / scan_key / "brain.png",
                f"{scan_key}: saved mean-BOLD slices",
            )
    st.subheader("Scan inventory")
    columns = [
        "participant_id",
        "n_volumes",
        "confound_rows",
        "sampling_interval_seconds",
        "duration_seconds",
        "processing_status",
    ]
    display_table(manifest[[c for c in columns if c in manifest]])
    st.subheader("Checks needing attention")
    attention = bundle["issues"].loc[
        bundle["issues"].severity.isin(["error", "review", "unavailable"])
    ]
    display_table(attention[["scan_key", "severity", "rule", "evidence"]])
    with st.expander("Validation demonstration — excluded from study results"):
        fixture_path = ROOT / "tests" / "fixtures" / "corrupted"
        if (fixture_path / "fixture.json").exists():
            metadata = json.loads((fixture_path / "fixture.json").read_text())
            st.write(
                f"Deliberately corrupted real excerpt: {metadata['image_volumes']} image volumes and {metadata['confound_rows']} confound rows."
            )
            st.caption(
                "This fixture tests mismatch detection. It is not a participant result and contributes no observations to the report."
            )
            fixture_issues = pd.read_csv(fixture_path / "issues.csv")
            display_table(fixture_issues[["rule", "evidence", "recommended_action"]])
        else:
            st.code("uv run python -m fmri_study_check fixture --source-bundle outputs/one")

elif view == "Scan Inspection":
    st.header("Scan inspection")
    if scan_key is None:
        st.info("No scans were acquired. See the recorded acquisition failure in Overview.")
        st.stop()
    record = manifest.loc[manifest.scan_key == scan_key].iloc[0]
    st.subheader(record.participant_id)
    image_if_available(
        bundle_path / "scans" / scan_key / "brain.png",
        "Mean preprocessed BOLD; orthogonal slices for visual review",
    )
    image_if_available(bundle_path / "scans" / scan_key / "motion.png")
    st.caption(
        "The first stored difference-based value is preserved in the export but omitted from summaries. Shading marks exclusions over original acquisition time; gaps are never compressed."
    )
    frames = for_scan("frame_exclusions")
    if not frames.empty:
        selected = st.slider("Original acquisition frame", 0, len(frames) - 1, 0, key="frame")
        display_table(frames.loc[frames.original_index == selected])
        with st.expander("All original frames and exclusion reasons"):
            display_table(frames)
    with st.expander("Source files and recorded metadata"):
        st.json(record.to_dict())
    st.subheader("Evidence and recommended actions")
    show_scan_issues()

elif view == "Analysis Comparison":
    st.header("Analysis comparison")
    st.write(
        "All methods are fixed; the FD condition adds only exclusions at FD > 0.5 mm. Both conditions omit frame zero and any source non-steady-state frames."
    )
    retained = for_scan("retention")
    if retained.empty:
        st.warning(
            "Analysis unavailable: no condition results were produced for this scan. See the recorded issues."
        )
        show_scan_issues()
    else:
        for row in retained.itertuples():
            if row.status != "complete":
                st.warning(f"{row.condition} unavailable: {row.reason}")
        display_table(
            retained[
                [
                    "condition",
                    "status",
                    "retained_volumes",
                    "retained_minutes",
                    "excluded_volumes",
                    "residual_df",
                ]
            ]
        )
        comparison = for_scan("comparison")
        if not comparison.empty:
            row = comparison.iloc[0]
            cols = st.columns(3)
            cols[0].metric("Mean absolute Δr", f"{row.mean_absolute_change:.5f}")
            cols[1].metric("Maximum absolute Δr", f"{row.max_absolute_change:.5f}")
            cols[2].metric(
                "Edge similarity",
                f"{row.edge_similarity:.5f}" if pd.notna(row.edge_similarity) else "Unavailable",
            )
            image_if_available(bundle_path / "scans" / scan_key / "matrices.png")
            st.caption(
                "Unique off-diagonal edges only. Same region ordering; correlation scales [-1, 1], difference scale [-2, 2]. Numerical exports retain diagonals."
            )
        st.info(
            "This is descriptive sensitivity to a documented rule. Changed matrices do not establish biological accuracy or denoising success."
        )
        with st.expander("Motion summary denominators"):
            display_table(for_scan("motion_summary"))
        with st.expander("Fixed analysis settings"):
            st.json(settings)

elif view == "Coverage Supplement":
    st.header("Brain coverage — separate supplement")
    st.info(
        "This separate audit covers the original eight adults. Primary source files, matrices, exclusions and analysis statuses are preserved. A computable matrix does not establish adequate brain coverage."
    )
    supplement_path = ROOT / "investigations" / "eight-adult-coverage"
    if not (supplement_path / "completed.json").is_file():
        st.warning("The separate coverage audit is not available yet.")
        st.code("uv run python investigations/eight-adult-coverage/audit.py")
    else:
        try:
            supplement_signature = file_digest(supplement_path / "completed.json")
            # Verify the small derived artifacts on every navigation, even when the ZIP is cached.
            supplement = load_supplement(str(supplement_path))
        except (ValueError, OSError, KeyError) as exc:
            st.error(str(exc))
            st.stop()
        coverage_summary = supplement["participant_summary"]
        participant = (
            manifest.loc[manifest.scan_key == scan_key, "participant_id"].iloc[0]
            if scan_key is not None
            else coverage_summary.participant_id.iloc[0]
        )
        participant_available = participant in set(coverage_summary.participant_id)
        if supplement["status"] != "complete":
            st.warning(
                f"Coverage audits completed for {len(coverage_summary)}/8 participants. Specific unavailable results are retained below."
            )
        if participant in {"sub-pixar123", "sub-pixar124"}:
            st.warning(
                f"{participant}: a published reanalysis excluded this participant for poor brain coverage. Inspect the measured spatial support below; this supplement introduces no new primary exclusions."
            )
        cols = st.columns(3)
        cols[0].metric("Coverage audits", f"{len(coverage_summary)}/8")
        cols[1].metric("Primary paired matrices", summary["comparisons_complete"])
        cols[2].metric("Coverage review flags", len(supplement["issues"]))
        st.caption(
            "The independent SPM8 mask is not the 2019 fMRIPrep mask. Tutorial temporal variation is a proxy influenced by masking, interpolation and int8 quantization. The two fractions measure different things."
        )
        display_table(
            coverage_summary[
                [
                    "participant_id",
                    "primary_status",
                    "archived_mask_cortical_atlas_fraction",
                    "archived_zero_mask_regions",
                    "tutorial_sparse_regions_below_50pct",
                    "tutorial_constant_regional_means",
                ]
            ]
        )
        image_if_available(supplement_path / "coverage_heatmap.png")
        st.subheader(f"{participant}: spatial support")
        if participant_available:
            image_if_available(supplement_path / "scans" / participant / "coverage.png")
        else:
            st.warning(
                f"Coverage audit unavailable for {participant}. See the specific failure below; no stale scan figure is presented."
            )
        region_rows = supplement["region_coverage"].loc[
            supplement["region_coverage"].participant_id == participant
        ]
        display_table(
            region_rows[
                [
                    "lineage",
                    "region_id",
                    "region_name",
                    "grid_voxels",
                    "grid_varying_voxels",
                    "grid_varying_fraction",
                    "regional_mean_constant",
                    "atlas_voxels",
                    "atlas_in_grid_voxels",
                    "atlas_mask_voxels",
                    "atlas_mask_unknown_voxels",
                    "atlas_mask_fraction",
                ]
            ]
        )
        st.caption(
            "Mask overlap uses every voxel in the original 2-mm atlas parcel, including out-of-grid locations. Tutorial variation uses parcel centers on the 4-mm image grid. Undefined mask locations remain unknown; bounds are exported. The 80% mask and 50% variation review bins are project conventions, not validated acceptability criteria."
        )
        with st.expander("Observed issues and possible explanations"):
            display_table(
                supplement["issues"].loc[supplement["issues"].participant_id == participant]
            )
            if supplement["failures"]:
                st.json(supplement["failures"])
        with st.expander("Source investigation and exact preprocessing gap"):
            st.write(
                "The exact 2019 fMRIPrep parent BOLD, brain mask and preprocessing HTML QA were not located in the public catalogs checked. Archived SPM8 data and native MRIQC QA corroborate the coverage concern but cannot isolate each preprocessing contribution."
            )
            st.markdown(
                "[Published reanalysis: participant exclusions](https://pmc.ncbi.nlm.nih.gov/articles/PMC10635736/) · [Tutorial preprocessing methods](https://osf.io/wjtyq/)"
            )
            previous_report = ROOT / "investigations/sub-pixar124-upstream/REPORT.md"
            if previous_report.is_file():
                st.download_button(
                    "Download sub-pixar124 source investigation",
                    previous_report.read_bytes(),
                    file_name="sub-pixar124-upstream-investigation.md",
                    mime="text/markdown",
                )
            st.json(supplement["settings"])
        report_bytes = (supplement_path / "report.html").read_bytes()
        st.download_button(
            "Download separate coverage report",
            report_bytes,
            file_name="fmri-coverage-supplement.html",
            mime="text/html",
        )
        st.download_button(
            "Download coverage CSVs, figures and methods",
            cached_supplement_archive(str(supplement_path), supplement_signature),
            file_name="fmri-coverage-supplement.zip",
            mime="application/zip",
        )
        with st.expander("Read the self-contained coverage report"):
            st.iframe(
                report_bytes.decode(),
                height=1000,
                alt="Separate eight-adult coverage audit with actual findings, methods and limitations",
            )

elif view == "Report":
    st.header("Investigator report")
    st.caption(
        "A self-contained HTML report with question, sample, methods, checks, actual findings, limitations, next steps and source attribution."
    )
    st.caption("Saved analysis reports retain the original fMRI Study Check title and provenance.")
    report_bytes = (bundle_path / "report.html").read_bytes()
    st.download_button(
        "Download HTML report",
        report_bytes,
        file_name=f"fmri-study-check-{bundle_path.name}.html",
        mime="text/html",
    )
    st.download_button(
        "Download CSVs and methods",
        cached_archive(str(bundle_path), signature),
        file_name=f"fmri-study-check-{bundle_path.name}-exports.zip",
        mime="application/zip",
    )
    csv_files = sorted(bundle_path.glob("*.csv"))
    csv_file = st.selectbox(
        "Individual CSV export", csv_files, format_func=lambda p: p.name, key="csv_export"
    )
    st.download_button(
        "Download selected CSV", csv_file.read_bytes(), file_name=csv_file.name, mime="text/csv"
    )
    st.iframe(
        report_bytes.decode(),
        height=1000,
        alt="Investigator report with methods, findings and limitations",
    )

st.divider()
st.caption(
    "Public preprocessed data via Nilearn and OSF; original study Richardson et al. (2018). Source files remain unchanged. See Report for attribution and recorded versions."
)
