"""Static figures and an escaped, self-contained investigator-facing report."""

import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Environment, select_autoescape
from matplotlib.colors import Normalize
from nilearn.plotting import plot_epi

INK, TEAL, ORANGE = "#243642", "#16736c", "#c05d35"


def plot_brain(mean_image, path, participant):
    display = plot_epi(
        mean_image,
        title=f"{participant} · mean preprocessed BOLD",
        display_mode="ortho",
        cut_coords=(0, -20, 20),
        black_bg=False,
        colorbar=True,
        cmap="gray",
        figure=plt.figure(figsize=(10, 3.6)),
    )
    display.savefig(path, dpi=140)
    display.close()
    plt.close("all")


def plot_motion(frames, path, settings):
    rows = 2 if "std_dvars" in frames else 1
    figure, axes = plt.subplots(rows, 1, figsize=(11, 2.8 * rows), sharex=True, squeeze=False)
    axis = axes[0, 0]
    fd = frames.framewise_displacement.where(frames.fd_defined)
    axis.plot(frames.time_seconds, fd, color=TEAL, linewidth=1.3, label="Observed FD")
    axis.axhline(
        settings.fd_threshold_mm,
        color=ORANGE,
        linestyle="--",
        label=f"FD > {settings.fd_threshold_mm:g} mm",
    )
    for index, row in frames.iterrows():
        if not row.censored_keep:
            tr = frames.time_seconds.iloc[1] - frames.time_seconds.iloc[0] if len(frames) > 1 else 2
            axis.axvspan(row.time_seconds, row.time_seconds + tr, color=ORANGE, alpha=0.16)
    axis.set_ylabel("FD (mm)")
    axis.legend(loc="upper right", fontsize=8)
    axis.set_title(
        "Motion over original acquisition time · shaded frames excluded in FD condition",
        loc="left",
        fontsize=11,
    )
    if rows == 2:
        axes[1, 0].plot(
            frames.time_seconds,
            frames.std_dvars.where(~frames.initial_frame),
            color=INK,
            linewidth=1,
        )
        axes[1, 0].set_ylabel("Standardized DVARS")
        axes[1, 0].set_title(
            "Displayed for review; DVARS does not determine censoring", loc="left", fontsize=9
        )
    axes[-1, 0].set_xlabel("Original acquisition-relative time (seconds; index × TR)")
    for axis in axes[:, 0]:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.15)
    figure.tight_layout()
    figure.savefig(path, dpi=140, facecolor="white")
    plt.close(figure)


def plot_matrices(baseline, censored, regions, path):
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for axis, data, title, limit in zip(
        axes,
        [baseline, censored, censored - baseline],
        ["Baseline", "FD-censored", "Censored − baseline"],
        [1, 1, 2],
        strict=True,
    ):
        shown = np.ma.array(data.copy(), mask=np.eye(len(data), dtype=bool))
        artist = axis.imshow(
            shown, cmap="RdBu_r", norm=Normalize(-limit, limit), interpolation="nearest"
        )
        axis.set_title(title, fontsize=12)
        axis.set_xlabel("Region ID order")
        axis.set_ylabel("Region ID order")
        step = max(1, len(regions) // 8)
        positions = np.arange(0, len(regions), step)
        axis.set_xticks(positions, regions.region_id.iloc[positions].astype(str), fontsize=7)
        axis.set_yticks(positions, regions.region_id.iloc[positions].astype(str), fontsize=7)
        figure.colorbar(artist, ax=axis, shrink=0.7, label="Δr" if limit == 2 else "Pearson r")
    figure.savefig(path, dpi=140, facecolor="white")
    plt.close(figure)


def image_data(path):
    path = Path(path)
    return (
        "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
        if path.is_file()
        else None
    )


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>fMRI Study Check — investigator report</title>
<style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f3f4ef;color:#243642;font:15px/1.65 system-ui,sans-serif}
main{max-width:1120px;margin:auto;padding:54px 32px}header{border-top:5px solid #16736c;padding:20px 0 26px;border-bottom:1px solid #ccd3cc}
.eyebrow{font-size:12px;letter-spacing:.12em;color:#16736c;font-weight:700;text-transform:uppercase}h1{font:normal 44px/1.15 Georgia,serif;margin:14px 0}
h2{font-size:22px;margin:34px 0 12px}h3{font-size:18px}.lead{max-width:850px;font-size:18px}.badge{display:inline-block;border:1px solid #c7d4ca;padding:3px 10px;border-radius:20px;font-size:12px;background:#edf3ec}
.metrics{display:flex;gap:16px;flex-wrap:wrap;margin:24px 0}.metric{background:white;border:1px solid #d8ded6;padding:14px 22px;min-width:155px}.metric strong{display:block;font:32px Georgia,serif}.small{font-size:12px;color:#52636b}
.tablewrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:12px;background:white}th{text-align:left;background:#e6ece6}td,th{border-bottom:1px solid #dbe0da;padding:9px 10px;vertical-align:top}td{overflow-wrap:anywhere}
.notice{border-left:3px solid #c05d35;padding:12px 18px;background:#fff8ed}.figure{width:100%;height:auto;background:white;border:1px solid #dbe0da;margin:10px 0}details{margin:14px 0;background:white;border:1px solid #dbe0da;padding:14px 18px}summary{cursor:pointer;font-weight:650}a{color:#16736c}code{overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.issues td:nth-child(5){min-width:170px}
@media print{body{background:white}main{padding:15px}details{break-inside:avoid}.metric{min-width:120px}a{color:inherit}h2{break-after:avoid}}
</style></head><body><main>
<header><div class="eyebrow">Research data analyst portfolio · descriptive demonstration</div><h1>fMRI Study Check</h1><p class="lead">How does a fixed motion-censoring rule change usable data and regional correlations in a small adult movie-watching sample?</p><span class="badge">{{ summary.status|upper }}</span> <span class="small">Generated {{ provenance.created_at_utc }} · real public data only</span></header>
<div class="metrics"><div class="metric"><strong>{{ summary.acquired_scans }}/{{ summary.n_subjects }}</strong>scans inventoried</div><div class="metric"><strong>{{ summary.comparisons_complete }}</strong>paired comparisons</div><div class="metric"><strong>{{ errors }}</strong>blocking integrity/setup issues</div><div class="metric"><strong>{{ reviews }}</strong>flags needing review</div></div>
<h2>Question and sample</h2><p>This workflow inventories public preprocessed fMRI data and describes sensitivity to excluding FD &gt; {{ settings.fd_threshold_mm }} mm. Participants are adults selected by the Nilearn fetcher, not a representative population sample. The task is movie watching, not resting-state acquisition.</p>
<p><strong>Selected IDs:</strong> {{ provenance.selected_participant_ids|join(', ') if provenance.selected_participant_ids else 'None successfully acquired.' }}</p>
{{ manifest_html|safe }}
<h2>Methods held fixed</h2><p>Harvard–Oxford cortical atlas <code>cort-maxprob-thr25-2mm</code>, no hemisphere splitting, nearest-neighbor label resampling to the image grid, fixed fitted region-ID ordering. Cache raw regional means without cleaning. Regress six basic translation/rotation parameters, mean white-matter and CSF signals, and the supplied cosine drift terms. No global-signal regression, CompCor, extra smoothing, or additional temporal filter. Cosine cutoff is not inferred when metadata does not establish it.</p>
<p>Clean each condition separately using full-length signals/confounds and supported original-index <code>sample_mask</code> handling: detrend=False; standardize=zscore_sample; standardize_confounds=True; filter=False; high_pass=None; low_pass=None; ensure_finite=False. Compute ordinary Pearson correlations; this is an explicit estimator choice instead of the official example’s default shrinkage estimator.</p>
<p><strong>Baseline:</strong> exclude index zero and supplied non-steady-state flags. <strong>FD condition:</strong> the same exclusions plus FD strictly &gt; {{ settings.fd_threshold_mm }} mm. No DVARS, neighboring-frame, or short-segment exclusions. Every retained/excluded index is exported. Motion figures retain original acquisition time.</p>
<div class="notice"><strong>Initial values and denominators.</strong> Preserve all stored source values, including initial zeros or undefined values. Index zero has no preceding frame and is excluded from FD/DVARS summaries and both analysis conditions. FD summaries count only finite, defined FD observations; their denominators are exported separately from acquired and retained volumes. Retained minutes = retained volumes × sampling interval / 60.</div>
<p>A condition needs at least {{ settings.minimum_seconds }} retained seconds and {{ settings.minimum_residual_df }} nominal residual design degrees of freedom, computed as retained rows minus rank([intercept, retained nuisance columns]). These project guardrails do not establish stable connectivity or independent physiological degrees of freedom. Missing, constant or invalid regional calculations are reported as unavailable.</p>
<h2>Actual findings</h2>{% for finding in findings %}<p>{{ finding }}</p>{% endfor %}{% if summary.comparisons_complete == 0 %}<p class="notice">No real-data comparison could be produced. See the observed failures below; no observations or findings have been substituted.</p>{% endif %}
{{ retention_html|safe }}<p class="small">Paired conditions keep all other choices fixed. Additional FD exclusions count only frames that survived the common policy.</p>{{ comparison_html|safe }}
<p class="small">Changes summarize unique off-diagonal edges. Matrix plots share native atlas ordering and [-1, 1] scales; differences use a fixed [-2, 2] scale. Diagonals are hidden only in plots, preserved in numerical exports.</p>
<h3>Motion summary and denominators</h3>{{ motion_html|safe }}
{% for scan in scans %}<details><summary>{{ scan.participant_id }} · {{ scan.processing_status }} · expand figures</summary><p>{{ scan.finding }}</p>{% if scan.brain %}<img class="figure" src="{{ scan.brain }}" alt="Orthogonal slices of the participant's mean preprocessed BOLD image"><p class="small">Mean BOLD slices for visual review; this is not an anatomical diagnosis or a registration-quality certification.</p>{% endif %}{% if scan.motion %}<img class="figure" src="{{ scan.motion }}" alt="Framewise displacement and available standardized DVARS over original acquisition time">{% endif %}{% if scan.matrices %}<img class="figure" src="{{ scan.matrices }}" alt="Baseline, FD-censored and difference regional correlation matrices">{% endif %}</details>{% endfor %}
<h2>Checks and issues</h2><p>Checks cover file readability, image finiteness, identifier joins and unique keys, adult selection, confound columns and numeric values, missingness, timing, image/confound row agreement, exclusion indices and denominators, regional identities and valid correlations. Informational library messages and expected initial missingness are retained separately. Quality flags require review; possible explanations are not verified root causes.</p>
<details><summary>Full issue register, including informational records</summary>{{ issues_html|safe }}</details>
<h2>Limitations and next steps</h2><p>This small convenience sample and short, downsampled movie-watching excerpt supports a workflow demonstration. Shared movie responses, residual motion, physiological noise, atlas coverage and unequal retained samples can affect correlations. A changed matrix does not prove biological accuracy, denoising success or an individual difference. No population inference, causal claim, clinical interpretation or group hypothesis test is made.</p>
<p>Review brain slices and flagged frames, resolve integrity errors and missing provenance where possible, and use a study-specific protocol before extending this workflow. The separately labeled corrupted fixture demonstrates mismatch detection and contributes no observations or findings to this report.</p>
<h2>Reproduction and attribution</h2><p>{{ provenance.dataset_citation }}</p><p>Original dataset: {{ provenance.original_dataset_version|default('unknown') }}. OSF derivative release: {{ provenance.osf_derivative_version|default('unknown') }}. Preprocessing software version: {{ provenance.fmriprep_version|default('unknown') }}. Dataset origin and derivative version are distinct.</p>
<ul>{% for label,url in provenance.sources.items() %}<li><a href="{{ url }}">{{ label|replace('_',' ') }}</a></li>{% endfor %}</ul>
<p><strong>Adapted:</strong> Nilearn dataset/atlas fetching, regional signal extraction, fMRIPrep nuisance selection and supported sample-mask cleaning. <strong>Added here:</strong> integrity inventory, evidence-based issues, initial-frame policy, FD-only comparison, exclusion/denominator audit, reproducible exports, static reporting and cached inspection UI.</p>
<details><summary>Recorded analysis settings</summary><pre>{{ settings_json }}</pre></details><details><summary>Dependency versions and source provenance</summary><pre>{{ provenance_json }}</pre></details>
<p class="small">Download the accompanying CSVs, settings.json and provenance.json for exact parameters, fitted region identities, source paths and SHA-256 hashes. Report content is embedded; no network connection is needed to read this HTML.</p>
</main></body></html>"""


def write_report(directory):
    directory = Path(directory)
    summary, provenance, settings = [
        json.loads((directory / f"{name}.json").read_text())
        for name in ("summary", "provenance", "settings")
    ]
    tables = {
        name: pd.read_csv(directory / f"{name}.csv")
        for name in ("manifest", "issues", "retention", "motion_summary", "comparison")
    }
    if len(tables["manifest"]) and not tables["manifest"].data_kind.eq("real").all():
        raise ValueError("Refusing to include fixture data in a real-data report")
    scans = []
    for row in tables["manifest"].to_dict(orient="records"):
        path = directory / "scans" / row["scan_key"]
        retention = tables["retention"]
        paired = (
            retention.loc[retention.scan_key == row["scan_key"]]
            if "scan_key" in retention
            else pd.DataFrame()
        )
        finding = (
            "; ".join(
                f"{r.condition}: {r.retained_volumes} volumes ({r.retained_minutes:.2f} minutes), {r.status}"
                for r in paired.itertuples()
            )
            or "Analysis unavailable; see checks and issues."
        )
        scans.append(
            dict(
                **row,
                finding=finding,
                brain=image_data(path / "brain.png"),
                motion=image_data(path / "motion.png"),
                matrices=image_data(path / "matrices.png")
                if row["processing_status"] == "complete"
                else None,
            )
        )

    def table(name, columns=None):
        df = tables[name]
        if columns is not None:
            df = df[[c for c in columns if c in df]]
        if df.empty:
            return "<p class='small'>No rows available.</p>"
        return (
            '<div class="tablewrap">'
            + df.to_html(
                index=False,
                escape=True,
                na_rep="unavailable",
                float_format=lambda value: f"{value:.5g}",
                border=0,
            )
            + "</div>"
        )

    env = Environment(autoescape=select_autoescape(default=True))
    findings = [
        f"{summary['comparisons_complete']} of {summary['acquired_scans']} acquired scans produced paired correlation matrices."
    ]
    comparison = tables["comparison"]
    if not comparison.empty:
        changes = comparison.mean_absolute_change
        findings.append(
            f"Across available pairs, mean absolute off-diagonal correlation changes ranged from {changes.min():.5f} to {changes.max():.5f}."
        )
    retained = tables["retention"]
    if not retained.empty:
        excluded = retained.loc[retained.condition == "censored", "additional_fd_excluded_volumes"]
        if len(excluded):
            findings.append(
                f"Across acquired scans, the FD rule excluded {int(excluded.min())} to {int(excluded.max())} additional volumes after common exclusions."
            )
    unavailable = [
        r["participant_id"]
        for r in tables["manifest"].to_dict(orient="records")
        if r["processing_status"] != "complete"
    ]
    if unavailable:
        findings.append(
            "Paired results unavailable for: "
            + ", ".join(unavailable)
            + ". See the condition reasons and issue register; these scans remain in the sample inventory."
        )
    html = env.from_string(TEMPLATE).render(
        findings=findings,
        summary=summary,
        provenance=provenance,
        settings=settings,
        errors=int(tables["issues"].severity.eq("error").sum()),
        reviews=int(tables["issues"].severity.eq("review").sum()),
        scans=scans,
        manifest_html=table(
            "manifest",
            [
                "participant_id",
                "scan_key",
                "image_dimensions",
                "n_volumes",
                "sampling_interval_seconds",
                "duration_seconds",
                "confound_rows",
                "timing_source",
                "processing_status",
            ],
        ),
        retention_html=table("retention"),
        comparison_html=table("comparison"),
        motion_html=table("motion_summary"),
        issues_html=table("issues"),
        settings_json=json.dumps(settings, indent=2),
        provenance_json=json.dumps(provenance, indent=2),
    )
    temporary = directory / "report.html.tmp"
    temporary.write_text(html)
    temporary.replace(directory / "report.html")
