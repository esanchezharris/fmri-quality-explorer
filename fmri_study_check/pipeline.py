"""Sequential real-data acquisition, inspection, cached analysis and bundle verification."""

import csv
import hashlib
import importlib.metadata
import json
import shutil
import traceback
import uuid
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.image import resample_to_img
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.maskers import NiftiLabelsMasker

from . import __version__
from .core import (
    CONDITIONS,
    Settings,
    clean_condition,
    compare_matrices,
    frame_table,
    issue,
    join_sources,
    motion_summary,
    nss_columns,
    selected_columns,
    validate_confounds,
)

SOURCES = {
    "dataset_documentation": "https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_development_fmri.html",
    "official_example": "https://nilearn.github.io/stable/auto_examples/03_connectivity/plot_signal_extraction.html",
    "confound_documentation": "https://fmriprep.org/en/stable/outputs.html",
    "cleaning_documentation": "https://nilearn.github.io/stable/modules/generated/nilearn.signal.clean.html",
    "original_dataset": "https://openneuro.org/datasets/ds000228/versions/1.0.0",
    "derivative_files": "https://osf.io/5hju4/files/",
    "preprocessing_details": "https://osf.io/wjtyq/",
}
ISSUE_COLUMNS = [
    "scan_key",
    "severity",
    "stage",
    "rule",
    "evidence",
    "file_location",
    "column",
    "rows",
    "possible_explanations",
    "recommended_action",
]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n")
    temporary.replace(path)


def versions():
    names = [
        "nilearn",
        "nibabel",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "streamlit",
        "jinja2",
    ]
    return {name: importlib.metadata.version(name) for name in names}


def code_digest():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare_output(directory):
    """Archive previous outputs so retries cannot mix old matrices with new failures."""
    directory = Path(directory).resolve()
    archived = None
    if directory.exists() and any(directory.iterdir()):
        history = directory.parent / ".history"
        history.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        archived = history / f"{directory.name}-{stamp}-{uuid.uuid4().hex[:8]}"
        directory.rename(archived)
    directory.mkdir(parents=True, exist_ok=True)
    return archived


def fetch_adults(n, data_dir, *, fetcher=None):
    fetcher = datasets.fetch_development_fmri if fetcher is None else fetcher
    kwargs = dict(
        n_subjects=n,
        age_group="adults",
        reduce_confounds=False,
        data_dir=str(Path(data_dir).resolve()),
        resume=True,
        verbose=1,
    )
    event = dict(
        attempted_at_utc=now(),
        requested_age_group="adults",
        effective_age_group="adults",
        n_subjects=n,
        reduce_confounds=False,
        compatibility_error=None,
    )
    try:
        result = fetcher(**kwargs)
    except ValueError as exc:
        message = str(exc)
        if (
            "'age_group' must be one of" not in message
            or "'adult'" not in message
            or "'adults' was provided" not in message
        ):
            raise
        event.update(effective_age_group="adult", compatibility_error=message)
        kwargs["age_group"] = "adult"
        result = fetcher(**kwargs)
    event["completed_at_utc"] = now()
    return result, event


def inspect_scan(record, fetcher_tr):
    row = {k: v for k, v in record.items() if k != "phenotype"}
    row.update(
        data_kind="real",
        processing_status="pending",
        image_filename=Path(record["image_path"]).name,
        confounds_filename=Path(record["confounds_path"]).name,
    )
    issues, df, mean = [], None, None
    key, image_path, table_path = record["scan_key"], record["image_path"], record["confounds_path"]
    for field in ("image", "confounds"):
        path = record[f"{field}_path"]
        if Path(path).is_file():
            row[f"{field}_sha256"] = sha256(path)
            row[f"{field}_bytes"] = Path(path).stat().st_size
    try:
        img = nib.load(image_path)
        if len(img.shape) != 4 or min(img.shape) <= 0:
            raise ValueError(f"Expected nonempty 4D image; observed {img.shape}")
        if not np.isfinite(img.affine).all() or np.linalg.matrix_rank(img.affine) != 4:
            raise ValueError("Non-finite or singular image affine")
        if np.any(np.asarray(img.header.get_zooms()[:3]) <= 0):
            raise ValueError("Non-positive spatial voxel sizes")
        row.update(
            image_dimensions="x".join(map(str, img.shape)),
            n_volumes=int(img.shape[3]),
            voxel_sizes_mm=json.dumps([float(x) for x in img.header.get_zooms()[:3]]),
            affine=json.dumps(img.affine.tolist()),
            header_time_zoom=float(img.header.get_zooms()[3]),
            header_time_unit=img.header.get_xyzt_units()[1],
        )
        factor = {"sec": 1.0, "msec": 0.001, "usec": 0.000001}.get(row["header_time_unit"])
        zoom = row["header_time_zoom"]
        if factor is None or not np.isfinite(zoom) or zoom <= 0:
            tr = float(fetcher_tr)
            row["timing_source"] = "fetcher_metadata_fallback"
            issues.append(
                issue(
                    key,
                    "review",
                    "missing_header_timing",
                    f"Header time zoom={zoom}, unit={row['header_time_unit']}; using fetcher t_r={tr} seconds",
                    image_path,
                    explanations="The derivative header does not establish a sampling interval with known time units.",
                    action="Confirm dataset metadata; interpret acquisition-relative times using the explicitly recorded fetcher fallback.",
                )
            )
        else:
            tr = zoom * factor
            row["timing_source"] = "nifti_header"
            if not np.isclose(tr, fetcher_tr, rtol=1e-5, atol=1e-6):
                issues.append(
                    issue(
                        key,
                        "error",
                        "timing_agreement",
                        f"Header={tr} seconds; fetcher={fetcher_tr} seconds",
                        image_path,
                        action="Resolve the conflicting authoritative timing before analysis; do not silently override it.",
                    )
                )
        if not np.isfinite(tr) or tr <= 0:
            raise ValueError(f"Invalid sampling interval: {tr}")
        row.update(sampling_interval_seconds=tr, duration_seconds=img.shape[3] * tr)
        values = np.asarray(img.dataobj, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(
                f"Image contains {np.count_nonzero(~np.isfinite(values))} non-finite voxels/timepoints"
            )
        mean = nib.Nifti1Image(values.mean(axis=3, dtype=np.float64).astype(np.float32), img.affine)
        del values
    except (OSError, ValueError, EOFError, nib.filebasedimages.ImageFileError) as exc:
        issues.append(
            issue(key, "error", "image_readability", f"{type(exc).__name__}: {exc}", image_path)
        )
    try:
        with open(table_path, newline="") as stream:
            header = next(csv.reader(stream, delimiter="\t"))
        if len(set(header)) != len(header):
            issues.append(
                issue(
                    key,
                    "error",
                    "unique_confound_columns",
                    f"Duplicate columns in raw header: {header}",
                    table_path,
                )
            )
        df = pd.read_csv(table_path, sep="\t", na_values=["n/a"])
        row["confound_rows"] = len(df)
        row["confound_columns"] = json.dumps(list(df.columns))
        issues.extend(validate_confounds(df, key, table_path))
        if row.get("n_volumes") != len(df):
            issues.append(
                issue(
                    key,
                    "error",
                    "image_confound_alignment",
                    f"Image timepoints={row.get('n_volumes')}; confound rows={len(df)}",
                    table_path,
                    action="Check file pairing and truncation; reacquire the affected input. Do not trim or pad rows to make them match.",
                )
            )
    except (OSError, ValueError, StopIteration, pd.errors.ParserError) as exc:
        issues.append(
            issue(key, "error", "confound_readability", f"{type(exc).__name__}: {exc}", table_path)
        )
    row["processing_status"] = (
        "integrity_blocked" if any(x["severity"] == "error" for x in issues) else "validated"
    )
    return row, df, issues, mean


def extract_regions(image_path, atlas, cache_dir, input_hash, atlas_hash):
    signature = hashlib.sha256(
        json.dumps(
            dict(
                image=input_hash,
                atlas=atlas_hash,
                code=code_digest(),
                dependencies=versions(),
                extraction="raw_mean_nearest_no_cleaning",
            ),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    cache = Path(cache_dir) / "regional" / signature
    cache.mkdir(parents=True, exist_ok=True)
    if (cache / "complete.json").is_file():
        marker = json.loads((cache / "complete.json").read_text())
        if all(
            (cache / name).is_file() and sha256(cache / name) == digest
            for name, digest in marker.items()
        ):
            return np.load(cache / "signals.npy"), pd.read_csv(cache / "regions.csv"), True
    img = nib.load(image_path)
    labels_img = resample_to_img(
        atlas.maps, img, interpolation="nearest", force_resample=True, copy_header=True
    )
    masker = NiftiLabelsMasker(
        labels_img=labels_img,
        lut=atlas.lut,
        standardize=None,
        detrend=False,
        smoothing_fwhm=None,
        high_pass=None,
        low_pass=None,
        resampling_target=None,
        reports=False,
    )
    signals = masker.fit_transform(image_path, confounds=None, sample_mask=None)
    actual_ids = [int(masker.region_ids_[i]) for i in range(signals.shape[1])]
    expected_ids = atlas.lut.loc[atlas.lut["index"] != 0, "index"].astype(int).tolist()
    if sorted(actual_ids) != sorted(expected_ids) or len(set(actual_ids)) != len(actual_ids):
        raise ValueError(
            f"Atlas labels missing/duplicated after extraction: expected={expected_ids}; fitted={actual_ids}"
        )
    names = atlas.lut.set_index("index")["name"]
    order = np.argsort(actual_ids)
    signals = signals[:, order]
    ids = [actual_ids[i] for i in order]
    regions = pd.DataFrame(dict(region_id=ids, region_name=[names.loc[i] for i in ids]))
    np.save(cache / "signals.npy", signals)
    regions.to_csv(cache / "regions.csv", index=False)
    write_json(
        cache / "complete.json",
        {p.name: sha256(p) for p in [cache / "signals.npy", cache / "regions.csv"]},
    )
    return signals, regions, False


def analyze_scan(row, df, mean, atlas, atlas_hash, out, cache_dir, settings):
    from .reporting import plot_brain, plot_matrices, plot_motion

    key = row["scan_key"]
    scan_dir = out / "scans" / key
    scan_dir.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(
        json.dumps(
            dict(
                image=row["image_sha256"],
                confounds=row["confounds_sha256"],
                image_path=row["image_path"],
                confounds_path=row["confounds_path"],
                atlas=atlas_hash,
                settings=settings.to_dict(),
                tr=row["sampling_interval_seconds"],
                code=code_digest(),
                dependencies=versions(),
            ),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    result_cache = Path(cache_dir) / "scan_results" / signature
    if (result_cache / "complete.json").is_file():
        cached = json.loads((result_cache / "complete.json").read_text())
        if all(
            (result_cache / name).is_file() and sha256(result_cache / name) == digest
            for name, digest in cached["artifacts"].items()
        ):
            for name in cached["artifacts"]:
                if name != "result.json":
                    shutil.copy2(result_cache / name, scan_dir / name)
            payload = json.loads((result_cache / "result.json").read_text())
            restored = dict(payload["row"])
            restored.update({k: v for k, v in row.items() if k != "processing_status"})
            restored["analysis_cache_hit"] = True
            return (
                restored,
                payload["issues"],
                payload["motion"],
                payload["retention"],
                payload["comparison"],
            )
    issues, retention, comparison = [], [], []
    frames = frame_table(df, row["sampling_interval_seconds"], settings)
    frames.insert(0, "scan_key", key)
    frames.to_csv(scan_dir / "frames.csv", index=False)
    motion = []
    for population, subset in [("all_acquired", frames)] + [
        (c, frames.loc[frames[f"{c}_keep"]]) for c in CONDITIONS
    ]:
        motion.append(dict(scan_key=key, population=population, **motion_summary(subset)))
    plot_brain(mean, scan_dir / "brain.png", row["participant_id"])
    plot_motion(frames, scan_dir / "motion.png", settings)
    if frames.high_fd.any():
        issues.append(
            issue(
                key,
                "review",
                "high_fd_present",
                f"{int(frames.high_fd.sum())} frames exceed {settings.fd_threshold_mm} mm",
                row["confounds_path"],
                rows=frames.loc[frames.high_fd, "original_index"].tolist(),
                stage="quality",
                explanations="Elevated FD indicates estimated between-frame head displacement; it does not identify a unique cause.",
                action="Inspect motion and retained-data counts; compare the two fixed conditions.",
            )
        )
    if frames.non_steady_state.any():
        issues.append(
            issue(
                key,
                "review",
                "non_steady_state_present",
                f"{int(frames.non_steady_state.sum())} flagged frames excluded from both conditions",
                row["confounds_path"],
                rows=frames.loc[frames.non_steady_state, "original_index"].tolist(),
                stage="quality",
                explanations="Source confounds flag potential non-steady-state acquisition frames.",
                action="Inspect the supplied indicators and document their common exclusion.",
            )
        )
    condition_results = {}
    try:
        confounds, loader_mask = load_confounds(
            row["image_path"],
            strategy=["high_pass", "motion", "wm_csf"],
            motion="basic",
            wm_csf="basic",
            demean=False,
        )
        expected_columns = selected_columns(df)
        if set(confounds.columns) != set(expected_columns) or len(confounds) != len(df):
            raise ValueError(
                f"load_confounds selections disagree with raw TSV: {list(confounds.columns)} vs {expected_columns}"
            )
        np.testing.assert_allclose(
            confounds.to_numpy(),
            df[confounds.columns].to_numpy(dtype=float),
            atol=1e-10,
            rtol=1e-10,
        )
        expected_loader_kept = np.flatnonzero(~df[nss_columns(df)].eq(1).any(axis=1).to_numpy())
        actual_loader_kept = np.arange(len(df)) if loader_mask is None else np.asarray(loader_mask)
        if not np.array_equal(actual_loader_kept, expected_loader_kept):
            raise ValueError(
                "load_confounds non-steady-state mask disagrees with original source flags"
            )
        signals, regions, hit = extract_regions(
            row["image_path"], atlas, cache_dir, row["image_sha256"], atlas_hash
        )
        row["regional_cache_hit"] = hit
        if signals.shape[0] != row["n_volumes"]:
            raise ValueError("Regional signal rows do not match image timepoints")
        row["n_regions"] = len(regions)
        row["selected_nuisance_columns"] = json.dumps(list(confounds.columns))
        regions.to_csv(scan_dir / "regions.csv", index=False)
        confounds.to_csv(scan_dir / "selected_confounds.csv", index=False)
        raw_frame = pd.DataFrame(signals, columns=[f"roi_{r}" for r in regions.region_id])
        raw_frame.insert(0, "time_seconds", frames.time_seconds)
        raw_frame.insert(0, "original_index", frames.original_index)
        raw_frame.to_csv(scan_dir / "raw_timeseries.csv", index=False)
        for condition in CONDITIONS:
            kept = frames.loc[frames[f"{condition}_keep"], "original_index"].to_numpy(dtype=int)
            with warnings.catch_warnings(record=True) as emitted:
                warnings.simplefilter("always")
                result = clean_condition(
                    signals, confounds, kept, row["sampling_interval_seconds"], settings
                )
            invalid_columns = result.get("invalid_region_columns", [])
            if invalid_columns:
                affected = regions.iloc[invalid_columns]
                labels = [f"region {r.region_id} ({r.region_name})" for r in affected.itertuples()]
                stds = np.std(signals[kept][:, invalid_columns], axis=0)
                result["reason"] += (
                    "; affected "
                    + ", ".join(labels)
                    + f"; retained raw standard deviations={stds.tolist()}"
                )
            condition_results[condition] = result
            for warning in emitted:
                issues.append(
                    issue(
                        key,
                        "info",
                        "library_warning",
                        str(warning.message),
                        row["image_path"],
                        stage="analysis",
                        explanations="Message emitted by the pinned analysis dependency.",
                        action="Retain dependency versions and review this message before upgrading.",
                    )
                )
            retention.append(
                dict(
                    scan_key=key,
                    condition=condition,
                    status=result["status"],
                    reason=result["reason"],
                    retained_volumes=result["retained_volumes"],
                    retained_minutes=result["retained_minutes"],
                    design_rank=result["design_rank"],
                    residual_df=result["residual_df"],
                    acquired_volumes=len(frames),
                    excluded_volumes=len(frames) - len(kept),
                    retained_fraction=len(kept) / len(frames),
                    common_excluded_volumes=int((~frames.baseline_keep).sum()),
                    additional_fd_excluded_volumes=int(
                        (frames.baseline_keep & ~frames.censored_keep).sum()
                    )
                    if condition == "censored"
                    else 0,
                )
            )
            if len(kept) / len(frames) < 0.8:
                issues.append(
                    issue(
                        key,
                        "review",
                        "retention_below_80_percent",
                        f"{condition}: {len(kept)}/{len(frames)} retained; project review rule <80%",
                        row["confounds_path"],
                        stage="quality",
                        explanations="Initial-frame, source non-steady-state and/or FD exclusions reduce usable samples.",
                        action="Review excluded frames and guardrails; this flag is not an automatic scientific rejection.",
                    )
                )
            if result["status"] == "complete":
                cleaned_frame = pd.DataFrame(
                    result["signals"], columns=[f"roi_{r}" for r in regions.region_id]
                )
                cleaned_frame.insert(0, "time_seconds", kept * row["sampling_interval_seconds"])
                cleaned_frame.insert(0, "original_index", kept)
                cleaned_frame.to_csv(scan_dir / f"cleaned_{condition}.csv", index=False)
                pd.DataFrame(
                    result["correlation"], index=regions.region_id, columns=regions.region_id
                ).to_csv(scan_dir / f"correlation_{condition}.csv", index_label="region_id")
            else:
                issues.append(
                    issue(
                        key,
                        "unavailable",
                        "condition_unavailable",
                        f"{condition}: {result['reason']}",
                        row["image_path"],
                        stage="analysis",
                        explanations="Insufficient usable samples, limited spatial coverage, masking, or absent temporal variation are possible explanations; none is established as the root cause.",
                        action="Review retained data and affected regions; do not interpret or replace the unavailable matrix.",
                    )
                )
        if all(condition_results[c]["status"] == "complete" for c in CONDITIONS):
            comparison.append(
                dict(
                    scan_key=key,
                    **compare_matrices(*(condition_results[c]["correlation"] for c in CONDITIONS)),
                )
            )
            plot_matrices(
                *(condition_results[c]["correlation"] for c in CONDITIONS),
                regions,
                scan_dir / "matrices.png",
            )
            row["processing_status"] = "complete"
        else:
            row["processing_status"] = "analysis_unavailable"
        write_json(
            scan_dir / "methods.json",
            dict(
                selected_nuisance_columns=list(confounds.columns),
                loader_non_steady_state_indices=sorted(
                    set(range(len(df))) - set(actual_loader_kept.tolist())
                ),
                region_ids=regions.region_id.tolist(),
                conditions={
                    c: {k: v for k, v in r.items() if k not in ("signals", "correlation")}
                    for c, r in condition_results.items()
                },
            ),
        )
    except (ValueError, OSError, AssertionError, KeyError, RuntimeError) as exc:
        row["processing_status"] = "analysis_unavailable"
        issues.append(
            issue(
                key,
                "unavailable",
                "analysis_failure",
                f"{type(exc).__name__}: {exc}",
                row["image_path"],
                stage="analysis",
                action="Inspect the exact failure and saved inputs/settings; rerun after resolving it. Do not substitute observations.",
            )
        )
        write_json(
            scan_dir / "failure.json", dict(error=str(exc), traceback=traceback.format_exc())
        )
    row["analysis_cache_hit"] = False
    if not any(x["rule"] == "analysis_failure" for x in issues):
        result_cache.mkdir(parents=True, exist_ok=True)
        for path in scan_dir.iterdir():
            if path.is_file():
                shutil.copy2(path, result_cache / path.name)
        write_json(
            result_cache / "result.json",
            dict(row=row, issues=issues, motion=motion, retention=retention, comparison=comparison),
        )
        write_json(
            result_cache / "complete.json",
            dict(
                artifacts={
                    p.name: sha256(p)
                    for p in result_cache.iterdir()
                    if p.is_file() and p.name != "complete.json"
                }
            ),
        )
    return row, issues, motion, retention, comparison


def export_tables(out, manifest, issues, motion, retention, comparisons):
    manifest_table = (
        pd.DataFrame(manifest)
        if manifest
        else pd.DataFrame(
            columns=[
                "scan_key",
                "participant_id",
                "data_kind",
                "processing_status",
                "image_path",
                "confounds_path",
            ]
        )
    )
    manifest_table.to_csv(out / "manifest.csv", index=False)
    pd.DataFrame(issues, columns=ISSUE_COLUMNS).to_csv(out / "issues.csv", index=False)
    pd.DataFrame(
        motion,
        columns=[
            "scan_key",
            "population",
            "n_volumes",
            "n_fd_observed",
            "mean_fd_mm",
            "median_fd_mm",
            "p95_fd_mm",
            "max_fd_mm",
            "high_fd_count",
            "high_fd_fraction",
        ],
    ).to_csv(out / "motion_summary.csv", index=False)
    pd.DataFrame(
        retention,
        columns=[
            "scan_key",
            "condition",
            "status",
            "reason",
            "retained_volumes",
            "retained_minutes",
            "design_rank",
            "residual_df",
            "acquired_volumes",
            "excluded_volumes",
            "retained_fraction",
            "common_excluded_volumes",
            "additional_fd_excluded_volumes",
        ],
    ).to_csv(out / "retention.csv", index=False)
    pd.DataFrame(
        comparisons,
        columns=[
            "scan_key",
            "n_unique_edges",
            "mean_absolute_change",
            "max_absolute_change",
            "mean_signed_change",
            "edge_similarity",
        ],
    ).to_csv(out / "comparison.csv", index=False)
    frame_tables, region_tables, series_tables, matrices = [], [], [], []
    for row in manifest:
        directory = out / "scans" / row["scan_key"]
        if (directory / "frames.csv").exists():
            frame_tables.append(pd.read_csv(directory / "frames.csv"))
        if (directory / "regions.csv").exists():
            regions = pd.read_csv(directory / "regions.csv")
            region_tables.append(regions.assign(scan_key=row["scan_key"]))
            for condition in ("raw", *CONDITIONS):
                filename = (
                    "raw_timeseries.csv" if condition == "raw" else f"cleaned_{condition}.csv"
                )
                if (directory / filename).exists():
                    series_tables.append(
                        pd.read_csv(directory / filename).assign(
                            scan_key=row["scan_key"], condition=condition
                        )
                    )
            for condition in CONDITIONS:
                path = directory / f"correlation_{condition}.csv"
                if path.exists():
                    matrix = pd.read_csv(path, index_col=0)
                    long = (
                        matrix.rename_axis("region_i")
                        .reset_index()
                        .melt(id_vars="region_i", var_name="region_j", value_name="correlation")
                    )
                    matrices.append(long.assign(scan_key=row["scan_key"], condition=condition))
    for name, tables, empty_columns in [
        ("frame_exclusions", frame_tables, ["scan_key", "original_index"]),
        ("regions", region_tables, ["scan_key", "region_id", "region_name"]),
        ("regional_timeseries", series_tables, ["scan_key", "condition", "original_index"]),
        ("matrices", matrices, ["scan_key", "condition", "region_i", "region_j", "correlation"]),
    ]:
        (
            pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=empty_columns)
        ).to_csv(out / f"{name}.csv", index=False)


def run_study(n_subjects, data_dir, output_dir, cache_dir, *, checkpoint=None):
    from .reporting import write_report

    settings = Settings()
    out = Path(output_dir).resolve()
    if (
        out == Path(data_dir).resolve()
        or out in Path(data_dir).resolve().parents
        or Path(data_dir).resolve() in out.parents
    ):
        raise ValueError("Output and preserved source directories must not overlap")
    if (out / "completed.json").is_file():
        prior_provenance = json.loads((out / "provenance.json").read_text())
        if (
            prior_provenance.get("code_sha256") == code_digest()
            and prior_provenance.get("dependencies") == versions()
            and prior_provenance.get("requested_n_subjects") == n_subjects
        ):
            try:
                verified = verify_bundle(out)
                previous_issues = pd.read_csv(out / "issues.csv")
                retryable = previous_issues.rule.isin(
                    ["analysis_failure", "workflow_failure"]
                ).any()
                if verified["acquired_scans"] == n_subjects and not retryable:
                    print(f"Reusing verified completed bundle: {out}", flush=True)
                    return verified["all_analyses_complete"]
            except (ValueError, OSError, AssertionError):
                pass
    archived = prepare_output(out)
    if archived:
        print(f"Preserved previous outputs at {archived}", flush=True)
    complete = out / "completed.json"
    write_json(
        out / "attempt.json", dict(started_at_utc=now(), n_subjects=n_subjects, status="running")
    )
    manifest, issues, motion, retention, comparisons = [], [], [], [], []
    provenance = dict(
        project_version=__version__,
        code_sha256=code_digest(),
        dependencies=versions(),
        sources=SOURCES,
        original_dataset_version="ds000228/1.0.0 (documented origin)",
        osf_derivative_version="unknown: no release identifier supplied by fetcher",
        fmriprep_version="unknown: not supplied with downloaded TSV/NIfTI files",
        dataset_citation="Richardson, Lisandrelli, Riobueno-Naylor & Saxe (2018). Development of the social brain from age three to twelve years. Nature Communications 9, 1027.",
        requested_n_subjects=n_subjects,
        selected_participant_ids=[],
        data_kind="real",
        created_at_utc=now(),
    )
    snapshot = out / "pipeline_source"
    snapshot.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, snapshot / path.name)
    write_json(
        out / "settings.json",
        dict(
            **asdict(settings),
            estimator="ordinary Pearson correlation; numpy.corrcoef",
            strategy=["high_pass", "motion", "wm_csf"],
            motion="basic",
            wm_csf="basic",
            demean=False,
            extraction=dict(
                standardize=None,
                detrend=False,
                smoothing_fwhm=None,
                high_pass=None,
                low_pass=None,
                confounds=None,
                sample_mask=None,
                resampling="nearest-neighbor atlas to image grid",
            ),
            cleaning=dict(
                detrend=False,
                standardize="zscore_sample",
                standardize_confounds=True,
                filter=False,
                low_pass=None,
                high_pass=None,
                ensure_finite=False,
            ),
            initial_policy="Exclude original index zero in both conditions, even if source FD/DVARS are stored as zero; exclude index zero from difference-based summaries. Preserve source values.",
            common_exclusions="initial frame and supplied non-steady-state flags",
            censor_rule="FD > 0.5 mm only; no neighboring frames, DVARS rule, or segment pruning",
            residual_df_rule="retained rows minus rank([intercept, retained nuisance columns]); nominal, not independent physiological df",
            frequency_cutoff="unknown: supplied cosine drift columns used without inventing a cutoff",
            symmetric_split=False,
            matrix_scale=[-1, 1],
            difference_scale=[-2, 2],
        ),
    )
    try:
        if n_subjects == 8:
            if checkpoint is None:
                raise ValueError(
                    "Eight-adult acquisition requires --checkpoint outputs/one with a verified successful one-participant result"
                )
            verified = verify_bundle(checkpoint)
            if verified["n_subjects"] != 1 or not verified["all_analyses_complete"]:
                raise ValueError(
                    "One-participant checkpoint has not passed both analysis conditions"
                )
        data, acquisition = fetch_adults(n_subjects, data_dir)
        provenance["acquisition"] = acquisition
        records = join_sources(data.func, data.confounds, data.phenotypic)
        if len(records) != n_subjects or len({x["participant_id"] for x in records}) != n_subjects:
            raise ValueError(
                f"Requested {n_subjects} distinct participants; acquired {len(records)} scans"
            )
        provenance["selected_participant_ids"] = [x["participant_id"] for x in records]
        if n_subjects == 8:
            prior = json.loads((Path(checkpoint) / "provenance.json").read_text())[
                "selected_participant_ids"
            ]
            if not set(prior).issubset(provenance["selected_participant_ids"]):
                raise ValueError(
                    "Eight-adult selection did not preserve the verified first participant"
                )
        data.phenotypic.to_csv(out / "participants.csv", index=False)
        (out / "dataset_description.txt").write_text(str(data.description))
        provenance["fetcher_t_r_seconds"] = float(data.t_r)
        write_json(
            out / "source_inspection.json",
            {
                r["scan_key"]: {
                    "confound_columns": list(
                        pd.read_csv(r["confounds_path"], sep="\t", nrows=0).columns
                    ),
                    "phenotype": r["phenotype"],
                    "image_path": r["image_path"],
                    "confounds_path": r["confounds_path"],
                }
                for r in records
            },
        )
        atlas = datasets.fetch_atlas_harvard_oxford(
            settings.atlas,
            data_dir=str(Path(data_dir).resolve()),
            symmetric_split=False,
            resume=True,
            verbose=1,
        )
        atlas_path = Path(atlas.filename) if isinstance(atlas.filename, str) else None
        if atlas_path is None or not atlas_path.is_file():
            raise ValueError("Atlas fetcher did not provide a readable source atlas filename")
        atlas_hash = sha256(atlas_path)
        provenance["atlas"] = dict(
            name=settings.atlas,
            source_path=str(atlas_path),
            sha256=atlas_hash,
            description=atlas.description,
            labels=atlas.lut.to_dict(orient="records"),
            symmetric_split=False,
        )
        for record in records:
            print(f"Inspecting {record['scan_key']}", flush=True)
            row, df, scan_issues, mean = inspect_scan(record, data.t_r)
            issues.extend(scan_issues)
            if row["processing_status"] == "validated":
                row, more, scan_motion, scan_retention, scan_comparison = analyze_scan(
                    row, df, mean, atlas, atlas_hash, out, cache_dir, settings
                )
                issues.extend(more)
                motion.extend(scan_motion)
                retention.extend(scan_retention)
                comparisons.extend(scan_comparison)
            manifest.append(row)
            export_tables(out, manifest, issues, motion, retention, comparisons)
            write_json(out / "provenance.json", provenance)
            print(f"{record['scan_key']}: {row['processing_status']}", flush=True)
    except Exception as exc:
        issues.append(
            issue(
                "acquisition_or_setup",
                "error",
                "workflow_failure",
                f"{type(exc).__name__}: {exc}",
                str(data_dir),
                stage="acquisition",
                action="Read failure.json and rerun the same command after resolving the stated failure; completed downloads and outputs are preserved.",
            )
        )
        write_json(
            out / "failure.json",
            dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()),
        )
    export_tables(out, manifest, issues, motion, retention, comparisons)
    provenance["finished_at_utc"] = now()
    write_json(out / "provenance.json", provenance)
    all_complete = (
        len(manifest) == n_subjects
        and all(x["processing_status"] == "complete" for x in manifest)
        and not any(x["severity"] == "error" for x in issues)
    )
    write_json(
        out / "summary.json",
        dict(
            n_subjects=n_subjects,
            acquired_scans=len(manifest),
            all_analyses_complete=all_complete,
            status="complete" if all_complete else "partial",
            comparisons_complete=len(comparisons),
            data_kind="real",
        ),
    )
    write_report(out)
    artifacts = {
        str(p.relative_to(out)): sha256(p)
        for p in sorted(out.rglob("*"))
        if p.is_file()
        and p.name not in ("completed.json", "attempt.json")
        and not p.name.endswith(".tmp")
    }
    write_json(
        complete,
        dict(
            schema_version=1,
            status="complete" if all_complete else "partial",
            artifacts=artifacts,
            created_at_utc=now(),
        ),
    )
    write_json(
        out / "attempt.json",
        dict(
            finished_at_utc=now(),
            n_subjects=n_subjects,
            status="complete" if all_complete else "partial",
        ),
    )
    if all_complete:
        verify_bundle(out)
    return all_complete


def verify_bundle(directory, *, check_source_hashes=True):
    directory = Path(directory).resolve()
    marker = directory / "completed.json"
    if not marker.is_file():
        raise ValueError(f"No completed bundle marker at {directory}; run the CLI first")
    completed = json.loads(marker.read_text())
    for relative, digest in completed["artifacts"].items():
        if sha256(directory / relative) != digest:
            raise ValueError(f"Output artifact changed or incomplete: {relative}")
    summary = json.loads((directory / "summary.json").read_text())
    manifest = pd.read_csv(directory / "manifest.csv")
    if len(manifest) and not manifest.data_kind.eq("real").all():
        raise ValueError("Fixture or non-real observations present in real-data manifest")
    if manifest.scan_key.duplicated().any():
        raise ValueError("Duplicate scan keys in manifest")
    retention = pd.read_csv(directory / "retention.csv")
    motion = pd.read_csv(directory / "motion_summary.csv")
    for record in manifest.to_dict(orient="records"):
        if check_source_hashes:
            for source in ("image", "confounds"):
                if sha256(record[f"{source}_path"]) != record[f"{source}_sha256"]:
                    raise ValueError(f"Preserved source hash changed: {record[f'{source}_path']}")
        path = directory / "scans" / record["scan_key"]
        if not (path / "frames.csv").exists():
            continue
        frames = pd.read_csv(path / "frames.csv")
        if len(frames) != record["n_volumes"] or len(frames) != record["confound_rows"]:
            raise ValueError("Exported frames disagree with source alignment")
        np.testing.assert_array_equal(frames.original_index, np.arange(len(frames)))
        np.testing.assert_allclose(
            frames.time_seconds, frames.original_index * record["sampling_interval_seconds"]
        )
        if frames.loc[0, ["baseline_keep", "censored_keep", "fd_defined"]].any():
            raise ValueError("Initial-frame policy violated")
        for condition in CONDITIONS:
            rows = retention.loc[
                (retention.scan_key == record["scan_key"]) & (retention.condition == condition)
            ]
            if rows.empty:
                continue
            result = rows.iloc[0]
            kept = frames.loc[frames[f"{condition}_keep"], "original_index"].to_numpy()
            if len(kept) != result.retained_volumes or not np.isclose(
                result.retained_minutes, len(kept) * record["sampling_interval_seconds"] / 60
            ):
                raise ValueError("Retention denominator or duration mismatch")
            if result.status == "complete":
                series = pd.read_csv(path / f"cleaned_{condition}.csv")
                np.testing.assert_array_equal(series.original_index, kept)
                matrix = pd.read_csv(path / f"correlation_{condition}.csv", index_col=0)
                np.testing.assert_allclose(
                    matrix.to_numpy(),
                    np.corrcoef(series.filter(regex=r"^roi_\d+$").to_numpy(), rowvar=False),
                    atol=1e-8,
                )
        expected_motion = motion_summary(frames)
        actual = motion.loc[
            (motion.scan_key == record["scan_key"]) & (motion.population == "all_acquired")
        ].iloc[0]
        for name, value in expected_motion.items():
            if value is not None and not np.isclose(actual[name], value):
                raise ValueError(f"Motion summary mismatch: {name}")
    return summary
