"""Auditable integrity, motion-mask and descriptive-correlation rules."""

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.signal import clean

MOTION = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
NUISANCE = MOTION + ["csf", "white_matter"]
REQUIRED = NUISANCE + ["framewise_displacement"]
ENTITY_NAMES = ("sub", "ses", "task", "acq", "run", "echo")
CONDITIONS = ("baseline", "censored")


@dataclass(frozen=True)
class Settings:
    fd_threshold_mm: float = 0.5
    minimum_seconds: float = 120.0
    minimum_residual_df: int = 20
    atlas: str = "cort-maxprob-thr25-2mm"
    schema_version: int = 1

    def to_dict(self):
        return asdict(self)


def issue(
    scan_key,
    severity,
    rule,
    evidence,
    path,
    *,
    column="",
    rows=None,
    explanations="Input may be incomplete, malformed, or paired incorrectly.",
    action="Inspect the cited source and reacquire or correct the pairing; preserve originals.",
    stage="integrity",
):
    return dict(
        scan_key=scan_key,
        severity=severity,
        stage=stage,
        rule=rule,
        evidence=str(evidence),
        file_location=str(path),
        column=column,
        rows=[] if rows is None else [int(x) for x in rows],
        possible_explanations=explanations,
        recommended_action=action,
    )


def entities(path):
    found = dict(re.findall(r"(?:^|_)(sub|ses|task|acq|run|echo)-([^_\.]+)", Path(path).name))
    if "sub" not in found or "task" not in found:
        raise ValueError(f"Participant/task entities missing in {path}")
    return {name: found.get(name) for name in ENTITY_NAMES}


def key_for(path):
    ent = entities(path)
    return "_".join(f"{name}-{ent[name]}" for name in ENTITY_NAMES if ent[name] is not None)


def join_sources(funcs, confounds, phenotypic):
    """Require unambiguous one-to-one scan and participant identity joins."""

    def index_paths(paths):
        index = {}
        for path in paths:
            key = key_for(path)
            if key in index:
                raise ValueError(f"Duplicate scan key {key}: {index[key]} and {path}")
            index[key] = str(path)
        return index

    images, tables = index_paths(funcs), index_paths(confounds)
    if images.keys() != tables.keys():
        raise ValueError(
            f"Image/confound scan keys differ: image-only={sorted(images.keys() - tables.keys())}; confound-only={sorted(tables.keys() - images.keys())}"
        )
    if "participant_id" not in phenotypic or phenotypic.participant_id.duplicated().any():
        raise ValueError("Missing or duplicate phenotype participant_id")
    phenotype = phenotypic.set_index("participant_id")
    result = []
    for key, path in sorted(images.items()):
        ent = entities(path)
        participant_id = f"sub-{ent['sub']}"
        if participant_id not in phenotype.index:
            raise ValueError(f"No phenotype match for {participant_id}: {path}")
        row = phenotype.loc[participant_id]
        if "Child_Adult" not in row or str(row.Child_Adult).lower() != "adult":
            raise ValueError(f"Adult selection not confirmed for {participant_id}: {row.to_dict()}")
        if "Age" in row and (not np.isfinite(float(row.Age)) or float(row.Age) < 18):
            raise ValueError(f"Non-adult/invalid age for {participant_id}: {row.Age}")
        result.append(
            dict(
                scan_key=key,
                participant_id=participant_id,
                session=ent["ses"],
                task=ent["task"],
                acquisition=ent["acq"],
                run=ent["run"],
                echo=ent["echo"],
                image_path=path,
                confounds_path=tables[key],
                phenotype=row.to_dict(),
            )
        )
    if not result:
        raise ValueError("No scans returned")
    return result


def selected_columns(df):
    return NUISANCE + sorted(c for c in df.columns if re.fullmatch(r"cosine\d+", c))


def nss_columns(df):
    return [c for c in df.columns if c.startswith("non_steady_state_outlier")]


def expected_initial_missing(column):
    return column in {"framewise_displacement", "dvars", "std_dvars"} or bool(
        re.fullmatch(r".+_derivative1(?:_power2)?", column)
    )


def validate_confounds(df, scan_key, path):
    issues = []
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        issues.append(issue(scan_key, "error", "required_columns", f"Missing {missing}", path))
    if not any(re.fullmatch(r"cosine\d+", c) for c in df.columns):
        issues.append(
            issue(
                scan_key,
                "error",
                "cosine_regressors",
                "No supplied cosine drift columns",
                path,
                action="Check the full confounds source; this fixed strategy cannot run without its supplied drift regressors.",
            )
        )
    required = set(selected_columns(df) + ["framewise_displacement"] + nss_columns(df))
    for column in df.columns:
        severity = "error" if column in required else "review"
        numeric = pd.to_numeric(df[column], errors="coerce")
        bad_numeric = df[column].notna() & numeric.isna()
        if bad_numeric.any():
            issues.append(
                issue(
                    scan_key,
                    severity,
                    "numeric_values",
                    f"{bad_numeric.sum()} non-numeric values",
                    path,
                    column=column,
                    rows=np.flatnonzero(bad_numeric),
                )
            )
        missing_mask = df[column].isna().to_numpy()
        allowed = np.zeros(len(df), dtype=bool)
        if len(df) and expected_initial_missing(column):
            allowed[0] = True
        unexpected = missing_mask & ~allowed
        if unexpected.any():
            issues.append(
                issue(
                    scan_key,
                    severity,
                    "unexpected_missingness",
                    f"{unexpected.sum()} unexpected missing values",
                    path,
                    column=column,
                    rows=np.flatnonzero(unexpected),
                )
            )
        expected = missing_mask & allowed
        if expected.any():
            issues.append(
                issue(
                    scan_key,
                    "info",
                    "expected_initial_undefined",
                    "Undefined initial difference-based measurement; preserved as missing",
                    path,
                    column=column,
                    rows=[0],
                    explanations="Difference-based quantities have no preceding frame at index zero.",
                    action="Exclude index zero in both conditions and omit undefined values from summary denominators.",
                )
            )
        infinite = np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan))
        if infinite.any():
            issues.append(
                issue(
                    scan_key,
                    severity,
                    "finite_values",
                    f"{infinite.sum()} infinite values",
                    path,
                    column=column,
                    rows=np.flatnonzero(infinite),
                )
            )
        if column in nss_columns(df):
            invalid = ~numeric.isin([0, 1])
            if invalid.any():
                issues.append(
                    issue(
                        scan_key,
                        "error",
                        "binary_nss",
                        "Non-steady-state flags must be zero or one",
                        path,
                        column=column,
                        rows=np.flatnonzero(invalid),
                    )
                )
        if column in selected_columns(df) and numeric.nunique() <= 1:
            issues.append(
                issue(
                    scan_key,
                    "review",
                    "constant_nuisance",
                    f"{column} is constant",
                    path,
                    column=column,
                    explanations="The source regressor may have no variation in this scan.",
                    action="Review the source; nominal degrees of freedom use measured design rank.",
                )
            )
    if "framewise_displacement" in df:
        negative = pd.to_numeric(df.framewise_displacement, errors="coerce") < 0
        if negative.any():
            issues.append(
                issue(
                    scan_key,
                    "error",
                    "nonnegative_fd",
                    "FD is negative",
                    path,
                    column="framewise_displacement",
                    rows=np.flatnonzero(negative),
                )
            )
    return issues


def frame_table(df, tr, settings):
    n = len(df)
    index = np.arange(n)
    nss = df[nss_columns(df)].eq(1).any(axis=1).to_numpy()
    fd = df.framewise_displacement.to_numpy(dtype=float)
    initial = index == 0
    common = initial | nss
    high = fd > settings.fd_threshold_mm
    result = pd.DataFrame(
        dict(
            original_index=index,
            time_seconds=index * tr,
            framewise_displacement=fd,
            initial_frame=initial,
            fd_defined=(index > 0) & np.isfinite(fd),
            non_steady_state=nss,
            high_fd=high,
            baseline_keep=~common,
            censored_keep=~(common | high),
        )
    )
    for condition in CONDITIONS:
        result[f"{condition}_exclusion_reasons"] = [
            ";".join(
                reason
                for reason, flag in [
                    ("initial_frame", initial[i]),
                    ("non_steady_state", nss[i]),
                    ("fd_above_threshold", high[i] and condition == "censored"),
                ]
                if flag
            )
            for i in index
        ]
    for column in ("dvars", "std_dvars"):
        if column in df:
            result[column] = pd.to_numeric(df[column], errors="coerce")
    return result


def motion_summary(frames):
    fd = frames.framewise_displacement.to_numpy(dtype=float)
    defined = frames.fd_defined.to_numpy(dtype=bool) & np.isfinite(fd)
    observed = fd[defined]
    high_count = int(frames.loc[defined, "high_fd"].sum())
    return dict(
        n_volumes=len(frames),
        n_fd_observed=len(observed),
        mean_fd_mm=float(np.mean(observed)) if len(observed) else None,
        median_fd_mm=float(np.median(observed)) if len(observed) else None,
        p95_fd_mm=float(np.percentile(observed, 95)) if len(observed) else None,
        max_fd_mm=float(np.max(observed)) if len(observed) else None,
        high_fd_count=high_count,
        high_fd_fraction=high_count / len(observed) if len(observed) else None,
    )


def clean_condition(signals, confounds, kept, tr, settings):
    signals = np.asarray(signals, dtype=float)
    c = np.asarray(confounds, dtype=float)
    kept = np.asarray(kept)
    if signals.ndim != 2 or c.ndim != 2 or signals.shape[0] != c.shape[0]:
        raise ValueError("Full-length signal/confound rows must align before sample_mask")
    if (
        kept.dtype.kind not in "iu"
        or kept.ndim != 1
        or np.any(np.diff(kept) <= 0)
        or np.any(kept < 0)
        or np.any(kept >= len(signals))
    ):
        raise ValueError("sample_mask must contain sorted unique original integer indices")
    finite = np.isfinite(signals).all() and np.isfinite(c).all()
    design = np.column_stack([np.ones(len(kept)), c[kept]])
    rank = int(np.linalg.matrix_rank(design)) if len(kept) and finite else 0
    result = dict(
        status="unavailable",
        reason="",
        retained_volumes=len(kept),
        retained_minutes=len(kept) * tr / 60,
        design_rank=rank,
        residual_df=len(kept) - rank,
        signals=None,
        correlation=None,
        invalid_region_columns=[],
    )
    if not finite:
        result.update(
            reason="Non-finite regional signals or nuisance values; no matrix calculated.",
            design_rank=None,
            residual_df=None,
        )
        return result
    if (
        len(kept) * tr < settings.minimum_seconds
        or result["residual_df"] < settings.minimum_residual_df
    ):
        result["reason"] = (
            f"Usable-data guardrail: {len(kept) * tr:g} seconds and {result['residual_df']} nominal residual df; need {settings.minimum_seconds:g} seconds and {settings.minimum_residual_df} df."
        )
        return result
    constant = np.flatnonzero(np.std(signals[kept], axis=0) <= 1e-12)
    if len(constant):
        result["reason"] = f"Constant regional signal columns: {constant.tolist()}"
        result["invalid_region_columns"] = constant.tolist()
        return result
    cleaned = clean(
        signals,
        confounds=c,
        sample_mask=kept,
        detrend=False,
        standardize="zscore_sample",
        standardize_confounds=True,
        filter=False,
        low_pass=None,
        high_pass=None,
        ensure_finite=False,
        t_r=tr,
    )
    variance = np.std(cleaned, axis=0, ddof=1)
    if not np.isfinite(cleaned).all() or np.any(variance <= 1e-12):
        result["reason"] = "Non-finite or constant residual regional signal after cleaning."
        result["invalid_region_columns"] = np.flatnonzero(
            ~np.isfinite(variance) | (variance <= 1e-12)
        ).tolist()
        return result
    correlation = np.corrcoef(cleaned, rowvar=False)
    if not valid_matrix(correlation):
        result["reason"] = (
            "Correlation matrix failed finiteness, symmetry, diagonal or bounds checks."
        )
        return result
    result.update(status="complete", signals=cleaned, correlation=correlation)
    return result


def valid_matrix(matrix):
    return (
        matrix.ndim == 2
        and matrix.shape[0] == matrix.shape[1]
        and matrix.shape[0] >= 2
        and np.isfinite(matrix).all()
        and np.allclose(matrix, matrix.T, atol=1e-10)
        and np.allclose(np.diag(matrix), 1, atol=1e-10)
        and np.max(np.abs(matrix)) <= 1 + 1e-10
    )


def compare_matrices(baseline, censored):
    if baseline.shape != censored.shape or not valid_matrix(baseline) or not valid_matrix(censored):
        raise ValueError(
            "Comparable matrices must have identical validated region identities and shape"
        )
    upper = np.triu_indices_from(baseline, k=1)
    a, b = baseline[upper], censored[upper]
    delta = b - a
    similarity = float(np.corrcoef(a, b)[0, 1]) if min(a.std(), b.std()) > 1e-12 else None
    return dict(
        n_unique_edges=len(a),
        mean_absolute_change=float(np.abs(delta).mean()),
        max_absolute_change=float(np.abs(delta).max()),
        mean_signed_change=float(delta.mean()),
        edge_similarity=similarity,
    )
