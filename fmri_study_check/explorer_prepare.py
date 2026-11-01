"""Prepare immutable display assets explicitly; the Streamlit UI never imports this module."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img

VERSION = 1
SOURCES = ("tutorial", "archive")
SETTINGS = {
    "mean_image": "Arithmetic mean of all acquired frames; display-only float32 volume",
    "regional_series": "Uncleaned full-run parcel means; original indices and source intensity units",
    "tutorial_support": "Exact stored-code temporal variation / resampled tutorial parcel voxel centers",
    "archive_support": "Known included SPM8 mask locations / all original native 2-mm atlas parcel centers, including out-of-grid locations",
    "unknown_policy": "Separate known-inclusion and unknown-membership bits; never impute source mask NaNs",
    "atlas_interpolation": "Nearest-neighbor; no new registration or replacement atlas",
    "fd_policy": "Saved primary masks; frame zero excluded from both; additional censoring strictly FD > 0.5 mm",
    "biological_interpretation": "Variation and mask overlap do not establish biological signal or denoising accuracy",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_index(records):
    result = {}
    for record in records:
        pid, role = record["participant_id"], record["role"]
        if not record.get("source_filename", "").startswith(pid + "_"):
            raise ValueError(f"Source filename does not match participant: {pid}/{role}")
        if (pid, role) in result:
            raise ValueError(f"Duplicate source participant/role: {pid}/{role}")
        result[pid, role] = record
    return result


def support_summary(roi, included, world_x, unknown=None):
    roi, included, x = np.asarray(roi), np.asarray(included), np.asarray(world_x)
    unknown = np.zeros(roi.shape, dtype=bool) if unknown is None else np.asarray(unknown)
    if any(a.shape != roi.shape for a in (included, x, unknown)) or not np.isfinite(x).all():
        raise ValueError("Support, coordinate and unknown grids must agree")
    result = {}
    for name, side in {
        "whole": np.ones(roi.shape, dtype=bool),
        "left": x < -1e-6,
        "midline": np.abs(x) <= 1e-6,
        "right": x > 1e-6,
    }.items():
        selected = roi & side
        total = int(selected.sum())
        count = int((selected & included).sum())
        missing = int((selected & unknown).sum())
        lower = count / total if total else None
        upper = (count + missing) / total if total else None
        result[name] = dict(
            total=total,
            included=count,
            unknown=missing,
            fraction=lower if not missing else None,
            lower=lower,
            upper=upper,
        )
    return result


def regional_series(values, labels, region_ids, slope=1.0, intercept=0.0):
    if values.ndim != 4 or labels.shape != values.shape[:3] or not np.isfinite(values).all():
        raise ValueError("Finite four-dimensional image data and matching labels are required")
    if not np.isfinite([slope, intercept]).all() or slope == 0:
        raise ValueError("Invalid source scaling")
    result = {}
    for region in region_ids:
        selected = values[labels == region]
        if not len(selected):
            result[str(region)] = None
            continue
        mean = selected.mean(axis=0, dtype=np.float64) * slope + intercept
        result[str(region)] = mean.tolist()
    return result


def frame_payload(frames, tr):
    if not np.isfinite(tr) or tr <= 0:
        raise ValueError("Invalid sampling time")
    n = len(frames)
    if not np.array_equal(frames.original_index, np.arange(n)) or not np.allclose(
        frames.time_seconds, np.arange(n) * tr
    ):
        raise ValueError("Original frame indices or acquisition times disagree")
    stored = pd.to_numeric(frames.framewise_displacement, errors="coerce").to_numpy(float)
    defined = frames.fd_defined.to_numpy(bool)
    if n == 0 or defined[0] or not defined[1:].all() or not np.isfinite(stored[defined]).all():
        raise ValueError("Unexpected undefined or non-finite FD observations")
    baseline = frames.baseline_keep.to_numpy(bool)
    censored = frames.censored_keep.to_numpy(bool)
    if baseline[0] or not np.array_equal(censored, baseline & ~(defined & (stored > 0.5))):
        raise ValueError("Saved censoring membership disagrees with the fixed FD rule")

    def reasons(name):
        return [None if pd.isna(v) else str(v) for v in frames[name]]

    return dict(
        original_index=np.arange(n).tolist(),
        time_seconds=(np.arange(n) * tr).tolist(),
        fd=[float(v) if d else None for v, d in zip(stored, defined, strict=True)],
        fd_stored=[float(v) if np.isfinite(v) else None for v in stored],
        baseline_keep=baseline.tolist(),
        censored_keep=censored.tolist(),
        baseline_reason=reasons("baseline_exclusion_reasons"),
        censored_reason=reasons("censored_exclusion_reasons"),
        n_fd_observed=int(defined.sum()),
        high_fd_count=int((stored[defined] > 0.5).sum()),
        mean_fd_mm=float(stored[defined].mean()) if defined.any() else None,
    )


def resample(values, source, target):
    image = nib.Nifti1Image(np.asarray(values, dtype=np.uint8), source.affine)
    return np.asarray(
        resample_to_img(
            image,
            target,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
            fill_value=0,
        ).dataobj
    )


def world_coordinates(image):
    coordinates = np.indices(image.shape[:3]).reshape(3, -1).T
    return nib.affines.apply_affine(image.affine, coordinates).reshape((*image.shape[:3], 3))


def region_focus(labels, coords, region):
    points = coords[labels == region]
    result = {}
    for name, keep in {
        "left": points[:, 0] < -1e-6,
        "right": points[:, 0] > 1e-6,
        "midline": np.abs(points[:, 0]) <= 1e-6,
    }.items():
        selected = points[keep]
        if len(selected):
            center = selected.mean(axis=0)
            result[name] = selected[np.argmin(np.sum((selected - center) ** 2, axis=1))].tolist()
    return result


def source_arrays(image):
    if (
        len(image.shape) != 4
        or not np.isfinite([image.dataobj.slope, image.dataobj.inter]).all()
        or image.dataobj.slope == 0
    ):
        raise ValueError("Invalid image dimensions or scaling")
    values = image.dataobj.get_unscaled()
    if not np.isfinite(values).all():
        raise ValueError("BOLD data contain non-finite values")
    mean = (
        values.mean(axis=3, dtype=np.float64) * image.dataobj.slope + image.dataobj.inter
    ).astype(np.float32)
    varying = values.min(axis=3) != values.max(axis=3)
    return values, mean, varying


def write_volume(path, values, reference):
    image = nib.Nifti1Image(values, reference.affine)
    nib.save(image, path)


def prepare_scan(root, directory, record, assets, atlas, names, expected, retention):
    pid, scan = record["participant_id"], record["scan_key"]
    frames = pd.read_csv(root / "outputs/eight/scans" / scan / "frames.csv")
    motion = frame_payload(frames, float(record["sampling_interval_seconds"]))
    n = int(record["n_volumes"])
    if len(frames) != n:
        raise ValueError("Image/primary-frame alignment disagrees")
    metadata = dict(
        schema_version=VERSION,
        participant_id=pid,
        scan_key=scan,
        n_volumes=n,
        tr_seconds=float(record["sampling_interval_seconds"]),
        duration_seconds=n * float(record["sampling_interval_seconds"]),
        primary_status=record["processing_status"],
        motion=motion,
        sources={},
        regions=[],
        settings=SETTINGS,
        primary_artifacts={
            name: sha256(root / "outputs/eight/scans" / scan / name)
            for name in ("frames.csv", "raw_timeseries.csv")
        },
    )
    metadata["primary_analysis"] = [
        dict(
            condition=r.condition,
            status=r.status,
            reason=None if pd.isna(r.reason) else r.reason,
            retained_volumes=int(r.retained_volumes),
            retained_minutes=float(r.retained_minutes),
            residual_df=int(r.residual_df),
        )
        for r in retention.loc[retention.scan_key == scan].itertuples()
    ]
    atlas_labels = np.asarray(atlas.dataobj)
    atlas_coords = world_coordinates(atlas)
    source_series, source_stats, focus = {}, {}, {}
    primary_raw = pd.read_csv(root / "outputs/eight/scans" / scan / "raw_timeseries.csv")
    comparison_raw = pd.read_csv(
        root / "investigations/eight-adult-coverage/scans" / pid / "source_roi_timeseries.csv"
    )
    for source in SOURCES:
        path = (
            Path(record["image_path"])
            if source == "tutorial"
            else root / assets[pid, "spm8_bold"]["path"]
        )
        image = nib.load(path)
        if image.shape[3] != n:
            raise ValueError(f"Source/frame count mismatch: {pid}/{source}")
        values, mean, varying = source_arrays(image)
        labels = np.asarray(
            resample_to_img(
                atlas, image, interpolation="nearest", force_resample=True, copy_header=True
            ).dataobj,
            dtype=np.uint8,
        )
        coords = world_coordinates(image)
        source_series[source] = regional_series(
            values, labels, sorted(names), image.dataobj.slope, image.dataobj.inter
        )
        source_stats[source], focus[source] = {}, {}
        if source == "archive":
            mask_image = nib.load(root / assets[pid, "spm8_mask"]["path"])
            if mask_image.shape != image.shape[:3] or not np.array_equal(
                mask_image.affine, image.affine
            ):
                raise ValueError("Archive BOLD/mask grid mismatch")
            mask = np.asarray(mask_image.dataobj)
            if np.isinf(mask).any() or not np.isin(mask[np.isfinite(mask)], [0, 1]).all():
                raise ValueError("Archive mask has invalid finite values")
            support, unknown = mask == 1, np.isnan(mask)
            atlas_support = resample(support, image, atlas).astype(bool)
            atlas_unknown = resample(unknown, image, atlas).astype(bool)
        else:
            support, unknown = varying, np.zeros(varying.shape, dtype=bool)
        for region in sorted(names):
            selected = labels == region
            reference = expected.loc[
                pid, "tutorial_4mm" if source == "tutorial" else "archive_spm8_2mm", region
            ]
            n_grid, n_vary = int(selected.sum()), int((selected & varying).sum())
            if n_grid != reference.grid_voxels or n_vary != reference.grid_varying_voxels:
                raise ValueError(
                    f"Display count differs from sealed audit: {pid}/{source}/{region}"
                )
            if source == "tutorial":
                measured = support_summary(selected, support, coords[..., 0], unknown)
                np.testing.assert_allclose(
                    source_series[source][str(region)],
                    primary_raw[f"roi_{region}"],
                    rtol=1e-10,
                    atol=1e-8,
                )
            else:
                measured = support_summary(
                    atlas_labels == region, atlas_support, atlas_coords[..., 0], atlas_unknown
                )
                whole = measured["whole"]
                if (
                    whole["total"] != reference.atlas_voxels
                    or whole["included"] != reference.atlas_mask_voxels
                    or whole["unknown"] != reference.atlas_mask_unknown_voxels
                ):
                    raise ValueError(
                        f"Mask denominator/count differs from sealed audit: {pid}/{region}"
                    )
                if region in (14, 37):
                    np.testing.assert_allclose(
                        source_series[source][str(region)],
                        comparison_raw[f"archive_spm8_2mm_roi_{region}"],
                        rtol=1e-10,
                        atol=1e-8,
                    )
            signal = source_series[source][str(region)]
            source_stats[source][str(region)] = dict(
                support=measured,
                grid_voxels=n_grid,
                grid_varying_voxels=n_vary,
                constant=bool(np.all(np.asarray(signal) == signal[0]))
                if signal is not None
                else None,
            )
            focus[source][str(region)] = region_focus(labels, coords, region)
        source_directory = directory / source
        source_directory.mkdir(exist_ok=True)
        for filename, array in [
            ("mean.nii.gz", mean),
            ("atlas.nii.gz", labels),
            ("support.nii.gz", support.astype(np.uint8)),
            ("unknown.nii.gz", unknown.astype(np.uint8)),
        ]:
            write_volume(source_directory / filename, array, image)
        foreground = mean[varying]
        window = (
            [0.0, float(np.percentile(foreground, 99))]
            if len(foreground)
            else [float(mean.min()), float(mean.max())]
        )
        if window[1] <= window[0]:
            raise ValueError("Display image has no usable intensity range")
        metadata["sources"][source] = dict(
            title="Tutorial fMRIPrep" if source == "tutorial" else "Independent SPM8 archive",
            grid_mm=[float(x) for x in image.header.get_zooms()[:3]],
            image_shape=list(image.shape[:3]),
            source_filename=path.name,
            source_sha256=sha256(path),
            mask_source_filename=None
            if source == "tutorial"
            else mask_image.get_filename().split("/")[-1],
            mask_source_sha256=None if source == "tutorial" else sha256(mask_image.get_filename()),
            window=window,
            preprocessing="fMRIPrep 1.3.0.post2, mask, 4-mm resampling, int8 distribution"
            if source == "tutorial"
            else "SPM8 / MATLAB, 5-mm smoothing, normalized 2-mm grid",
            support_kind="temporal_variation" if source == "tutorial" else "brain_mask",
            support_label="Varying voxel centers"
            if source == "tutorial"
            else "Archived mask overlap",
            denominator="Parcel centers on the 4-mm image grid"
            if source == "tutorial"
            else "All native 2-mm atlas parcel centers",
            note="Matching fMRIPrep mask unavailable. Temporal variation is a proxy affected by masking, interpolation and int8 quantization."
            if source == "tutorial"
            else "This is an independent preprocessing lineage, not the matching fMRIPrep parent. A 2-mm normalized grid is not native 2-mm acquisition resolution.",
            unknown_voxels=int(unknown.sum()),
            volumes={
                name: f"{source}/{name}.nii.gz" for name in ("mean", "atlas", "support", "unknown")
            },
        )
        del values
    for region in sorted(names):
        metadata["regions"].append(
            dict(
                id=region,
                name=names[region],
                focus={s: focus[s][str(region)] for s in SOURCES},
                **{
                    s: {**source_stats[s][str(region)], "series": source_series[s][str(region)]}
                    for s in SOURCES
                },
            )
        )
    # Independently recorded hemisphere totals already exist for the two investigation regions.
    hemis = pd.read_csv(
        root / "investigations/eight-adult-coverage/overnight-followups/hemisphere_coverage.csv"
    )
    for r in metadata["regions"]:
        if r["id"] not in (14, 37):
            continue
        for source, evidence, field in [
            ("tutorial", "tutorial_grid_4mm", "varying_voxels"),
            ("archive", "spm8_full_atlas_2mm", "mask_voxels"),
        ]:
            original = hemis.loc[
                (hemis.participant_id == pid)
                & (hemis.region_id == r["id"])
                & (hemis.evidence == evidence)
            ].set_index("hemisphere")
            for side in ("left", "right", "midline"):
                measured = r[source]["support"][side]
                if (
                    measured["total"] != original.loc[side, "voxels"]
                    or measured["included"] != original.loc[side, field]
                ):
                    raise ValueError("Display hemisphere counts disagree with sealed follow-up")
    save_json(directory / "metadata.json", metadata)


def valid_marker(directory, key=None):
    marker_path = directory / "completed.json"
    if not marker_path.is_file():
        return False
    marker = json.loads(marker_path.read_text())
    return (key is None or marker.get("key") == key) and all(
        (directory / p).is_file() and sha256(directory / p) == h
        for p, h in marker["artifacts"].items()
    )


def verify_saved_artifacts(directory):
    marker = json.loads((directory / "completed.json").read_text())
    for relative, expected in marker["artifacts"].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Saved scientific artifact changed: {path}")


def prepare_explorer(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    protected = [
        root / p
        for p in ("data/source", "outputs/one", "outputs/eight", "investigations", ".cache")
    ]
    if any(destination.is_relative_to(p) or p.is_relative_to(destination) for p in protected):
        raise ValueError(
            "Display output must be isolated from sources, primary results and investigations"
        )
    destination.mkdir(parents=True, exist_ok=True)
    for original in (
        root / "outputs/eight",
        root / "investigations/eight-adult-coverage",
        root / "investigations/eight-adult-coverage/overnight-followups",
    ):
        verify_saved_artifacts(original)
    manifest = pd.read_csv(root / "outputs/eight/manifest.csv")
    if (
        manifest.participant_id.duplicated().any()
        or manifest.scan_key.duplicated().any()
        or not manifest.data_kind.eq("real").all()
        or set(manifest.participant_id) != {f"sub-pixar{i}" for i in range(123, 131)}
    ):
        raise ValueError("Invalid original participant/scan identifiers")
    audit = root / "investigations/eight-adult-coverage"
    assets = source_index(json.loads((audit / "download_manifest.json").read_text()))
    expected = (
        pd.read_csv(audit / "region_coverage.csv")
        .set_index(["participant_id", "lineage", "region_id"])
        .sort_index()
    )
    p = json.loads((root / "outputs/eight/provenance.json").read_text())
    atlas_path = Path(p["atlas"]["source_path"])
    if sha256(atlas_path) != p["atlas"]["sha256"]:
        raise ValueError("Atlas bytes differ from saved provenance")
    atlas = nib.load(atlas_path)
    names = {int(r["index"]): r["name"] for r in p["atlas"]["labels"] if r["index"] != 0}
    retention = pd.read_csv(root / "outputs/eight/retention.csv")
    versions = {
        name: importlib.metadata.version(name) for name in ("numpy", "nibabel", "nilearn", "pandas")
    }
    shared = {
        "atlas": sha256(atlas_path),
        "sealed_counts": sha256(audit / "region_coverage.csv"),
        "sealed_hemisphere": sha256(audit / "overnight-followups/hemisphere_coverage.csv"),
        "retention": sha256(root / "outputs/eight/retention.csv"),
        "primary_manifest": sha256(root / "outputs/eight/manifest.csv"),
        "primary_provenance": sha256(root / "outputs/eight/provenance.json"),
        "source_inventory": sha256(audit / "download_manifest.json"),
        "code": sha256(__file__),
        "versions": versions,
        "settings": SETTINGS,
    }
    completed, hits, failures = [], [], []
    for record in manifest.to_dict("records"):
        pid = record["participant_id"]
        directory = destination / "scans" / pid
        directory.parent.mkdir(exist_ok=True)
        try:
            input_files = [(Path(record["image_path"]), record["image_sha256"])]
            for role in ("spm8_bold", "spm8_mask"):
                a = assets[pid, role]
                if a["status"] != "complete":
                    raise ValueError(f"Source asset unavailable: {pid}/{role}")
                input_files.append((root / a["path"], a["sha256"]))
            for path, h in input_files:
                if sha256(path) != h:
                    raise ValueError(f"Source bytes changed: {path}")
            for filename in ("frames.csv", "raw_timeseries.csv"):
                path = root / "outputs/eight/scans" / record["scan_key"] / filename
                input_files.append((path, sha256(path)))
            reference_series = audit / "scans" / pid / "source_roi_timeseries.csv"
            input_files.append((reference_series, sha256(reference_series)))
            key = hashlib.sha256(
                json.dumps(
                    {**shared, "inputs": [h for _, h in input_files]}, sort_keys=True
                ).encode()
            ).hexdigest()
            if directory.is_dir() and valid_marker(directory, key):
                hits.append(pid)
                print(f"{pid}: verified display cache reused", flush=True)
            else:
                temporary = directory.with_name(pid + ".preparing")
                if temporary.exists():
                    shutil.rmtree(temporary)
                temporary.mkdir()
                prepare_scan(root, temporary, record, assets, atlas, names, expected, retention)
                files = {
                    str(f.relative_to(temporary)): sha256(f)
                    for f in temporary.rglob("*")
                    if f.is_file()
                }
                save_json(
                    temporary / "completed.json",
                    dict(key=key, artifacts=files, data_kind="real_display_assets"),
                )
                if directory.exists():
                    history = destination / "history"
                    history.mkdir(exist_ok=True)
                    directory.replace(
                        history / f"{pid}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
                    )
                temporary.replace(directory)
                print(f"{pid}: display assets prepared and reconciled", flush=True)
            completed.append(pid)
        except (ValueError, OSError, KeyError, AssertionError) as exc:
            failures.append(dict(participant_id=pid, error=str(exc)))
            print(f"{pid}: display unavailable: {exc}", flush=True)
    save_json(destination / "failures.json", failures)
    save_json(
        destination / "provenance.json",
        dict(
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            inputs=shared,
            atlas=p["atlas"],
            cache_hits=hits,
            primary_results_changed=False,
        ),
    )
    artifacts = {
        str(f.relative_to(destination)): sha256(f)
        for pid in completed
        for f in (destination / "scans" / pid).rglob("*")
        if f.is_file()
    }
    artifacts.update(
        {name: sha256(destination / name) for name in ("failures.json", "provenance.json")}
    )
    save_json(
        destination / "completed.json",
        dict(
            schema_version=VERSION,
            data_kind="real_display_assets",
            status="complete" if not failures else "partial",
            participant_ids=completed,
            artifacts=artifacts,
        ),
    )
    return dict(prepared=len(completed), cache_hits=hits, failures=failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = prepare_explorer(
        args.root, args.output_dir or args.root / "data/derived/coverage_explorer"
    )
    print(json.dumps(result, indent=2))
    return bool(result["failures"])


if __name__ == "__main__":
    raise SystemExit(main())
