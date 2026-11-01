"""Supplementary coverage audit. Reads immutable inputs; writes only beside this script."""

import argparse
import base64
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from metrics import join_assets, nearest, reference_profile, region_profile
from nilearn.image import resample_to_img

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PAPER = "https://pmc.ncbi.nlm.nih.gov/articles/PMC10635736/"
ROLES = ("native_bold", "native_t1", "spm8_bold", "spm8_mask", "spm8_anat", "mriqc")
LINEAGES = ("tutorial_4mm", "archive_spm8_2mm")
SETTINGS = {
    "atlas": "Harvard-Oxford cort-maxprob-thr25-2mm; existing unsplit 48-region atlas",
    "data_kind": "real",
    "analysis_type": "supplementary coverage audit; no connectivity computation",
    "atlas_reference_denominator": "All native 2-mm atlas voxels per parcel, including out-of-grid locations",
    "grid_denominator": "Atlas parcel voxel centers after nearest-neighbor placement on each image grid",
    "variation": "At least one stored voxel value differs exactly from its first-frame value across all acquired frames",
    "primary_support_caveat": "Temporal variation is a proxy affected by int8 quantization and interpolation; it is not a brain mask or proof of biological signal",
    "spm8_mask_caveat": "Independent SPM8 analysis mask, not the matching 2019 fMRIPrep mask or a validated functional-coverage standard",
    "limited_mask_review_threshold": 0.8,
    "sparse_primary_variation_review_threshold": 0.5,
    "threshold_interpretation": "Descriptive project review bins, not validated acceptability or exclusion criteria",
    "frames": "All acquired frames, original indices retained; no censoring, filtering, regression, standardization or smoothing added",
    "resampling": "Nearest-neighbor for labels, binary masks, rectangular image domain and variation support; fill outside grid with 0",
    "undefined_mask_policy": "NaN mask voxels stay unknown. Report known included counts, unknown counts and full-denominator lower/upper bounds. A point fraction is unavailable if a parcel contains any unknown mask locations. No imputation or source editing.",
    "reference_space_caveat": "Atlas placement uses supplied world coordinates; archive MNI normalization differs from tutorial MNI152NLin2009cAsym and atlas registration is not re-estimated",
    "primary_results_policy": "Never overwrite primary files or introduce new primary exclusions",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def archive_scan_attempt(destination, history):
    """Move a failed attempt intact out of current scan artifacts, then start a clean failure record."""
    destination, history = Path(destination), Path(history)
    archived = None
    if destination.exists() and any(destination.iterdir()):
        history.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archived = history / f"{destination.name}-{stamp}"
        destination.replace(archived)
    destination.mkdir(parents=True, exist_ok=True)
    return archived


def check_protected():
    before = json.loads((OUT / "protected-before.json").read_text())
    modified = [p for p, h in before.items() if (ROOT / p).is_file() and digest(ROOT / p) != h]
    removed = [p for p in before if not (ROOT / p).is_file()]
    # New application modules are permitted; new observations in the protected data/results are not.
    protected_trees = (
        "data/source",
        "outputs/one",
        "outputs/eight",
        ".cache",
        "tests/fixtures",
        "investigations/sub-pixar124-region14",
        "investigations/sub-pixar124-upstream",
    )
    current = {
        str(p.relative_to(ROOT))
        for folder in protected_trees
        for p in (ROOT / folder).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    added = sorted(current - before.keys())
    result = dict(
        protected_file_count=len(before),
        modified=modified,
        removed=removed,
        added=added,
        unchanged=not (modified or removed or added),
    )
    write_json(OUT / "preservation.json", result)
    if not result["unchanged"]:
        raise ValueError(f"Protected inputs or primary results changed: {result}")
    return result


def labels_on(atlas, image):
    return np.asarray(
        resample_to_img(
            atlas, image, interpolation="nearest", force_resample=True, copy_header=True
        ).dataobj
    )


def unscaled(image):
    data = image.dataobj.get_unscaled()
    if (
        not np.isfinite(data).all()
        or not np.isfinite([image.dataobj.slope, image.dataobj.inter]).all()
        or image.dataobj.slope == 0
    ):
        raise ValueError("Image has non-finite observations or invalid scaling")
    return data


def scan_cache_key(record, assets, atlas_path):
    return hashlib.sha256(
        json.dumps(
            {
                "primary_sha256": record["image_sha256"],
                "assets": {role: assets[role]["sha256"] for role in ROLES},
                "atlas_sha256": digest(atlas_path),
                "settings": SETTINGS,
                "audit_code": digest(__file__),
                "metrics_code": digest(OUT / "metrics.py"),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def inspect_scan(record, assets, atlas, region_ids, names, destination):
    participant = record["participant_id"]
    primary = nib.load(record["image_path"])
    paths = {role: ROOT / assets[role]["path"] for role in ROLES}
    bold, mask_image = nib.load(paths["spm8_bold"]), nib.load(paths["spm8_mask"])
    if mask_image.shape != bold.shape[:3] or not np.allclose(
        mask_image.affine, bold.affine, rtol=0, atol=1e-6
    ):
        raise ValueError("SPM8 BOLD/mask dimensions or affine disagree")
    mask = np.asarray(mask_image.dataobj)
    if primary.shape[3] != record["n_volumes"] or bold.shape[3] != primary.shape[3]:
        raise ValueError("Acquired-frame counts disagree across primary/archived BOLD")
    all_rows, means_for_plot, regional_means = [], {}, {}
    scaled_max_error = None
    for lineage, image, supplied in [(LINEAGES[0], primary, None), (LINEAGES[1], bold, mask)]:
        values = unscaled(image)
        label_grid = labels_on(atlas, image)
        grid_rows, varying, signals = region_profile(values, label_grid, region_ids, supplied)
        ref_rows = reference_profile(atlas, image, varying, region_ids, supplied)
        for grid_row, ref_row in zip(grid_rows, ref_rows, strict=True):
            if grid_row["region_id"] != ref_row["region_id"]:
                raise ValueError("Region identity join failed")
            all_rows.append(
                dict(
                    participant_id=participant,
                    scan_key=record["scan_key"],
                    lineage=lineage,
                    region_name=names[grid_row["region_id"]],
                    **grid_row,
                    **{k: v for k, v in ref_row.items() if k != "region_id"},
                )
            )
        decoded = np.stack(signals, axis=1) * image.dataobj.slope + image.dataobj.inter
        regional_means[lineage] = decoded
        means_for_plot[lineage] = (
            values.mean(axis=3, dtype=np.float64) * image.dataobj.slope + image.dataobj.inter
        )
        if lineage == LINEAGES[0]:
            previous = pd.read_csv(
                ROOT / "outputs/eight/scans" / record["scan_key"] / "raw_timeseries.csv"
            )
            expected = previous[[f"roi_{i}" for i in region_ids]].to_numpy()
            if expected.shape != decoded.shape or not np.array_equal(
                previous.original_index, np.arange(primary.shape[3])
            ):
                raise ValueError("Saved primary regional identities or original indices disagree")
            scaled_max_error = float(np.max(np.abs(decoded - expected)))
            np.testing.assert_allclose(decoded, expected, rtol=1e-10, atol=1e-8)
        del values
    frame_table = pd.DataFrame(
        {
            "original_index": np.arange(primary.shape[3]),
            "time_seconds": np.arange(primary.shape[3]) * record["sampling_interval_seconds"],
        }
    )
    # Region-14 and region-37 source observations only, not re-cleaned time series or new connectivity.
    for lineage in LINEAGES:
        for region in (14, 37):
            frame_table[f"{lineage}_roi_{region}"] = regional_means[lineage][
                :, region_ids.index(region)
            ]
    frame_table.to_csv(destination / "source_roi_timeseries.csv", index=False)
    native, t1 = nib.load(paths["native_bold"]), nib.load(paths["native_t1"])
    if native.shape[3] != primary.shape[3] or not np.isclose(
        native.header.get_zooms()[3], record["sampling_interval_seconds"]
    ):
        raise ValueError(
            "Native BOLD frame count or known TR disagrees with the primary acquisition metadata"
        )
    unscaled(
        native
    )  # Full read/finite check; do not infer registered atlas coverage from native headers.
    unscaled(t1)
    t1_values = np.asarray(t1.dataobj)
    html = paths["mriqc"].read_text()
    if participant.removeprefix("sub-") + "_pixar" not in html:
        raise ValueError("MRIQC report identity does not match participant")
    version = re.search(r"MRIQC version:\s*([^<\s]+)", html)
    mask_primary = nearest(mask == 1, bold, primary)
    primary_labels = labels_on(atlas, primary)
    sample_overlap = {
        region: int(((primary_labels == region) & (mask_primary > 0)).sum()) for region in (14, 37)
    }
    scan_info = dict(
        participant_id=participant,
        scan_key=record["scan_key"],
        data_kind="real",
        primary_status=record["processing_status"],
        primary_raw_means_max_absolute_error=scaled_max_error,
        primary_shape=list(primary.shape),
        primary_dtype=str(primary.get_data_dtype()),
        spm8_shape=list(bold.shape),
        spm8_dtype=str(bold.get_data_dtype()),
        spm8_header_tr=float(bold.header.get_zooms()[3]),
        native_shape=list(native.shape),
        native_voxel_sizes_mm=[float(x) for x in native.header.get_zooms()[:3]],
        native_tr_seconds=float(native.header.get_zooms()[3]),
        nominal_native_slice_slab_mm=float(native.shape[2] * native.header.get_zooms()[2]),
        mriqc_version=version.group(1) if version else "unknown",
        bold_and_t1_full_readability_and_finite=True,
        spm8_mask_nonfinite_voxels=int(np.isnan(mask).sum()),
        primary_region14_centers_in_spm8_mask=sample_overlap[14],
        primary_region37_centers_in_spm8_mask=sample_overlap[37],
        published_poor_coverage_exclusion=participant in {"sub-pixar123", "sub-pixar124"},
    )
    missing_indices = np.argwhere(np.isnan(mask))
    missing_locations = pd.DataFrame(missing_indices, columns=["voxel_i", "voxel_j", "voxel_k"])
    if len(missing_indices):
        missing_locations[["world_x_mm", "world_y_mm", "world_z_mm"]] = nib.affines.apply_affine(
            bold.affine, missing_indices
        )
    missing_locations.to_csv(destination / "mask_missing_locations.csv", index=False)
    # Side-by-side archive slices use the same supplied world coordinates, with no new registration.
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    for column, z_world in enumerate((-38, -22, 2)):
        for row, (lineage, image) in enumerate(zip(LINEAGES, (primary, bold), strict=True)):
            mean_image = nib.Nifti1Image(means_for_plot[lineage], image.affine)
            reference = np.asarray(
                resample_to_img(
                    mean_image, atlas, interpolation="linear", force_resample=True, copy_header=True
                ).dataobj
            )
            z = int(
                round(nib.affines.apply_affine(np.linalg.inv(atlas.affine), [0, 0, z_world])[2])
            )
            x = np.arange(atlas.shape[0]) * atlas.affine[0, 0] + atlas.affine[0, 3]
            y = np.arange(atlas.shape[1]) * atlas.affine[1, 1] + atlas.affine[1, 3]
            axis = axes[row, column]
            axis.imshow(
                reference[:, :, z].T,
                origin="lower",
                extent=(x[0] + 1, x[-1] - 1, y[0] - 1, y[-1] + 1),
                cmap="gray",
                vmin=0,
                vmax=np.percentile(reference[reference > 0], 99),
            )
            for region, color in ((14, "#f1a13a"), (37, "#ce81dc")):
                roi = (np.asarray(atlas.dataobj) == region)[:, :, z].T
                if roi.any():
                    axis.contour(x, y, roi, levels=[0.5], colors=[color], linewidths=1)
            if row == 1:
                reference_mask = nearest(mask == 1, bold, atlas)[:, :, z].T
                if reference_mask.any():
                    axis.contour(
                        x, y, reference_mask, levels=[0.5], colors=["#41c9cb"], linewidths=0.7
                    )
            axis.set(
                title=f"{'Tutorial 4-mm' if row == 0 else 'SPM8 2-mm'} · z={z_world} mm",
                xlabel="MNI x (mm)",
                ylabel="MNI y (mm)",
            )
    figure.suptitle(
        f"{participant}: source coverage review\nOrange: region 14; purple: region 37; cyan: independent SPM8 mask. Relative image intensity per panel.",
        fontsize=12,
    )
    figure.savefig(destination / "coverage.png", dpi=130)
    plt.close(figure)
    if participant in {"sub-pixar123", "sub-pixar124"}:
        native_fov = nearest(np.ones(native.shape[:3], dtype=np.uint8), native, t1)
        center = int(round(nib.affines.apply_affine(np.linalg.inv(t1.affine), [0, 0, 0])[0]))
        figure, axis = plt.subplots(figsize=(7, 6), layout="constrained")
        axis.imshow(
            t1_values[center].T,
            origin="lower",
            cmap="gray",
            vmin=0,
            vmax=np.percentile(t1_values, 99),
        )
        axis.contour(native_fov[center].T, levels=[0.5], colors=["#f1a13a"], linewidths=1.3)
        axis.set(
            title=f"{participant}: acquired BOLD extent on native T1\nScanner header coordinates only; no estimated registration",
            xlabel="T1 voxel j",
            ylabel="T1 voxel k",
        )
        figure.savefig(destination / "native_fov.png", dpi=130)
        plt.close(figure)
    return pd.DataFrame(all_rows), scan_info


def issues_from_regions(regions):
    issues = []
    for row in regions.to_dict("records"):
        rule = None
        if row["lineage"] == LINEAGES[1] and row["atlas_mask_fraction"] < 0.8:
            rule = (
                "no_archived_mask_support"
                if row["atlas_mask_voxels"] == 0
                else "limited_archived_mask_support"
            )
            evidence = f"{int(row['atlas_mask_voxels'])}/{int(row['atlas_voxels'])} full-atlas voxels in independent SPM8 mask ({row['atlas_mask_fraction']:.4f}); 0.8 is a descriptive review bin"
            explanations = "Limited acquired coverage, mask construction, normalization/alignment differences or template/atlas placement; exact 2019 fMRIPrep contribution is unresolved"
            location = (
                f"{row['participant_id']}_analysis_mask.nii.gz; atlas region {row['region_id']}"
            )
        elif row["lineage"] == LINEAGES[0] and row["grid_varying_fraction"] < 0.5:
            rule = "sparse_tutorial_temporal_support"
            evidence = f"{int(row['grid_varying_voxels'])}/{int(row['grid_voxels'])} tutorial parcel voxel centers vary over the full acquired run ({row['grid_varying_fraction']:.4f}); 0.5 is a descriptive review bin"
            explanations = "Limited coverage or masking, int8 quantization, resampling, or low voxel variability; variation alone cannot identify neural signal or the root cause"
            location = f"{row['participant_id']}_task-pixar_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz; atlas region {row['region_id']}"
        if rule:
            issues.append(
                dict(
                    participant_id=row["participant_id"],
                    region_id=row["region_id"],
                    region_name=row["region_name"],
                    severity="review",
                    rule=rule,
                    observed_evidence=evidence,
                    file_location=location,
                    possible_explanations=explanations,
                    recommended_action="Inspect the spatial evidence and obtain matching fMRIPrep parent BOLD/mask/report before deciding whether the parcel or scan is suitable. Preserve primary results; do not automatically exclude or repair observations.",
                )
            )
    return pd.DataFrame(
        issues,
        columns=[
            "participant_id",
            "region_id",
            "region_name",
            "severity",
            "rule",
            "observed_evidence",
            "file_location",
            "possible_explanations",
            "recommended_action",
        ],
    )


def summaries(regions, scans):
    rows = []
    for scan in scans:
        subset = regions.loc[regions.participant_id == scan["participant_id"]]
        primary = subset.loc[subset.lineage == LINEAGES[0]]
        archive = subset.loc[subset.lineage == LINEAGES[1]]
        row = dict(scan)
        row.update(
            cortical_atlas_voxels=int(archive.atlas_voxels.sum()),
            archived_mask_atlas_voxels=int(archive.atlas_mask_voxels.sum()),
            archived_mask_unknown_atlas_voxels=int(archive.atlas_mask_unknown_voxels.sum()),
            archived_mask_cortical_atlas_fraction=(
                float(archive.atlas_mask_voxels.sum() / archive.atlas_voxels.sum())
                if not archive.atlas_mask_unknown_voxels.sum()
                else None
            ),
            archived_mask_cortical_atlas_fraction_lower=float(
                archive.atlas_mask_voxels.sum() / archive.atlas_voxels.sum()
            ),
            archived_mask_cortical_atlas_fraction_upper=float(
                (archive.atlas_mask_voxels.sum() + archive.atlas_mask_unknown_voxels.sum())
                / archive.atlas_voxels.sum()
            ),
            archived_zero_mask_regions=int(
                (archive.atlas_mask_voxels.eq(0) & archive.atlas_mask_unknown_voxels.eq(0)).sum()
            ),
            archived_regions_below_80pct=int(archive.atlas_mask_fraction.lt(0.8).sum()),
            tutorial_sparse_regions_below_50pct=int(primary.grid_varying_fraction.lt(0.5).sum()),
            tutorial_constant_regional_means=int(primary.regional_mean_constant.sum()),
        )
        for region in (14, 37):
            p = primary.loc[primary.region_id == region].iloc[0]
            a = archive.loc[archive.region_id == region].iloc[0]
            for name in (
                "grid_voxels",
                "grid_varying_voxels",
                "grid_varying_fraction",
                "regional_mean_constant",
            ):
                row[f"tutorial_r{region}_{name}"] = p[name]
            for name in (
                "atlas_voxels",
                "atlas_mask_voxels",
                "atlas_mask_fraction",
                "atlas_varying_voxels",
            ):
                row[f"spm8_r{region}_{name}"] = (
                    float(a[name]) if name == "atlas_mask_fraction" else int(a[name])
                )
        rows.append(row)
    return pd.DataFrame(rows)


def render_report(summary, regions, issues, failures, preservation, provenance):
    figure, axes = plt.subplots(2, 1, figsize=(13, 9), layout="constrained", sharex=True)
    ids = list(range(1, 49))
    participants = summary.participant_id.tolist()
    images = []
    for axis, lineage, field, title in [
        (
            axes[0],
            LINEAGES[1],
            "atlas_mask_fraction",
            "Independent SPM8 mask / full 2-mm atlas parcel (includes out-of-grid atlas voxels)",
        ),
        (
            axes[1],
            LINEAGES[0],
            "grid_varying_fraction",
            "Tutorial varying voxel centers / resampled 4-mm parcel (variation proxy, not brain mask)",
        ),
    ]:
        table = (
            regions.loc[regions.lineage == lineage]
            .pivot(index="participant_id", columns="region_id", values=field)
            .reindex(index=participants, columns=ids)
        )
        images.append(
            axis.imshow(
                table.to_numpy(),
                aspect="auto",
                interpolation="nearest",
                cmap="viridis",
                vmin=0,
                vmax=1,
            )
        )
        axis.set(yticks=np.arange(len(participants)), yticklabels=participants, title=title)
        axis.set_xticks(np.arange(48), labels=ids, fontsize=7)
        for region in (14, 37):
            axis.axvline(region - 1, color="#e38c32", linewidth=0.6, alpha=0.8)
    axes[1].set_xlabel("Fixed Harvard–Oxford region ID (1–48); orange marks 14 and 37")
    figure.colorbar(images[0], ax=axes, label="Fraction, fixed scale 0–1", fraction=0.025)
    figure.suptitle(
        "Eight-adult coverage supplement · two independent evidence layers", fontsize=14
    )
    figure.savefig(OUT / "coverage_heatmap.png", dpi=140)
    plt.close(figure)
    columns = [
        "participant_id",
        "primary_status",
        "archived_mask_cortical_atlas_fraction",
        "archived_zero_mask_regions",
        "archived_regions_below_80pct",
        "tutorial_sparse_regions_below_50pct",
        "tutorial_constant_regional_means",
    ]
    focus = summary.loc[summary.participant_id.isin(["sub-pixar123", "sub-pixar124"])].to_dict(
        "records"
    )
    method_text = "\n".join(f"- **{key}**: {value}" for key, value in SETTINGS.items())
    text = "# Eight-adult brain-coverage audit\n\nSupplementary findings only. Primary source files and results are unchanged.\n\n"
    text += "## Question and sample\n\nDo numerical results hide absent or sparse atlas-region support? The original eight adults, sub-pixar123 through sub-pixar130, are retained; no participant is replaced. Two lineages are evaluated separately: the existing 4-mm fMRIPrep tutorial derivative and an independently preprocessed 2-mm SPM8 archive.\n\n"
    text += "## Actual scan summaries\n\n" + summary[columns].to_csv(index=False) + "\n"
    for row in focus:
        text += f"### {row['participant_id']}\n\nPrimary status remains `{row['primary_status']}`. Region 14: {row['tutorial_r14_grid_varying_voxels']}/{row['tutorial_r14_grid_voxels']} tutorial voxel centers vary; {row['spm8_r14_atlas_mask_voxels']}/{row['spm8_r14_atlas_voxels']} full-atlas voxels overlap the independent SPM8 mask. Region 37: {row['tutorial_r37_grid_varying_voxels']}/{row['tutorial_r37_grid_voxels']} tutorial voxel centers vary; {row['spm8_r37_atlas_mask_voxels']}/{row['spm8_r37_atlas_voxels']} full-atlas voxels overlap that mask.\n\n"
    for row in summary.loc[summary.spm8_mask_nonfinite_voxels > 0].to_dict("records"):
        text += f"### {row['participant_id']}: undefined source mask values\n\nThe archived mask contains {row['spm8_mask_nonfinite_voxels']} NaN voxels; {row['archived_mask_unknown_atlas_voxels']} full-atlas cortical locations overlap them. Unknown values were preserved, not filled. Original voxel/world coordinates are exported in the participant's mask_missing_locations.csv.\n\n"
    text += "## Interpretation and limits\n\nA nonconstant regional average or computable correlation matrix does not establish adequate spatial coverage. The 4-mm temporal-variation map is influenced by masking, interpolation and int8 quantization; it is not a brain mask. The SPM8 mask is independent corroboration, not the exact 2019 fMRIPrep mask. The two mask/variation percentages have different meanings and denominators and must not be pooled or subtracted as a pipeline effect. The archive is on a 2-mm normalized grid after 5-mm smoothing, not native 2-mm acquisition resolution. No registration is re-estimated. Template and normalization differences limit precise parcel correspondence.\n\n"
    text += f"A [published reanalysis]({PAPER}) excluded both sub-pixar123 and sub-pixar124 for poor brain coverage. It used SPM12, a third lineage; its exclusion is context, not an imported primary-analysis rule. The exact fMRIPrep parent images, masks and preprocessing HTML QA are not available in the materials used here. MRIQC 0.9.6 supplies native acquisition QA and must not be described as that missing preprocessing report. Native FOV illustrations for 123/124 use scanner header coordinates only, with no estimated registration.\n\n"
    text += "## Methods and denominators\n\n" + method_text + "\n\n"
    text += "## Checks and failures\n\nParticipant/role joins, full source hashes, binary mask values, image/mask affine and dimensions, full BOLD readability/finite values, frame counts, original indices, all region identities, direct-source mean agreement with saved raw regional series, full-atlas and within-grid denominators, report/CSV reconciliation and protected-file preservation are checked.\n\n"
    text += f"Completed scan audits: {len(summary)}/8. Acquisition or scan failures: {len(failures)}. Protected files unchanged: {preservation['unchanged']} ({preservation['protected_file_count']} files). See issues.csv for observed evidence, file/region location, review rule, possible explanations and recommended actions. No issue automatically changes a primary status.\n\n"
    text += "## Next steps\n\nInspect the regional slice figures with the investigator. Obtain the matching 2019 fMRIPrep BOLD, mask, registration panels and acquisition metadata. Agree on a prespecified coverage policy before a future separately versioned analysis; do not retrospectively claim that a numerical matrix validates biological accuracy. The unsent source request and portfolio case study are separate documents.\n\n"
    text += "## Source attribution and reproduction\n\nOriginal dataset: Richardson et al. (2018), ds000228 R1.0.0. Archive methods: `https://s3.amazonaws.com/openneuro/ds000228/ds000228_R1.0.0/uncompressed/derivatives/README` (SPM8/MATLAB despite its historical folder name). Tutorial methods: `https://osf.io/wjtyq/` (fMRIPrep 1.3.0.post2, masking, 4-mm downsampling, int8). Source file URLs, archive timestamps and SHA-256/MD5 values are recorded in download_manifest.json; atlas and dependency metadata are in provenance.json.\n\n```sh\nuv run python investigations/eight-adult-coverage/acquire.py\nuv run python investigations/eight-adult-coverage/audit.py\nuv run python investigations/eight-adult-coverage/audit.py --verify-only\n```\n\nDownloads are sequential and skip verified existing assets. Completed per-scan outputs are keyed by input, atlas, settings and code hashes. Reruns verify artifacts before reuse; no primary cache is written.\n"
    (OUT / "REPORT.md").write_text(text)
    environment = Environment(
        loader=FileSystemLoader(OUT), autoescape=select_autoescape(["html", "j2"])
    )
    template = environment.get_template("report.html.j2")

    def embedded(path):
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()

    html = template.render(
        summary=summary,
        summary_rows=summary.to_dict("records"),
        focus=focus,
        settings=SETTINGS,
        issues_count=len(issues),
        failures=failures,
        preservation=preservation,
        paper=PAPER,
        generated_at=provenance["generated_at_utc"],
        heatmap=embedded(OUT / "coverage_heatmap.png"),
        scan_figures=[
            (r["participant_id"], embedded(OUT / "scans" / r["participant_id"] / "coverage.png"))
            for r in focus
        ],
    )
    (OUT / "report.html").write_text(html)


def verify():
    marker = json.loads((OUT / "completed.json").read_text())
    for relative, expected in marker["artifacts"].items():
        path = (OUT / relative).resolve()
        if not path.is_relative_to(OUT) or digest(path) != expected:
            raise ValueError(f"Coverage supplement artifact changed: {relative}")
    for asset in json.loads((OUT / "download_manifest.json").read_text()):
        if asset["status"] == "complete" and digest(ROOT / asset["path"]) != asset["sha256"]:
            raise ValueError(f"Supplementary source bytes changed: {asset['path']}")
    summary = pd.read_csv(OUT / "participant_summary.csv")
    regions = pd.read_csv(OUT / "region_coverage.csv")
    if regions.duplicated(["participant_id", "lineage", "region_id"]).any():
        raise ValueError("Duplicate coverage scan/lineage/region keys")
    for row in regions.to_dict("records"):
        for count, total, fraction in [
            ("grid_varying_voxels", "grid_voxels", "grid_varying_fraction"),
            ("atlas_varying_voxels", "atlas_voxels", "atlas_varying_fraction"),
            ("atlas_in_grid_voxels", "atlas_voxels", "atlas_in_grid_fraction"),
        ]:
            if row[total] and (
                not 0 <= row[count] <= row[total]
                or not np.isclose(row[count] / row[total], row[fraction])
            ):
                raise ValueError(f"Coverage denominator mismatch: {row}")
        if row["lineage"] == LINEAGES[1]:
            lower = row["atlas_mask_voxels"] / row["atlas_voxels"]
            upper = (row["atlas_mask_voxels"] + row["atlas_mask_unknown_voxels"]) / row[
                "atlas_voxels"
            ]
            if not (
                np.isclose(lower, row["atlas_mask_fraction_lower"])
                and np.isclose(upper, row["atlas_mask_fraction_upper"])
            ):
                raise ValueError("Mask denominator or unknown-value bounds mismatch")
            if row["atlas_mask_unknown_voxels"]:
                if pd.notna(row["atlas_mask_fraction"]):
                    raise ValueError("Unknown mask locations were silently imputed")
            elif not np.isclose(lower, row["atlas_mask_fraction"]):
                raise ValueError("Mask point estimate mismatch")
    report = (OUT / "report.html").read_text()
    for row in summary.to_dict("records"):
        if row["participant_id"] not in report:
            raise ValueError("Report omits a completed participant")
        for region in (14, 37):
            p = regions.loc[
                (regions.participant_id == row["participant_id"])
                & (regions.lineage == LINEAGES[0])
                & (regions.region_id == region)
            ].iloc[0]
            if p.grid_varying_voxels != row[f"tutorial_r{region}_grid_varying_voxels"]:
                raise ValueError("Summary/region export mismatch")
    protection = check_protected()
    print(
        json.dumps(
            dict(
                status=marker["status"],
                scans=len(summary),
                region_rows=len(regions),
                protected=protection,
            ),
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify()
        return
    check_protected()
    manifest = pd.read_csv(ROOT / "outputs/eight/manifest.csv")
    if (
        manifest.participant_id.duplicated().any()
        or manifest.scan_key.duplicated().any()
        or not manifest.data_kind.eq("real").all()
        or set(manifest.participant_id) != {f"sub-pixar{i}" for i in range(123, 131)}
    ):
        raise ValueError("Primary manifest identifiers or real-data selection are invalid")
    assets = json.loads((OUT / "download_manifest.json").read_text())
    joined = join_assets(assets, manifest.participant_id.tolist())
    primary_provenance = json.loads((ROOT / "outputs/eight/provenance.json").read_text())
    atlas_path = Path(primary_provenance["atlas"]["source_path"])
    atlas = nib.load(atlas_path)
    labels = primary_provenance["atlas"]["labels"]
    names = {r["index"]: r["name"] for r in labels if r["index"] != 0}
    region_ids = sorted(names)
    all_regions, all_scans, failures, cache_hits = [], [], [], []
    for record in manifest.to_dict("records"):
        participant = record["participant_id"]
        destination = OUT / "scans" / participant
        destination.mkdir(parents=True, exist_ok=True)
        try:
            if digest(record["image_path"]) != record["image_sha256"]:
                raise ValueError("Primary input hash differs from immutable manifest")
            for role in ROLES:
                asset = joined[participant].get(role)
                if (
                    not asset
                    or asset["status"] != "complete"
                    or digest(ROOT / asset["path"]) != asset["sha256"]
                ):
                    raise ValueError(
                        f"Supplementary asset missing or changed: {participant}/{role}"
                    )
            key = scan_cache_key(record, joined[participant], atlas_path)
            marker_path = destination / "completed.json"
            marker = json.loads(marker_path.read_text()) if marker_path.exists() else {}
            if marker.get("key") == key and all(
                (destination / p).is_file() and digest(destination / p) == h
                for p, h in marker["artifacts"].items()
            ):
                regions = pd.read_csv(destination / "region_coverage.csv")
                scan = json.loads((destination / "scan.json").read_text())
                cache_hits.append(participant)
                print(f"{participant}: verified content cache reused", flush=True)
            else:
                regions, scan = inspect_scan(
                    record, joined[participant], atlas, region_ids, names, destination
                )
                regions.to_csv(destination / "region_coverage.csv", index=False)
                write_json(destination / "scan.json", scan)
                artifact_names = [
                    "region_coverage.csv",
                    "scan.json",
                    "source_roi_timeseries.csv",
                    "coverage.png",
                    "mask_missing_locations.csv",
                ]
                if (destination / "native_fov.png").exists():
                    artifact_names.append("native_fov.png")
                write_json(
                    marker_path,
                    dict(key=key, artifacts={p: digest(destination / p) for p in artifact_names}),
                )
                print(f"{participant}: coverage measured; primary raw means reconciled", flush=True)
            failure_path = destination / "failure.json"
            if failure_path.exists():
                history = OUT / "attempts"
                history.mkdir(exist_ok=True)
                failure_path.replace(history / f"{participant}-first-failure.json")
            all_regions.append(regions)
            all_scans.append(scan)
        except (ValueError, OSError, KeyError, AssertionError) as exc:
            archive_scan_attempt(destination, OUT / "attempts")
            failures.append(
                dict(
                    participant_id=participant,
                    status="unavailable",
                    error=str(exc),
                    recovery_command="uv run python investigations/eight-adult-coverage/acquire.py && uv run python investigations/eight-adult-coverage/audit.py",
                )
            )
            write_json(destination / "failure.json", failures[-1])
            print(f"{participant}: audit unavailable: {exc}", flush=True)
    if not all_regions:
        write_json(OUT / "failures.json", failures)
        raise ValueError("No real scans could be audited; completed acquisitions are preserved")
    regions = pd.concat(all_regions, ignore_index=True)
    summary = summaries(regions, all_scans)
    issues = issues_from_regions(regions)
    for scan in all_scans:
        if scan["spm8_mask_nonfinite_voxels"]:
            unknown = int(
                summary.loc[
                    summary.participant_id == scan["participant_id"],
                    "archived_mask_unknown_atlas_voxels",
                ].iloc[0]
            )
            issues.loc[len(issues)] = dict(
                participant_id=scan["participant_id"],
                region_id=0,
                region_name="Source mask missingness",
                severity="review",
                rule="undefined_archive_mask_values",
                observed_evidence=f"{scan['spm8_mask_nonfinite_voxels']} NaN mask voxels; {unknown} full-atlas cortical locations overlap unknown values. Exact source indices/world coordinates are exported.",
                file_location=f"{scan['participant_id']}_analysis_mask.nii.gz; see scans/{scan['participant_id']}/mask_missing_locations.csv",
                possible_explanations="Source mask creation or distribution introduced undefined values; exact mechanism is unverified",
                recommended_action="Preserve source bytes, treat these locations as unknown, inspect the coordinate export and request source clarification. Do not fill them as excluded or included.",
            )
    regions.to_csv(OUT / "region_coverage.csv", index=False)
    summary.to_csv(OUT / "participant_summary.csv", index=False)
    issues.to_csv(OUT / "issues.csv", index=False)
    write_json(OUT / "failures.json", failures)
    write_json(OUT / "settings.json", SETTINGS)
    import importlib.metadata

    provenance = dict(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        participant_ids=manifest.participant_id.tolist(),
        primary_manifest_sha256=digest(ROOT / "outputs/eight/manifest.csv"),
        atlas=primary_provenance["atlas"],
        dependencies={
            name: importlib.metadata.version(name)
            for name in (
                "nibabel",
                "nilearn",
                "numpy",
                "pandas",
                "matplotlib",
                "jinja2",
                "requests",
            )
        },
        code_hashes={
            p.name: digest(p) for p in (Path(__file__), OUT / "metrics.py", OUT / "acquire.py")
        },
        source_urls={
            "archive_methods": "https://s3.amazonaws.com/openneuro/ds000228/ds000228_R1.0.0/uncompressed/derivatives/README",
            "tutorial_methods": "https://osf.io/wjtyq/",
            "published_exclusions": PAPER,
        },
        cache_hits=cache_hits,
        connectivity_recomputed=False,
    )
    write_json(OUT / "provenance.json", provenance)
    protection = check_protected()
    render_report(summary, regions, issues, failures, protection, provenance)
    from verify_independently import main as verify_independent

    verify_independent()
    artifact_names = [
        "report.html",
        "REPORT.md",
        "participant_summary.csv",
        "region_coverage.csv",
        "issues.csv",
        "failures.json",
        "settings.json",
        "provenance.json",
        "download_manifest.json",
        "coverage_heatmap.png",
        "audit.py",
        "metrics.py",
        "acquire.py",
        "report.html.j2",
        "UNSENT_SOURCE_REQUEST.md",
        "PORTFOLIO_CASE_STUDY.md",
        "verify_independently.py",
        "independent_verification.json",
        "preservation.json",
        "protected-before.json",
    ]
    for scan in all_scans:
        relative_directory = Path("scans") / scan["participant_id"]
        scan_marker = json.loads((OUT / relative_directory / "completed.json").read_text())
        artifact_names.extend(str(relative_directory / p) for p in scan_marker["artifacts"])
        artifact_names.append(str(relative_directory / "completed.json"))
    artifact_names.extend(
        str(Path("scans") / failure["participant_id"] / "failure.json") for failure in failures
    )
    write_json(
        OUT / "completed.json",
        dict(
            schema_version=1,
            data_kind="real_supplement",
            status="complete" if len(all_scans) == 8 and not failures else "partial",
            participant_ids=summary.participant_id.tolist(),
            artifacts={p: digest(OUT / p) for p in artifact_names},
        ),
    )
    verify()
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
