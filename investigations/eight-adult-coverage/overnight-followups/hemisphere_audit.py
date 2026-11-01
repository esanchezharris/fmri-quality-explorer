"""Coordinate-stratified coverage follow-up; no source/result edits or connectivity analysis."""

import argparse
import base64
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

OUT = Path(__file__).resolve().parent
AUDIT = OUT.parent
ROOT = AUDIT.parents[1]
REGIONS = (14, 37)
STRATA = ("left", "midline", "right")
SETTINGS = {
    "scope": "Eight original adults, existing atlas IDs 14 and 37; descriptive coordinate stratification only",
    "strata": "Supplied MNI world x < -1e-6 mm: left; abs(x) <= 1e-6 mm: midline; x > 1e-6 mm: right",
    "atlas": "Existing unsplit Harvard-Oxford cort-maxprob-thr25-2mm; no replacement atlas or estimated registration",
    "mask_denominator": "All native 2-mm atlas parcel centers in each x stratum, including out-of-image-grid locations",
    "tutorial_denominator": "Parcel centers after nearest-neighbor placement on the existing 4-mm tutorial grid, stratified by their own world x",
    "spm8_grid_denominator": "Parcel centers after nearest-neighbor placement on the archived 2-mm SPM8 grid",
    "variation": "Exact stored-code minimum differs from maximum across all acquired frames; checked finite nonzero NIfTI scaling preserves constancy",
    "undefined_mask": "NaNs stay unknown. Lower=known included/total, upper=(known included+unknown)/total; no point fraction if unknown>0",
    "empty_stratum": "Zero denominator is unavailable, never 0 percent or 100 percent coverage",
    "sampling": "Explicit nearest-center affine sampling, half ties rounded up; outside source center domain has no image support",
    "frames": "All original acquired indices, 0 through n_volumes-1; no new frame exclusions or cleaning",
    "limits": "Independent SPM8 mask is not matching 2019 fMRIPrep mask. Normalization/template differences and quantization limit causal interpretation. Temporal variation does not certify biological signal.",
    "primary_results": "No changes to source files, primary results, atlas, pipeline, caches or sealed 70-artifact coverage audit",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def preservation():
    before = json.loads((OUT / "protected-before.json").read_text())
    removed = [name for name in before if not (ROOT / name).is_file()]
    changed = [
        name for name, h in before.items() if (ROOT / name).is_file() and sha256(ROOT / name) != h
    ]
    current = set()
    for folder in (
        "data/source",
        "outputs/one",
        "outputs/eight",
        ".cache",
        "fmri_study_check",
        "tests",
        "docs",
        "investigations/sub-pixar124-region14",
        "investigations/sub-pixar124-upstream",
        "investigations/eight-adult-coverage",
    ):
        current.update(
            str(p.relative_to(ROOT))
            for p in (ROOT / folder).rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and not p.is_relative_to(OUT)
            and p != AUDIT / "OVERNIGHT_HANDOFF.md"
        )
    current.update(
        name
        for name in ("app.py", "README.md", "pyproject.toml", "uv.lock", ".python-version")
        if (ROOT / name).is_file()
    )
    added = sorted(current - before.keys())
    marker = json.loads((AUDIT / "completed.json").read_text())
    invalid = [name for name, h in marker["artifacts"].items() if sha256(AUDIT / name) != h]
    result = dict(
        protected_files=len(before),
        modified=changed,
        removed=removed,
        added=added,
        sealed_audit_artifacts=len(marker["artifacts"]),
        sealed_artifact_mismatches=invalid,
        unchanged=not (removed or changed or added or invalid),
    )
    save_json(OUT / "preservation.json", result)
    if not result["unchanged"]:
        raise ValueError(f"Preservation check failed: {result}")
    return result


def hemisphere_masks(world_x):
    x = np.asarray(world_x)
    if not np.isfinite(x).all():
        raise ValueError("World coordinates must be finite")
    return {"left": x < -1e-6, "midline": np.abs(x) <= 1e-6, "right": x > 1e-6}


def stratify(labels, world_x, varying, region, mask=None, domain=None):
    labels, world_x, varying = np.asarray(labels), np.asarray(world_x), np.asarray(varying)
    if labels.shape != world_x.shape or labels.shape != varying.shape:
        raise ValueError("Label, coordinate and variation grids must agree")
    if mask is not None:
        mask = np.asarray(mask)
        if (
            mask.shape != labels.shape
            or np.isinf(mask).any()
            or not np.isin(mask[np.isfinite(mask)], [0, 1]).all()
        ):
            raise ValueError("Finite mask values must be binary and match the grid")
    if domain is None:
        domain = np.ones(labels.shape, dtype=bool)
    rows = []
    for name, hemisphere in hemisphere_masks(world_x).items():
        roi = (labels == region) & hemisphere
        n = int(roi.sum())
        nv = int((roi & varying.astype(bool)).sum())
        row = dict(
            region_id=int(region),
            hemisphere=name,
            voxels=n,
            in_grid_voxels=int((roi & domain).sum()),
            varying_voxels=nv,
            varying_fraction=nv / n if n else None,
        )
        if mask is not None:
            included = int((roi & (mask == 1)).sum())
            unknown = int((roi & np.isnan(mask)).sum())
            lower = included / n if n else None
            upper = (included + unknown) / n if n else None
            row.update(
                mask_voxels=included,
                mask_unknown_voxels=unknown,
                mask_fraction=lower if not unknown else None,
                mask_fraction_lower=lower,
                mask_fraction_upper=upper,
            )
        rows.append(row)
    return rows


def world_x(image):
    coords = np.indices(image.shape[:3]).reshape(3, -1).T
    return nib.affines.apply_affine(image.affine, coords)[:, 0].reshape(image.shape[:3])


def sample_on_grid(values, source, target):
    """Sample source values at target voxel centers without silently dropping target locations."""
    coords = np.indices(target.shape[:3]).reshape(3, -1).T
    mapped = nib.affines.apply_affine(np.linalg.inv(source.affine) @ target.affine, coords)
    valid = np.all((mapped >= 0) & (mapped <= np.array(source.shape[:3]) - 1), axis=1)
    indices = np.floor(mapped + 0.5).astype(int)
    result = np.zeros(target.shape[:3], dtype=np.asarray(values).dtype)
    result.ravel()[valid] = np.asarray(values)[tuple(indices[valid].T)]
    return result, valid.reshape(target.shape[:3])


def varying_map(image):
    if len(image.shape) != 4 or image.shape[3] < 2:
        raise ValueError("A readable four-dimensional run with at least two frames is required")
    if (
        not np.isfinite([image.dataobj.slope, image.dataobj.inter]).all()
        or image.dataobj.slope == 0
    ):
        raise ValueError("Invalid NIfTI scaling")
    values = image.dataobj.get_unscaled()
    if not np.isfinite(values).all():
        raise ValueError("Non-finite BOLD observations")
    return values.min(axis=3) != values.max(axis=3)


def inputs():
    manifest = pd.read_csv(ROOT / "outputs/eight/manifest.csv")
    if (
        manifest.participant_id.duplicated().any()
        or set(manifest.participant_id) != {f"sub-pixar{i}" for i in range(123, 131)}
        or not manifest.data_kind.eq("real").all()
    ):
        raise ValueError("Original eight real participants must be retained exactly")
    records = json.loads((AUDIT / "download_manifest.json").read_text())
    assets = {}
    for r in records:
        key = (r["participant_id"], r["role"])
        if key in assets or not r["source_filename"].startswith(r["participant_id"] + "_"):
            raise ValueError("Duplicate or mismatched supplementary source identifier")
        assets[key] = r
    provenance = json.loads((AUDIT / "provenance.json").read_text())
    atlas_path = Path(provenance["atlas"]["source_path"])
    atlas = nib.load(atlas_path)
    names = {r["index"]: r["name"] for r in provenance["atlas"]["labels"]}
    return manifest, assets, atlas_path, atlas, names


def measure(record, assets, atlas, names, directory):
    pid = record["participant_id"]
    source = Path(record["image_path"])
    spm = ROOT / assets[pid, "spm8_bold"]["path"]
    mask_path = ROOT / assets[pid, "spm8_mask"]["path"]
    primary, archive, mask_image = nib.load(source), nib.load(spm), nib.load(mask_path)
    if archive.shape[3] != primary.shape[3] or primary.shape[3] != record["n_volumes"]:
        raise ValueError("Original-frame counts disagree")
    if mask_image.shape != archive.shape[:3] or not np.array_equal(
        mask_image.affine, archive.affine
    ):
        raise ValueError("Archived BOLD and mask grids disagree")
    membership = np.asarray(mask_image.dataobj)
    if np.isinf(membership).any() or not np.isin(membership[np.isfinite(membership)], [0, 1]).all():
        raise ValueError("Invalid finite mask values")
    atlas_labels = np.asarray(atlas.dataobj)
    primary_labels, _ = sample_on_grid(atlas_labels, atlas, primary)
    archive_labels, _ = sample_on_grid(atlas_labels, atlas, archive)
    pv, av = varying_map(primary), varying_map(archive)
    atlas_variation, atlas_domain = sample_on_grid(av, archive, atlas)
    included, _ = sample_on_grid((membership == 1).astype(np.uint8), archive, atlas)
    unknown, _ = sample_on_grid(np.isnan(membership).astype(np.uint8), archive, atlas)
    atlas_membership = included.astype(float)
    atlas_membership[unknown > 0] = np.nan
    rows = []
    for evidence, labels, image, variation, mask, domain in [
        ("tutorial_grid_4mm", primary_labels, primary, pv, None, None),
        ("spm8_grid_2mm", archive_labels, archive, av, membership, None),
        (
            "spm8_full_atlas_2mm",
            atlas_labels,
            atlas,
            atlas_variation,
            atlas_membership,
            atlas_domain,
        ),
    ]:
        x = world_x(image)
        for region in REGIONS:
            for row in stratify(labels, x, variation, region, mask, domain):
                rows.append(
                    dict(
                        participant_id=pid,
                        scan_key=record["scan_key"],
                        evidence=evidence,
                        region_name=names[region],
                        **row,
                    )
                )
    result = pd.DataFrame(rows)
    result.to_csv(directory / "hemisphere_coverage.csv", index=False)
    np.savez_compressed(directory / "variation_maps.npz", tutorial=pv, spm8=av)
    return result


def reconcile(table):
    sealed = pd.read_csv(AUDIT / "region_coverage.csv").set_index(
        ["participant_id", "lineage", "region_id"]
    )
    checks = []
    for (pid, evidence, region), group in table.groupby(
        ["participant_id", "evidence", "region_id"]
    ):
        original = sealed.loc[
            pid, "tutorial_4mm" if evidence == "tutorial_grid_4mm" else "archive_spm8_2mm", region
        ]
        prefix = "atlas" if evidence == "spm8_full_atlas_2mm" else "grid"
        fields = {"voxels": f"{prefix}_voxels", "varying_voxels": f"{prefix}_varying_voxels"}
        if prefix == "atlas":
            fields["in_grid_voxels"] = "atlas_in_grid_voxels"
        if evidence != "tutorial_grid_4mm":
            fields.update(
                mask_voxels=f"{prefix}_mask_voxels",
                mask_unknown_voxels=f"{prefix}_mask_unknown_voxels",
            )
        for new, old in fields.items():
            observed, expected = int(group[new].sum()), int(original[old])
            if observed != expected:
                raise ValueError(
                    f"Stratum sums disagree with sealed counts: {pid}/{evidence}/{region}/{new}"
                )
            checks.append(
                dict(
                    participant_id=pid,
                    evidence=evidence,
                    region_id=int(region),
                    measure=new,
                    stratum_sum=observed,
                    sealed_count=expected,
                    status="matched",
                )
            )
    pd.DataFrame(checks).to_csv(OUT / "reconciliation.csv", index=False)
    return len(checks)


def report(table, metadata):
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained", sharey=True)
    participants = metadata["participant_ids"]
    for column, region in enumerate(REGIONS):
        for row, evidence, field, title in [
            (
                0,
                "spm8_full_atlas_2mm",
                "mask_fraction",
                "Independent SPM8 mask / native atlas stratum",
            ),
            (
                1,
                "tutorial_grid_4mm",
                "varying_fraction",
                "Tutorial varying centers / 4-mm grid stratum",
            ),
        ]:
            axis = axes[row, column]
            for offset, hemi, color in [(-0.2, "left", "#246d91"), (0.2, "right", "#c48428")]:
                data = (
                    table.loc[
                        (table.evidence == evidence)
                        & (table.region_id == region)
                        & (table.hemisphere == hemi)
                    ]
                    .set_index("participant_id")
                    .reindex(participants)
                )
                axis.bar(
                    np.arange(8) + offset,
                    data[field] * 100,
                    width=0.38,
                    label=hemi.title(),
                    color=color,
                )
            axis.set(
                title=f"Region {region}: {title}",
                xticks=np.arange(8),
                xticklabels=[x.removeprefix("sub-pixar") for x in participants],
                ylim=(0, 105),
                ylabel="Percent (fixed 0–100 scale)",
            )
            axis.legend(loc="upper left", fontsize=8)
    figure.suptitle(
        "World-x coverage stratification: separate evidence layers\nMidline retained in CSVs; empty strata are unavailable. No new registration or primary exclusions.",
        fontsize=13,
    )
    figure.savefig(OUT / "hemisphere_coverage.png", dpi=140)
    plt.close(figure)
    focus = table.loc[
        table.participant_id.isin(["sub-pixar123", "sub-pixar124"]) & table.hemisphere.ne("midline")
    ].copy()
    environment = Environment(
        loader=FileSystemLoader(OUT), autoescape=select_autoescape(["html", "j2"])
    )
    figure_data = (
        "data:image/png;base64,"
        + base64.b64encode((OUT / "hemisphere_coverage.png").read_bytes()).decode()
    )
    html = environment.get_template("report.html.j2").render(
        rows=focus.to_dict("records"),
        all_rows=table.to_dict("records"),
        settings=SETTINGS,
        generated=metadata["generated_at_utc"],
        figure=figure_data,
        protected=metadata["preservation"],
        evidence_labels={
            "tutorial_grid_4mm": "Tutorial 4-mm varying centers",
            "spm8_grid_2mm": "SPM8 2-mm varying centers",
            "spm8_full_atlas_2mm": "SPM8 mask / full native atlas stratum",
        },
    )
    (OUT / "report.html").write_text(html)
    text = "# Hemisphere-level coverage follow-up\n\nCoordinate stratification of the existing unsplit atlas; no new registration, connectivity calculation or primary exclusion.\n\n"
    for pid in ("sub-pixar123", "sub-pixar124"):
        text += f"## {pid}\n\n"
        for region in REGIONS:
            for hemi in ("left", "right"):
                mask = table.loc[
                    (table.participant_id == pid)
                    & (table.region_id == region)
                    & (table.hemisphere == hemi)
                    & table.evidence.eq("spm8_full_atlas_2mm")
                ].iloc[0]
                primary = table.loc[
                    (table.participant_id == pid)
                    & (table.region_id == region)
                    & (table.hemisphere == hemi)
                    & table.evidence.eq("tutorial_grid_4mm")
                ].iloc[0]
                text += f"- Region {region}, {hemi}: independent mask {int(mask.mask_voxels)}/{int(mask.voxels)} native atlas centers; tutorial variation {int(primary.varying_voxels)}/{int(primary.voxels)} grid centers.\n"
        text += "\n"
    text += "## Interpretation and limits\n\nThese are supplied-coordinate strata, not newly registered hemisphere-specific atlases. Independent SPM8 mask inclusion and tutorial temporal variation have different denominators and meanings and must not be subtracted as a preprocessing effect. The SPM8 mask is not the matching 2019 fMRIPrep mask; template/registration differences and int8 quantization remain limits. Missing mask membership stays unknown, and empty strata are unavailable. No observation was imputed, no matrix recalculated and no biological-validity claim made. See the full CSV for all eight original participants and explicit midline rows.\n\n"
    text += "## Reproduce\n\n```sh\nuv run python investigations/eight-adult-coverage/overnight-followups/hemisphere_audit.py\nuv run python investigations/eight-adult-coverage/overnight-followups/verify_independently.py\nuv run pytest -q investigations/eight-adult-coverage/overnight-followups/test_hemisphere.py -p no:cacheprovider\n```\n\nExact source paths/hashes, dependency versions, all analysis settings and original acquired indices are recorded in provenance.json and settings.json. New content-based caches live only under this follow-up directory.\n"
    (OUT / "REPORT.md").write_text(text)


def verify():
    marker = json.loads((OUT / "completed.json").read_text())
    for relative, expected in marker["artifacts"].items():
        path = (OUT / relative).resolve()
        if not path.is_relative_to(OUT) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Changed follow-up artifact: {relative}")
    table = pd.read_csv(OUT / "hemisphere_coverage.csv")
    if (
        len(table) != 144
        or table.duplicated(["participant_id", "evidence", "region_id", "hemisphere"]).any()
    ):
        raise ValueError("Unexpected hemisphere export grain")
    protection = preservation()
    print(
        json.dumps(
            dict(status="verified", table_rows=len(table), preservation=protection), indent=2
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify()
        return
    protected = preservation()
    manifest, assets, atlas_path, atlas, names = inputs()
    rows, cache_hits, source_records = [], [], []
    for record in manifest.to_dict("records"):
        pid = record["participant_id"]
        directory = OUT / "scans" / pid
        directory.mkdir(parents=True, exist_ok=True)
        paths = [("tutorial_bold", Path(record["image_path"]), record["image_sha256"])]
        for role in ("spm8_bold", "spm8_mask"):
            asset = assets[pid, role]
            if asset["status"] != "complete":
                raise ValueError(f"Required source asset unavailable: {pid}/{role}")
            paths.append((role, ROOT / asset["path"], asset["sha256"]))
        for role, path, expected in paths:
            if sha256(path) != expected:
                raise ValueError(f"Source hash changed: {pid}/{role}")
            source_records.append(
                dict(participant_id=pid, role=role, path=str(path), sha256=expected)
            )
        key = hashlib.sha256(
            json.dumps(
                dict(
                    sources=[p[2] for p in paths],
                    atlas=sha256(atlas_path),
                    code=sha256(__file__),
                    settings=SETTINGS,
                ),
                sort_keys=True,
            ).encode()
        ).hexdigest()
        marker_path = directory / "completed.json"
        marker = json.loads(marker_path.read_text()) if marker_path.is_file() else {}
        if marker.get("key") == key and all(
            (directory / p).is_file() and sha256(directory / p) == h
            for p, h in marker["artifacts"].items()
        ):
            result = pd.read_csv(directory / "hemisphere_coverage.csv")
            cache_hits.append(pid)
            print(f"{pid}: verified follow-up cache reused", flush=True)
        else:
            result = measure(record, assets, atlas, names, directory)
            save_json(
                marker_path,
                dict(
                    key=key,
                    artifacts={
                        p: sha256(directory / p)
                        for p in ("hemisphere_coverage.csv", "variation_maps.npz")
                    },
                ),
            )
            print(f"{pid}: coordinate-stratified counts measured", flush=True)
        rows.append(result)
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(OUT / "hemisphere_coverage.csv", index=False)
    reconciled = reconcile(table)
    save_json(OUT / "settings.json", SETTINGS)
    metadata = dict(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        participant_ids=manifest.participant_id.tolist(),
        source_records=source_records,
        atlas_path=str(atlas_path),
        atlas_sha256=sha256(atlas_path),
        original_acquisition_indices={
            r["participant_id"]: {
                "first": 0,
                "last": int(r["n_volumes"]) - 1,
                "count": int(r["n_volumes"]),
                "tr_seconds": float(r["sampling_interval_seconds"]),
            }
            for r in manifest.to_dict("records")
        },
        dependencies={
            p: importlib.metadata.version(p)
            for p in ("numpy", "pandas", "nibabel", "nilearn", "matplotlib", "jinja2")
        },
        code_sha256=sha256(__file__),
        cache_hits=cache_hits,
        reconciled_count_checks=reconciled,
        preservation=protected,
        connectivity_computed=False,
    )
    save_json(OUT / "provenance.json", metadata)
    report(table, metadata)
    protection = preservation()
    artifacts = [
        "hemisphere_coverage.csv",
        "reconciliation.csv",
        "report.html",
        "REPORT.md",
        "hemisphere_coverage.png",
        "provenance.json",
        "settings.json",
        "hemisphere_audit.py",
        "report.html.j2",
        "preservation.json",
        "protected-before.json",
    ]
    artifacts.extend(str(p.relative_to(OUT)) for p in (OUT / "scans").rglob("*") if p.is_file())
    save_json(
        OUT / "completed.json",
        dict(
            status="complete",
            data_kind="real_supplement",
            participant_ids=metadata["participant_ids"],
            artifacts={p: sha256(OUT / p) for p in artifacts},
            preservation=protection,
        ),
    )
    verify()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, AssertionError, KeyError) as error:
        save_json(
            OUT / "failure.json",
            dict(
                error=str(error),
                recovery_command="uv run python investigations/eight-adult-coverage/overnight-followups/hemisphere_audit.py",
            ),
        )
        raise
