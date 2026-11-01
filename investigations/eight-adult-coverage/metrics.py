"""Coverage counts with explicit grid and full-atlas denominators; no signal cleaning."""

import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img


def join_assets(records, participants):
    """Join by participant and role, independent of listing order."""
    if len(set(participants)) != len(participants):
        raise ValueError("Duplicate participant identifiers")
    result = {participant: {} for participant in participants}
    for record in records:
        participant = record["participant_id"]
        if participant not in result:
            raise ValueError(f"Unexpected participant: {participant}")
        if "source_filename" in record and not record["source_filename"].startswith(
            participant + "_"
        ):
            raise ValueError(f"Filename does not match participant: {record}")
        role = record["role"]
        if role in result[participant]:
            raise ValueError(f"Duplicate participant/role: {participant}/{role}")
        result[participant][role] = record
    return result


def region_profile(values, labels, region_ids, mask=None):
    """Count exact stored-code variation and calculate uncleaned full-run regional means."""
    values, labels = np.asarray(values), np.asarray(labels)
    if values.ndim != 4 or values.shape[:3] != labels.shape or values.shape[3] < 2:
        raise ValueError("Image and atlas label grids must agree and contain at least two frames")
    if not np.isfinite(values).all():
        raise ValueError("BOLD observations must be finite")
    if mask is not None:
        mask = np.asarray(mask)
        if (
            mask.shape != labels.shape
            or not np.isin(mask[np.isfinite(mask)], [0, 1]).all()
            or np.isinf(mask).any()
        ):
            raise ValueError(
                "Supplied mask must have binary finite values and match the image grid"
            )
    varying = np.any(values != values[..., :1], axis=3)
    rows, means = [], []
    for region in region_ids:
        roi = labels == region
        n = int(roi.sum())
        observed = int((roi & varying).sum())
        signal = values[roi].mean(axis=0, dtype=np.float64) if n else None
        constant = bool(np.all(signal == signal[0])) if n else None
        row = dict(
            region_id=int(region),
            grid_voxels=n,
            grid_varying_voxels=observed,
            grid_varying_fraction=observed / n if n else None,
            regional_mean_constant=constant,
            regional_mean_sd_stored_units=(0.0 if constant else float(signal.std())) if n else None,
        )
        if mask is not None:
            row.update(mask_counts(roi, mask == 1, np.isnan(mask), "grid"))
        rows.append(row)
        means.append(signal)
    return rows, varying, means


def mask_counts(roi, included, unknown, prefix):
    total = int(roi.sum())
    known_count = int((roi & included).sum())
    unknown_count = int((roi & unknown).sum())
    lower = known_count / total if total else None
    upper = (known_count + unknown_count) / total if total else None
    return {
        f"{prefix}_mask_voxels": known_count,
        f"{prefix}_mask_unknown_voxels": unknown_count,
        f"{prefix}_mask_fraction": lower if not unknown_count else None,
        f"{prefix}_mask_fraction_lower": lower,
        f"{prefix}_mask_fraction_upper": upper,
    }


def nearest(volume, source, target):
    image = nib.Nifti1Image(np.asarray(volume, dtype=np.uint8), source.affine)
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


def reference_profile(atlas, image, varying, region_ids, mask=None):
    """Keep the full atlas parcel as denominator, including locations outside the image grid."""
    if varying.shape != image.shape[:3]:
        raise ValueError("Variation grid does not match image")
    domain = nearest(np.ones(image.shape[:3], dtype=np.uint8), image, atlas)
    variable = nearest(varying, image, atlas)
    included = nearest(np.asarray(mask) == 1, image, atlas) if mask is not None else None
    unknown = nearest(np.isnan(mask), image, atlas) if mask is not None else None
    labels = np.asarray(atlas.dataobj)
    rows = []
    for region in region_ids:
        roi = labels == region
        n = int(roi.sum())
        n_domain = int((roi & (domain > 0)).sum())
        n_variable = int((roi & (variable > 0)).sum())
        row = dict(
            region_id=int(region),
            atlas_voxels=n,
            atlas_in_grid_voxels=n_domain,
            atlas_in_grid_fraction=n_domain / n if n else None,
            atlas_varying_voxels=n_variable,
            atlas_varying_fraction=n_variable / n if n else None,
        )
        if included is not None:
            row.update(mask_counts(roi, included > 0, unknown > 0, "atlas"))
        rows.append(row)
    return rows
