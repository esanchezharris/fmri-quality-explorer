"""Independent oracle: direct affine indexing, not Nilearn resampling or audit helpers."""

import json
from html.parser import HTMLParser
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


class SummaryTable(HTMLParser):
    def __init__(self):
        super().__init__()
        self.inside = False
        self.in_cell = False
        self.cell = ""
        self.row = []
        self.rows = []

    def handle_starttag(self, tag, attrs):
        if tag == "table" and dict(attrs).get("id") == "participant-summary":
            self.inside = True
        if self.inside and tag == "tr":
            self.row = []
        if self.inside and tag == "td":
            self.in_cell, self.cell = True, ""

    def handle_data(self, data):
        if self.in_cell:
            self.cell += data

    def handle_endtag(self, tag):
        if self.inside and tag == "td":
            self.row.append(self.cell.strip())
            self.in_cell = False
        if self.inside and tag == "tr" and self.row:
            self.rows.append(self.row)
        if self.inside and tag == "table":
            self.inside = False


def integer_locations(source, target, coordinates, require_integer=False):
    mapped = nib.affines.apply_affine(np.linalg.inv(target.affine) @ source.affine, coordinates)
    # Grid centers map exactly into the native atlas; the reverse 2-to-4-mm map has half ties.
    if require_integer:
        np.testing.assert_allclose(mapped, np.rint(mapped), atol=1e-6)
    ijk = np.floor(mapped + 0.5).astype(int)
    valid = np.all((mapped >= 0) & (mapped <= np.array(target.shape[:3]) - 1), axis=1)
    return ijk, valid


def main():
    regions = (
        pd.read_csv(OUT / "region_coverage.csv")
        .set_index(["participant_id", "lineage", "region_id"])
        .sort_index()
    )
    summary = pd.read_csv(OUT / "participant_summary.csv").set_index("participant_id")
    manifest = pd.read_csv(ROOT / "outputs/eight/manifest.csv").set_index("participant_id")
    provenance = json.loads((OUT / "provenance.json").read_text())
    atlas = nib.load(provenance["atlas"]["source_path"])
    atlas_values = np.asarray(atlas.dataobj)
    assets = json.loads((OUT / "download_manifest.json").read_text())
    by_key = {(r["participant_id"], r["role"]): ROOT / r["path"] for r in assets}
    grid_checks = reference_checks = 0
    for participant, record in manifest.iterrows():
        if participant not in summary.index:
            continue  # Explicitly unavailable scans remain in failures.json, never replaced.
        for lineage, source in [
            ("tutorial_4mm", Path(record.image_path)),
            ("archive_spm8_2mm", by_key[participant, "spm8_bold"]),
        ]:
            image = nib.load(source)
            data = image.dataobj.get_unscaled()
            variation = data.min(axis=3) != data.max(axis=3)
            coords = np.indices(image.shape[:3]).reshape(3, -1).T
            ai, valid = integer_locations(image, atlas, coords, require_integer=True)
            labels = np.zeros(len(coords), dtype=np.int16)
            labels[valid] = atlas_values[tuple(ai[valid].T)]
            mask = (
                np.asarray(nib.load(by_key[participant, "spm8_mask"]).dataobj)
                if lineage == "archive_spm8_2mm"
                else None
            )
            for region in range(1, 49):
                row = regions.loc[participant, lineage, region]
                grid_roi = labels == region
                assert int(grid_roi.sum()) == row.grid_voxels
                assert int(variation.ravel()[grid_roi].sum()) == row.grid_varying_voxels
                grid_checks += 1
                atlas_coords = np.argwhere(atlas_values == region)
                ii, in_domain = integer_locations(atlas, image, atlas_coords)
                assert len(atlas_coords) == row.atlas_voxels
                assert int(in_domain.sum()) == row.atlas_in_grid_voxels
                assert int(variation[tuple(ii[in_domain].T)].sum()) == row.atlas_varying_voxels
                if mask is not None:
                    direct = mask[tuple(ii[in_domain].T)]
                    assert np.count_nonzero(direct == 1) == row.atlas_mask_voxels
                    assert np.isnan(direct).sum() == row.atlas_mask_unknown_voxels
                reference_checks += 1
            del data, variation
        archive = regions.loc[participant, "archive_spm8_2mm"]
        assert (
            archive.atlas_mask_voxels.sum()
            == summary.loc[participant, "archived_mask_atlas_voxels"]
        )
        assert (
            archive.atlas_mask_unknown_voxels.sum()
            == summary.loc[participant, "archived_mask_unknown_atlas_voxels"]
        )
    # Reconcile the visible HTML summary cells, not just a hidden JSON payload.
    parser = SummaryTable()
    parser.feed((OUT / "report.html").read_text())
    assert len(parser.rows) == len(summary)
    for participant, status, overlap, zero, low, sparse, constant in parser.rows:
        row = summary.loc[participant]
        assert status == row.primary_status
        assert overlap == f"{row.archived_mask_cortical_atlas_fraction * 100:.1f}%"
        assert [int(x) for x in (zero, low, sparse, constant)] == [
            int(row[x])
            for x in (
                "archived_zero_mask_regions",
                "archived_regions_below_80pct",
                "tutorial_sparse_regions_below_50pct",
                "tutorial_constant_regional_means",
            )
        ]
    for participant, column, expected in [
        ("sub-pixar123", "spm8_r14_atlas_mask_voxels", 171),
        ("sub-pixar123", "spm8_r37_atlas_mask_voxels", 50),
        ("sub-pixar123", "tutorial_r37_grid_varying_voxels", 20),
        ("sub-pixar124", "tutorial_r14_grid_varying_voxels", 0),
        ("sub-pixar130", "spm8_mask_nonfinite_voxels", 152),
        ("sub-pixar130", "archived_mask_unknown_atlas_voxels", 0),
    ]:
        if participant in summary.index:
            assert summary.loc[participant, column] == expected
    result = dict(
        status="passed" if len(summary) == 8 else "passed_for_available_scans",
        independent_method="Direct affine indexing with explicit nearest-center sampling and voxel min/max; no audit metric helpers or Nilearn resampling",
        grid_region_checks=grid_checks,
        full_atlas_region_checks=reference_checks,
        html_summary_rows_reconciled=len(parser.rows),
        all_eight_verified=len(summary) == 8,
        special_123_124_and_unknown_130_checks="passed",
    )
    (OUT / "independent_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
