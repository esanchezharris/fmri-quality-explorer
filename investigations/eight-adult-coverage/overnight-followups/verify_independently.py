"""Independent hemisphere oracle: Nilearn resampling, separate coordinate formula and raw comparisons."""

import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img

OUT = Path(__file__).resolve().parent
AUDIT = OUT.parent
ROOT = AUDIT.parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resample(values, image, target):
    source = nib.Nifti1Image(np.asarray(values, dtype=np.uint8), image.affine)
    return np.asarray(
        resample_to_img(
            source,
            target,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
            fill_value=0,
        ).dataobj
    )


def labels_on(atlas, target):
    return np.asarray(
        resample_to_img(
            atlas,
            target,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
            fill_value=0,
        ).dataobj
    )


def coordinate_strata(image):
    i, j, k = np.ogrid[: image.shape[0], : image.shape[1], : image.shape[2]]
    x = (
        image.affine[0, 0] * i
        + image.affine[0, 1] * j
        + image.affine[0, 2] * k
        + image.affine[0, 3]
    )
    x = np.broadcast_to(x, image.shape[:3])
    # Supplied grids have integer-mm world x, so exact sign partitions independently test the tolerance-based implementation.
    np.testing.assert_allclose(x, np.rint(x), atol=1e-8)
    return {"left": x < 0, "midline": x == 0, "right": x > 0}


class VisibleTable(HTMLParser):
    def __init__(self):
        super().__init__()
        self.inside = self.cell_active = False
        self.row, self.rows, self.cell = [], [], ""

    def handle_starttag(self, tag, attrs):
        if tag == "table" and dict(attrs).get("id") == "focused-counts":
            self.inside = True
        if self.inside and tag == "tr":
            self.row = []
        if self.inside and tag == "td":
            self.cell_active, self.cell = True, ""

    def handle_data(self, data):
        if self.cell_active:
            self.cell += data

    def handle_endtag(self, tag):
        if self.inside and tag == "td":
            self.row.append(self.cell.strip())
            self.cell_active = False
        if self.inside and tag == "tr" and self.row:
            self.rows.append(self.row)
        if self.inside and tag == "table":
            self.inside = False


def main():
    table = (
        pd.read_csv(OUT / "hemisphere_coverage.csv")
        .set_index(["participant_id", "evidence", "region_id", "hemisphere"])
        .sort_index()
    )
    assert len(table) == 144 and table.index.is_unique
    metadata = json.loads((OUT / "provenance.json").read_text())
    atlas = nib.load(metadata["atlas_path"])
    native_labels = np.asarray(atlas.dataobj)
    inputs = {(r["participant_id"], r["role"]): Path(r["path"]) for r in metadata["source_records"]}
    count_checks = fraction_checks = 0
    for pid in metadata["participant_ids"]:
        primary = nib.load(inputs[pid, "tutorial_bold"])
        archived = nib.load(inputs[pid, "spm8_bold"])
        membership = np.asarray(nib.load(inputs[pid, "spm8_mask"]).dataobj)
        # Recompute variation from first-frame differences rather than the analysis min/max or cached maps.
        primary_raw, archive_raw = primary.dataobj.get_unscaled(), archived.dataobj.get_unscaled()
        pv = np.any(primary_raw != primary_raw[..., :1], axis=3)
        av = np.any(archive_raw != archive_raw[..., :1], axis=3)
        for evidence, grid, labels, varying, included, unknown, domain in [
            (
                "tutorial_grid_4mm",
                primary,
                labels_on(atlas, primary),
                pv,
                None,
                None,
                np.ones(primary.shape[:3], dtype=bool),
            ),
            (
                "spm8_grid_2mm",
                archived,
                labels_on(atlas, archived),
                av,
                membership == 1,
                np.isnan(membership),
                np.ones(archived.shape[:3], dtype=bool),
            ),
            (
                "spm8_full_atlas_2mm",
                atlas,
                native_labels,
                resample(av, archived, atlas) > 0,
                resample(membership == 1, archived, atlas) > 0,
                resample(np.isnan(membership), archived, atlas) > 0,
                resample(np.ones(archived.shape[:3], dtype=np.uint8), archived, atlas) > 0,
            ),
        ]:
            for region in (14, 37):
                for hemi, side in coordinate_strata(grid).items():
                    roi = (labels == region) & side
                    record = table.loc[pid, evidence, region, hemi]
                    expected = dict(
                        voxels=int(roi.sum()),
                        in_grid_voxels=int((roi & domain).sum()),
                        varying_voxels=int((roi & varying).sum()),
                    )
                    if included is not None:
                        expected.update(
                            mask_voxels=int((roi & included).sum()),
                            mask_unknown_voxels=int((roi & unknown).sum()),
                        )
                    for column, value in expected.items():
                        assert record[column] == value, (pid, evidence, region, hemi, column)
                        count_checks += 1
                    n = expected["voxels"]
                    if not n:
                        assert pd.isna(record.varying_fraction)
                        if included is not None:
                            assert (
                                pd.isna(record.mask_fraction)
                                and pd.isna(record.mask_fraction_lower)
                                and pd.isna(record.mask_fraction_upper)
                            )
                    else:
                        assert np.isclose(
                            record.varying_fraction, expected["varying_voxels"] / n, atol=1e-12
                        )
                        fraction_checks += 1
                        if included is not None:
                            lower = expected["mask_voxels"] / n
                            upper = (expected["mask_voxels"] + expected["mask_unknown_voxels"]) / n
                            assert np.isclose(record.mask_fraction_lower, lower, atol=1e-12)
                            assert np.isclose(record.mask_fraction_upper, upper, atol=1e-12)
                            if expected["mask_unknown_voxels"]:
                                assert pd.isna(record.mask_fraction)
                            else:
                                assert np.isclose(record.mask_fraction, lower, atol=1e-12)
                            fraction_checks += 3
        del primary_raw, archive_raw, pv, av
    reconciliation = pd.read_csv(OUT / "reconciliation.csv")
    assert (reconciliation.stratum_sum == reconciliation.sealed_count).all()
    parser = VisibleTable()
    parser.feed((OUT / "report.html").read_text())
    assert len(parser.rows) == 16
    evidence_names = {
        "Tutorial 4-mm varying centers": "tutorial_grid_4mm",
        "SPM8 mask / full native atlas stratum": "spm8_full_atlas_2mm",
    }
    for pid, label, region, hemi, ratio, percent in parser.rows:
        evidence = evidence_names[label]
        record = table.loc[pid, evidence, int(region), hemi]
        numerator = int(
            record.mask_voxels if evidence == "spm8_full_atlas_2mm" else record.varying_voxels
        )
        fraction = (
            record.mask_fraction if evidence == "spm8_full_atlas_2mm" else record.varying_fraction
        )
        assert ratio == f"{numerator} / {int(record.voxels)}"
        assert percent == f"{fraction * 100:.1f}%"
    # Literal checks independently derived from actual source-coordinate observations, not this verifier's counting helpers.
    assert table.loc["sub-pixar123", "spm8_full_atlas_2mm", 14, "right"].mask_voxels == 4
    assert table.loc["sub-pixar123", "spm8_full_atlas_2mm", 37, "right"].mask_voxels == 1
    assert table.loc["sub-pixar123", "spm8_full_atlas_2mm", 14, "left"].mask_voxels == 167
    assert table.loc["sub-pixar123", "spm8_full_atlas_2mm", 37, "left"].mask_voxels == 49
    proof = dict(
        status="passed",
        independent_method="Nilearn nearest-neighbor resampling; explicit affine coefficient coordinate formula; raw first-frame comparison",
        hemisphere_rows_checked=len(table),
        integer_count_checks=count_checks,
        fraction_checks=fraction_checks,
        unsplit_count_reconciliations=len(reconciliation),
        visible_report_rows_reconciled=len(parser.rows),
        midline_zero_denominator_handling="verified unavailable",
        tutorial_vs_mask_denominators="kept separate",
        input_artifacts={
            name: digest(OUT / name)
            for name in (
                "hemisphere_coverage.csv",
                "reconciliation.csv",
                "report.html",
                "provenance.json",
            )
        },
    )
    (OUT / "independent_verification.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
