"""Deliberately corrupt a tiny real-data excerpt, isolated from study bundles."""

from pathlib import Path

import nibabel as nib
import pandas as pd

from .pipeline import ISSUE_COLUMNS, inspect_scan, sha256, write_json


def create_corrupted_fixture(record, directory, tr):
    directory = Path(directory).resolve()
    source_paths = [Path(record[field]).resolve() for field in ("image_path", "confounds_path")]
    for source in source_paths:
        target = directory / source.name
        if source.is_relative_to(directory) or (target.exists() and target.samefile(source)):
            raise ValueError(
                "Fixture destination overlaps a preserved source file; choose a separate fixture directory"
            )
    source_hashes = [sha256(path) for path in source_paths]
    directory.mkdir(parents=True, exist_ok=True)
    image = nib.load(record["image_path"])
    if image.shape[3] < 5:
        raise ValueError("Fixture needs at least five real source volumes")
    crop = tuple(slice(max(0, size // 2 - 1), min(size, size // 2 + 2)) for size in image.shape[:3])
    excerpt = image.slicer[(*crop, slice(0, 5))]
    image_path = directory / Path(record["image_path"]).name
    table_path = directory / Path(record["confounds_path"]).name
    nib.save(excerpt, image_path)
    pd.read_csv(record["confounds_path"], sep="\t").iloc[:4].to_csv(
        table_path, sep="\t", index=False
    )
    copied = dict(record, image_path=str(image_path), confounds_path=str(table_path))
    row, _, issues, _ = inspect_scan(copied, tr)
    row["data_kind"] = "fixture"
    pd.DataFrame([row]).to_csv(directory / "manifest.csv", index=False)
    pd.DataFrame(issues, columns=ISSUE_COLUMNS).to_csv(directory / "issues.csv", index=False)
    metadata = dict(
        data_kind="fixture",
        purpose="Deliberate image/confound mismatch demonstration only",
        source_image=record["image_path"],
        source_image_sha256=source_hashes[0],
        source_confounds=record["confounds_path"],
        source_confounds_sha256=source_hashes[1],
        spatial_crop=[[s.start, s.stop] for s in crop],
        original_image_indices=[0, 1, 2, 3, 4],
        original_confound_indices=[0, 1, 2, 3],
        image_volumes=5,
        confound_rows=4,
        exclude_from_real_data_report=True,
    )
    write_json(directory / "fixture.json", metadata)
    (directory / "README.md").write_text(
        "# Deliberately corrupted real-data excerpt\n\n"
        "This tiny spatial crop contains five original image volumes and only four confound rows. "
        "The mismatch is intentional. It is a validation demonstration, never a study participant "
        "or an analysis result. `fixture.json` records the source hashes and original indices. "
        "No fixture observations enter the real-data reports. Source attribution: Richardson et al. "
        "(2018), via Nilearn's development_fmri OSF derivative of OpenNeuro ds000228.\n"
    )
    return metadata
