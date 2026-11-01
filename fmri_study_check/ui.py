"""Read completed artifacts only. No Nilearn imports or scientific computations."""

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pandas as pd

TABLES = (
    "manifest",
    "issues",
    "retention",
    "motion_summary",
    "comparison",
    "frame_exclusions",
    "regions",
)


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bundle(directory):
    directory = Path(directory).resolve()
    marker = directory / "completed.json"
    if not marker.is_file():
        raise ValueError("No completed bundle; run the analysis script first")
    completed = json.loads(marker.read_text())
    for relative, digest in completed["artifacts"].items():
        artifact = (directory / relative).resolve()
        if (
            not artifact.is_relative_to(directory)
            or not artifact.is_file()
            or file_digest(artifact) != digest
        ):
            raise ValueError(
                f"Bundle is incomplete or changed: {relative}. Run the CLI verification."
            )
    bundle = {name: pd.read_csv(directory / f"{name}.csv") for name in TABLES}
    if len(bundle["manifest"]) and not bundle["manifest"].data_kind.eq("real").all():
        raise ValueError("Fixture data cannot be loaded as a real study bundle")
    bundle.update(
        {
            name: json.loads((directory / f"{name}.json").read_text())
            for name in ("summary", "settings", "provenance")
        }
    )
    bundle["directory"] = str(directory)
    return bundle


def export_archive(directory):
    directory = Path(directory).resolve()
    marker = json.loads((directory / "completed.json").read_text())
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in sorted(marker["artifacts"]):
            if Path(relative).suffix in {".csv", ".json", ".txt"}:
                archive.write(directory / relative, relative)
    return buffer.getvalue()
