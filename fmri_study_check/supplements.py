"""Load completed coverage artifacts only; never import scientific or acquisition code."""

import io
import json
import zipfile
from pathlib import Path

import pandas as pd

from .ui import file_digest


def load_supplement(directory):
    directory = Path(directory).resolve()
    marker_path = directory / "completed.json"
    if not marker_path.is_file():
        raise ValueError("No completed coverage supplement; run the separate audit first")
    marker = json.loads(marker_path.read_text())
    if marker.get("schema_version") != 1 or marker.get("data_kind") != "real_supplement":
        raise ValueError("Unsupported coverage supplement or fixture data")
    for relative, expected in marker["artifacts"].items():
        path = (directory / relative).resolve()
        if (
            not path.is_relative_to(directory)
            or not path.is_file()
            or file_digest(path) != expected
        ):
            raise ValueError(f"Coverage artifact is missing or changed: {relative}")
    tables = {
        name: pd.read_csv(directory / f"{name}.csv")
        for name in ("participant_summary", "region_coverage", "issues")
    }
    summary, regions = tables["participant_summary"], tables["region_coverage"]
    if (
        summary.participant_id.duplicated().any()
        or regions.duplicated(["participant_id", "lineage", "region_id"]).any()
    ):
        raise ValueError("Duplicate coverage artifact identifiers")
    if set(summary.participant_id) != set(marker["participant_ids"]):
        raise ValueError("Coverage participant identifiers disagree with completed marker")
    tables.update(
        {
            name: json.loads((directory / f"{name}.json").read_text())
            for name in ("settings", "provenance", "failures")
        }
    )
    tables["status"] = marker["status"]
    return tables


def supplement_archive(directory):
    directory = Path(directory).resolve()
    load_supplement(directory)
    marker = json.loads((directory / "completed.json").read_text())
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in sorted(marker["artifacts"]):
            if Path(relative).suffix in {".csv", ".json", ".html", ".md", ".png", ".py", ".j2"}:
                archive.write(directory / relative, relative)
        archive.write(directory / "completed.json", "completed.json")
    return buffer.getvalue()
