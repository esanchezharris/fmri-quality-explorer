"""Artifact-only Coverage Explorer; no imaging, acquisition or analysis imports."""

import base64
import csv
import hashlib
import json
import re
from pathlib import Path

WEB = Path(__file__).resolve().parent / "web"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def script_json(value):
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def verify_primary_binding(metadata, bundle):
    """Compare selected scan artifacts without acquisition, imaging or analysis work."""
    bundle = Path(bundle)
    with (bundle / "manifest.csv").open(newline="") as stream:
        rows = [
            r for r in csv.DictReader(stream) if r["participant_id"] == metadata["participant_id"]
        ]
    if len(rows) != 1:
        raise ValueError("Prepared display and current primary participant identifiers disagree")
    row = rows[0]
    same = (
        row["scan_key"] == metadata["scan_key"]
        and row["image_sha256"] == metadata["sources"]["tutorial"]["source_sha256"]
        and row["processing_status"] == metadata["primary_status"]
        and float(row["n_volumes"]) == metadata["n_volumes"]
        and float(row["sampling_interval_seconds"]) == metadata["tr_seconds"]
    )
    for name, expected in metadata["primary_artifacts"].items():
        same = same and digest(bundle / "scans" / row["scan_key"] / name) == expected
    with (bundle / "retention.csv").open(newline="") as stream:
        selected = [r for r in csv.DictReader(stream) if r["scan_key"] == row["scan_key"]]
    conditions = []
    for r in selected:
        conditions.append(
            dict(
                condition=r["condition"],
                status=r["status"],
                reason=r["reason"] or None,
                retained_volumes=int(r["retained_volumes"]),
                retained_minutes=float(r["retained_minutes"]),
                residual_df=int(r["residual_df"]),
            )
        )
    same = same and sorted(conditions, key=lambda r: r["condition"]) == sorted(
        metadata["primary_analysis"], key=lambda r: r["condition"]
    )
    if not same:
        raise ValueError(
            "Prepared display is stale relative to the selected primary results; rerun display preparation"
        )


def load_explorer(directory, participant_id, primary_bundle=None):
    directory = Path(directory).resolve()
    if not (directory / "completed.json").is_file():
        raise ValueError(
            "No prepared Coverage Explorer assets; run the display preparation command"
        )
    if not re.fullmatch(r"sub-pixar\d+", participant_id):
        raise ValueError("Invalid participant identifier")
    marker = json.loads((directory / "completed.json").read_text())
    if marker.get("schema_version") != 1 or marker.get("data_kind") != "real_display_assets":
        raise ValueError("Unsupported prepared display assets or fixture data")
    if participant_id not in marker["participant_ids"]:
        raise ValueError(
            f"No prepared display assets for {participant_id}; completed work is preserved"
        )
    prefix = f"scans/{participant_id}/"
    selected = {name: h for name, h in marker["artifacts"].items() if name.startswith(prefix)}
    for name in ("provenance.json", "failures.json"):
        if name not in marker["artifacts"]:
            raise ValueError("Display artifact manifest is incomplete")
        selected[name] = marker["artifacts"][name]
    for relative, expected in selected.items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or digest(path) != expected:
            raise ValueError(f"Display artifact is missing or changed: {relative}")
    metadata_name = prefix + "metadata.json"
    if metadata_name not in selected:
        raise ValueError("Prepared metadata artifact is missing")
    metadata = json.loads((directory / metadata_name).read_text())
    metadata["asset_key"] = json.loads((directory / prefix / "completed.json").read_text())["key"]
    if metadata.get("participant_id") != participant_id or set(metadata["sources"]) != {
        "tutorial",
        "archive",
    }:
        raise ValueError("Display participant or evidence-lineage identifiers disagree")
    ids = [r["id"] for r in metadata["regions"]]
    if len(ids) != len(set(ids)) or ids != list(range(1, 49)):
        raise ValueError("Prepared atlas region identities are invalid")
    n = metadata["n_volumes"]
    if metadata["motion"]["original_index"] != list(range(n)):
        raise ValueError("Display original acquisition indices disagree")
    if primary_bundle is not None:
        verify_primary_binding(metadata, primary_bundle)
    for source in metadata["sources"].values():
        for name, relative in source["volumes"].items():
            path = (directory / prefix / relative).resolve()
            key = str(path.relative_to(directory)) if path.is_relative_to(directory) else ""
            if not key.startswith(prefix) or key not in selected:
                raise ValueError("Display volume artifact is outside the selected scan")
            source["volumes"][name] = base64.b64encode(path.read_bytes()).decode("ascii")
    for region in metadata["regions"]:
        for source in ("tutorial", "archive"):
            signal = region[source]["series"]
            if signal is not None and len(signal) != n:
                raise ValueError("Display source series and acquisition frames disagree")
    provenance = json.loads((directory / "provenance.json").read_text())
    metadata["prepared_with"] = provenance["inputs"]["versions"]
    metadata["preparation_inputs"] = provenance["inputs"]
    metadata["generated_at_utc"] = provenance["generated_at_utc"]
    metadata["atlas"] = {k: provenance["atlas"][k] for k in ("name", "sha256", "symmetric_split")}
    return metadata


def render_explorer(
    directory, participant_id, initial_source="tutorial", initial_region=14, primary_bundle=None
):
    metadata = load_explorer(directory, participant_id, primary_bundle)
    if initial_source not in metadata["sources"] or initial_region not in [
        r["id"] for r in metadata["regions"]
    ]:
        raise ValueError("Unknown initial region or source")
    metadata["initial_source"], metadata["initial_region"] = initial_source, initial_region
    vendor = WEB / "vendor"
    provenance = json.loads((vendor / "provenance.json").read_text())
    for name in ("niivue.esm.js", "NiiVue-LICENSE.txt"):
        if digest(vendor / name) != provenance["files"][name]:
            raise ValueError(f"Viewer library artifact changed: {name}")
    html = (WEB / "index.html").read_text()
    css = "<style>" + (WEB / "explorer.css").read_text() + "</style>"
    scripts = (
        '<script id="coverage-data" type="application/json">' + script_json(metadata) + "</script>"
        '<script id="niivue-library" type="application/json">'
        + script_json((vendor / "niivue.esm.js").read_text())
        + "</script>"
        '<script id="explorer-model" type="application/json">'
        + script_json((WEB / "explorer-model.mjs").read_text())
        + "</script>"
        '<script type="module">' + (WEB / "explorer.js").read_text() + "</script>"
    )
    return html.replace('<link rel="stylesheet" href="explorer.css">', css).replace(
        '<script type="module" src="explorer.js"></script>', scripts
    )


def show_explorer(payload, key="coverage_explorer"):
    from streamlit.components.v1 import declare_component

    component = declare_component("fmri_coverage_explorer", path=str(WEB))
    return component(data=payload, key=key, default=None)
