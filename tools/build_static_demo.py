"""Export verified saved artifacts for GitHub Pages; never acquire or analyze data."""

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fmri_study_check.explorer import load_explorer  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def assert_public(value):
    text = json.dumps(value, allow_nan=False)
    if any(token in text for token in ("/Users/", "/home/", "file://", "Downloads/")):
        raise ValueError("Local filesystem information must not be published")


def verified_primary_file(bundle, artifacts, relative):
    path = bundle / relative
    expected = artifacts.get(relative)
    if not expected or not path.is_file() or sha(path) != expected:
        raise ValueError(f"Primary artifact disagrees with completed marker: {relative}")
    return path


def validate_destination(site, project_root):
    site, project_root = site.resolve(), project_root.resolve()
    protected = [
        project_root / name
        for name in ("data", "outputs", "investigations", ".cache", "fmri_study_check")
    ]
    destinations = [site] + [
        site / name for name in ("data", ".data-staging", "viewer", "report.html", ".nojekyll")
    ]
    for destination in destinations:
        destination = destination.resolve()
        for source in protected:
            source = source.resolve()
            if (
                destination == source
                or destination.is_relative_to(source)
                or source.is_relative_to(destination)
            ):
                raise ValueError(
                    "Unsafe publication destination overlaps scientific inputs or outputs"
                )
    for name in ("data", "viewer"):
        if any(p.is_symlink() for p in (site / name).rglob("*")):
            raise ValueError("Generated publication directories must not contain symlinks")
    return site


def build(project_root):
    project_root = project_root.resolve()
    site = validate_destination(REPO / "site", project_root)
    prepared = project_root / "data/derived/coverage_explorer"
    primary = project_root / "outputs/eight"
    primary_artifacts = json.loads((primary / "completed.json").read_text())["artifacts"]
    for name in (
        "manifest.csv",
        "retention.csv",
        "comparison.csv",
        "motion_summary.csv",
        "provenance.json",
    ):
        verified_primary_file(primary, primary_artifacts, name)
    summary_provenance = json.loads((REPO / "examples/provenance.json").read_text())
    for relative, expected in summary_provenance["source_artifact_hashes"].items():
        if relative.startswith("outputs/eight/"):
            name = relative.removeprefix("outputs/eight/")
            if sha(verified_primary_file(primary, primary_artifacts, name)) != expected:
                raise ValueError(f"Public summary provenance is stale: {name}")
    marker = json.loads((prepared / "completed.json").read_text())
    ids = marker["participant_ids"]
    if ids != [f"sub-pixar{i}" for i in range(123, 131)]:
        raise ValueError("The public demonstration requires the recorded eight-adult sample")
    summaries = {r["participant_id"]: r for r in read_csv(REPO / "examples/results_summary.csv")}
    if set(summaries) != set(ids):
        raise ValueError("Public summary participants disagree with prepared assets")
    comparison = {r["scan_key"]: r for r in read_csv(primary / "comparison.csv")}
    staged = site / ".data-staging"
    if staged.exists():
        raise ValueError("Incomplete export staging exists; inspect it before retrying")
    staged.mkdir(parents=True)
    participants = []
    try:
        for pid in ids:
            payload = load_explorer(prepared, pid, primary_bundle=primary)
            summary = summaries[pid]
            verified_summary = dict(
                participant_id=pid,
                scan_key=payload["scan_key"],
                processing_status=payload["primary_status"],
                n_volumes=payload["n_volumes"],
            )
            for field in ("n_fd_observed", "mean_fd_mm", "high_fd_count"):
                verified_summary[field] = payload["motion"][field]
            for condition in ("baseline", "censored"):
                row = next(r for r in payload["primary_analysis"] if r["condition"] == condition)
                for measure in ("retained_volumes", "retained_minutes"):
                    verified_summary[f"{condition}_{measure}"] = row[measure]
            for field in (
                "n_unique_edges",
                "mean_absolute_change",
                "max_absolute_change",
                "mean_signed_change",
                "edge_similarity",
            ):
                verified_summary[field] = comparison.get(payload["scan_key"], {}).get(field, "")
            for field, observed in summary.items():
                expected = verified_summary[field]
                if expected == "":
                    same = observed == ""
                else:
                    try:
                        same = math.isclose(
                            float(observed), float(expected), rel_tol=1e-12, abs_tol=1e-12
                        )
                    except (ValueError, TypeError):
                        same = observed == expected
                if not same:
                    raise ValueError(
                        f"Public summary disagrees with verified primary artifacts: {pid}, {field}"
                    )
            payload["initial_region"] = 14
            payload["initial_source"] = "tutorial"
            assert_public(payload)
            target = staged / f"{pid}.json"
            target.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
            entry = dict(
                participant_id=pid,
                file=f"data/{pid}.json",
                sha256=sha(target),
                bytes=target.stat().st_size,
                summary=verified_summary,
                comparison=comparison.get(payload["scan_key"]),
                exports={},
            )
            scan = primary / "scans" / payload["scan_key"]
            names = ["frames.csv", "regions.csv", "methods.json"]
            if all(r["status"] == "complete" for r in payload["primary_analysis"]):
                names += ["matrices.png", "correlation_baseline.csv", "correlation_censored.csv"]
            for name in names:
                relative = (scan / name).relative_to(primary).as_posix()
                content = verified_primary_file(primary, primary_artifacts, relative).read_bytes()
                if name.endswith((".csv", ".json")):
                    assert_public(content.decode())
                destination = staged / pid / name
                destination.parent.mkdir(exist_ok=True)
                destination.write_bytes(content)
                entry["exports"][name] = dict(
                    file=f"data/{pid}/{name}", sha256=sha(destination), primary_relative=relative
                )
            participants.append(entry)
        for name in (
            "results_summary.csv",
            "coverage_summary.csv",
            "hemisphere_coverage.csv",
            "source_hashes.csv",
            "provenance.json",
        ):
            shutil.copyfile(REPO / "examples" / name, staged / name)
        manifest = dict(
            schema_version=1,
            data_kind="saved_real_results_demo",
            participants=participants,
            dataset_citation=json.loads((staged / "provenance.json").read_text())[
                "dataset_citation"
            ],
            exporter="tools/build_static_demo.py",
            analysis_performed=False,
            source_display_manifest_sha256=sha(prepared / "completed.json"),
            primary_completed_marker_sha256=sha(primary / "completed.json"),
            primary_manifest_sha256=sha(primary / "manifest.csv"),
        )
        assert_public(manifest)
        (staged / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        destination = site / "data"
        if destination.exists():
            shutil.rmtree(destination)  # Only our fixed, generated publication directory.
        staged.rename(destination)
    except Exception:
        # Preserve failed staging for inspection; never touch the source artifacts.
        raise
    shutil.copytree(REPO / "fmri_study_check/web", site / "viewer", dirs_exist_ok=True)
    report = (REPO / "examples/report.html").read_text()
    report = report.replace(
        'href="../docs/METHODS.md"',
        'href="https://github.com/esanchezharris/fmri-quality-explorer/blob/main/docs/METHODS.md"',
    )
    for name in (
        "results_summary.csv",
        "coverage_summary.csv",
        "hemisphere_coverage.csv",
        "source_hashes.csv",
        "provenance.json",
    ):
        report = report.replace(f'href="{name}"', f'href="data/{name}"')
    (site / "report.html").write_text(report)
    (site / ".nojekyll").touch()
    print(
        json.dumps(
            dict(
                participants=len(ids),
                display_bytes=sum(p["bytes"] for p in participants),
                destination="site/",
                analysis_performed=False,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="Local project containing completed real display assets and primary outputs",
    )
    build(parser.parse_args().project_root)
