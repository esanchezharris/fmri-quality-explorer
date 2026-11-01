"""Prepare a fresh checkout's separate coverage supplements; never reset preservation guards."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "investigations/eight-adult-coverage"
FOLLOWUP = AUDIT / "overnight-followups"
TREES = (
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
)
EXTRAS = ("app.py", "README.md", "pyproject.toml", "uv.lock", ".python-version")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_completed(directory):
    marker = directory / "completed.json"
    if not marker.is_file():
        return False
    data = json.loads(marker.read_text())
    if data.get("status") != "complete":
        raise ValueError(
            f"Existing supplement is incomplete; inspect its failures before retrying: {directory}"
        )
    for name, expected in data["artifacts"].items():
        path = (directory / name).resolve()
        if (
            not path.is_relative_to(directory.resolve())
            or not path.is_file()
            or sha256(path) != expected
        ):
            raise ValueError(f"Completed artifact is missing or changed: {name}")
    return True


def ensure_snapshot(root, directory, trees, extras):
    """Record fresh local paths once; an existing stale snapshot always blocks the run."""
    root, directory = root.resolve(), directory.resolve()
    path = directory / "protected-before.json"
    if not path.exists():
        files = {
            p.resolve()
            for tree in trees
            for p in Path(tree).rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and not p.resolve().is_relative_to(directory)
            and p.name not in (".DS_Store", "OVERNIGHT_HANDOFF.md")
        }
        files.update(p.resolve() for p in extras if p.is_file())
        if any(not p.is_relative_to(root) for p in files):
            raise ValueError("Protected input is outside the checkout")
        directory.mkdir(parents=True, exist_ok=True)
        snapshot = {str(p.relative_to(root)): sha256(p) for p in sorted(files)}
        with path.open("x") as stream:
            stream.write(json.dumps(snapshot, indent=2) + "\n")
    for name, expected in json.loads(path.read_text()).items():
        source = (root / name).resolve()
        if not source.is_relative_to(root) or not source.is_file() or sha256(source) != expected:
            raise ValueError(
                f"Protected input is missing or changed; snapshot was not reset: {name}"
            )


def run_script(relative):
    print(f"Running {relative}", flush=True)
    subprocess.run([sys.executable, str(ROOT / relative)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Acquire supplementary public data and run missing stages",
    )
    args = parser.parse_args()
    if not args.run:
        print(
            "Plan: verify primary one/eight bundles; acquire missing supplementary sources; run separate coverage and hemisphere audits; prepare display assets."
        )
        print(
            "Use --run after the primary stages. This may download about 1 GB; existing completed supplements are verified and reused."
        )
        return 0
    from fmri_study_check.pipeline import verify_bundle

    one = verify_bundle(ROOT / "outputs/one")
    eight = verify_bundle(ROOT / "outputs/eight")
    if one["comparisons_complete"] != 1 or eight["acquired_scans"] != 8:
        raise ValueError(
            "A successful one-participant checkpoint and all eight acquired adults are required"
        )
    trees, extras = [ROOT / p for p in TREES], [ROOT / p for p in EXTRAS]
    if not verify_completed(AUDIT):
        ensure_snapshot(ROOT, AUDIT, trees, extras)
        run_script("investigations/eight-adult-coverage/acquire.py")
        run_script("investigations/eight-adult-coverage/audit.py")
        if not verify_completed(AUDIT):
            raise ValueError("Coverage stage did not create a completed result")
    if not verify_completed(FOLLOWUP):
        ensure_snapshot(ROOT, FOLLOWUP, trees, extras)
        run_script("investigations/eight-adult-coverage/overnight-followups/hemisphere_audit.py")
        run_script(
            "investigations/eight-adult-coverage/overnight-followups/verify_independently.py"
        )
        if not verify_completed(FOLLOWUP):
            raise ValueError("Hemisphere stage did not create a completed result")
    run_script("fmri_study_check/explorer_prepare.py")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Supplement preparation stopped: {exc}", file=sys.stderr)
        raise SystemExit(1)
