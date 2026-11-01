"""Seal the standalone follow-up only after fresh independent evidence passes."""

import hashlib
import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_proof(directory):
    directory = Path(directory).resolve()
    proof = json.loads((directory / "independent_verification.json").read_text())
    if proof["status"] != "passed":
        raise ValueError("Independent verification did not pass")
    for relative, expected in proof["input_artifacts"].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or digest(path) != expected:
            raise ValueError(f"Independent proof is stale: {relative}")
    return proof


def main():
    from hemisphere_audit import preservation

    proof = verify_proof(OUT)
    marker = json.loads((OUT / "completed.json").read_text())
    for relative, expected in marker["artifacts"].items():
        if digest(OUT / relative) != expected:
            raise ValueError(f"Analysis artifact changed before finalization: {relative}")
    metadata = json.loads((OUT / "provenance.json").read_text())
    if metadata["code_sha256"] != digest(OUT / "hemisphere_audit.py"):
        raise ValueError("Analysis code changed after the recorded run")
    protection = preservation()
    tests = (OUT / "tests.log").read_text()
    if not re.search(r"6 passed", tests):
        raise ValueError("Six targeted checks have not passed")
    if (OUT / "lint.log").read_text().strip() != "All checks passed!":
        raise ValueError("Lint has not passed")
    extras = [
        "APP_WALKTHROUGH.md",
        "verify_independently.py",
        "independent_verification.json",
        "test_hemisphere.py",
        "tests.log",
        "lint.log",
        "finalize.py",
        "PLAN.md",
        "REVIEW.md",
        "README.md",
    ]
    for name in extras:
        marker["artifacts"][name] = digest(OUT / name)
    marker["verification"] = dict(
        independent_status=proof["status"],
        targeted_tests=6,
        hemisphere_rows=proof["hemisphere_rows_checked"],
        preserved_files=protection["protected_files"],
        sealed_coverage_artifacts=protection["sealed_audit_artifacts"],
    )
    (OUT / "completed.json").write_text(json.dumps(marker, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                status="finalized",
                artifacts=len(marker["artifacts"]),
                verification=marker["verification"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
