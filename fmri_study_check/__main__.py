import argparse
import sys
from pathlib import Path

from .pipeline import run_study, verify_bundle


def main():
    parser = argparse.ArgumentParser(
        description="fMRI Study Check — real-data, descriptive motion sensitivity"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Acquire, validate, analyze and report real data")
    run.add_argument("--n-subjects", type=int, choices=[1, 8], required=True)
    run.add_argument("--data-dir", type=Path, default=Path("data/source"))
    run.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument(
        "--checkpoint",
        type=Path,
        help="Verified one-participant bundle required before eight-adult acquisition",
    )
    verify = sub.add_parser("verify", help="Reconcile exports and verify immutable source hashes")
    verify.add_argument("bundle", type=Path)
    fixture = sub.add_parser(
        "fixture", help="Create a separately labeled 5-volume/4-row corrupted real excerpt"
    )
    fixture.add_argument("--source-bundle", type=Path, default=Path("outputs/one"))
    fixture.add_argument("--output-dir", type=Path, default=Path("tests/fixtures/corrupted"))
    args = parser.parse_args()
    if args.command == "fixture":
        import pandas as pd

        from .fixtures import create_corrupted_fixture

        verify_bundle(args.source_bundle)
        record = pd.read_csv(args.source_bundle / "manifest.csv").iloc[0].to_dict()
        result = create_corrupted_fixture(
            record, args.output_dir, record["sampling_interval_seconds"]
        )
        print(result)
        return 0
    if args.command == "verify":
        try:
            result = verify_bundle(args.bundle)
        except (ValueError, OSError, AssertionError) as exc:
            print(f"Verification failed: {exc}", file=sys.stderr)
            return 1
        print(result)
        return 0
    complete = run_study(
        args.n_subjects, args.data_dir, args.output_dir, args.cache_dir, checkpoint=args.checkpoint
    )
    print(
        f"{'Complete' if complete else 'Partial: inspect issues.csv and any recorded failure.json'} — {args.output_dir.resolve() / 'report.html'}"
    )
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
