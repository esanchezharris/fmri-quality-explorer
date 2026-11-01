"""Acquire supplementary archive assets sequentially; never write primary data/results."""

import hashlib
import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
BUCKET = "https://s3.amazonaws.com/openneuro"
PREFIX = "ds000228/ds000228_R1.0.0/uncompressed/"
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_records(records):
    (OUT / "download_manifest.json").write_text(json.dumps(records, indent=2) + "\n")


def catalog(prefix):
    path = OUT / "catalog" / (prefix.rstrip("/").replace("/", "_") + ".xml")
    if not path.exists():
        response = requests.get(
            BUCKET, params={"prefix": PREFIX + prefix, "max-keys": 1000}, timeout=(20, 90)
        )
        response.raise_for_status()
        path.write_bytes(response.content)
    document = ET.fromstring(path.read_bytes())
    if document.findtext("s:IsTruncated", namespaces=NS) == "true":
        raise ValueError(f"Truncated source catalog: {prefix}")
    return {
        node.findtext("s:Key", namespaces=NS): {
            "etag": node.findtext("s:ETag", namespaces=NS).strip('"'),
            "bytes": int(node.findtext("s:Size", namespaces=NS)),
            "last_modified": node.findtext("s:LastModified", namespaces=NS),
        }
        for node in document.findall("s:Contents", NS)
    }


def obtain(participant, role, key, metadata):
    original_name = key.rsplit("/", 1)[-1]
    destination = OUT / "downloads" / participant / original_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Reuse prior verified investigation inputs without copying or altering them.
    reuse = {
        "native_bold": "sub-pixar124_raw_bold.nii.gz",
        "native_t1": "sub-pixar124_raw_T1w.nii.gz",
        "mriqc": "sub-pixar124_mriqc_bold.html",
    }
    previous = ROOT / "investigations/sub-pixar124-upstream/downloads"
    candidate = previous / reuse.get(role, original_name)
    if participant == "sub-pixar124" and candidate.is_file():
        destination = candidate
    url = BUCKET + "/" + key
    if not destination.exists():
        temporary = destination.with_name(destination.name + ".partial")
        for attempt in range(3):
            try:
                with requests.get(url, stream=True, timeout=(20, 120)) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as stream:
                        for block in response.iter_content(1024 * 1024):
                            stream.write(block)
                if temporary.stat().st_size != metadata["bytes"]:
                    raise ValueError(f"Downloaded size differs: {key}")
                if "-" not in metadata["etag"] and digest(temporary, "md5") != metadata["etag"]:
                    raise ValueError(f"Downloaded MD5 differs: {key}")
                temporary.replace(destination)
                break
            except (requests.RequestException, ValueError):
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
    if destination.stat().st_size != metadata["bytes"]:
        raise ValueError(f"Existing supplementary asset has wrong size: {destination}")
    if "-" not in metadata["etag"] and digest(destination, "md5") != metadata["etag"]:
        raise ValueError(f"Existing supplementary asset has wrong MD5: {destination}")
    return dict(
        participant_id=participant,
        role=role,
        path=str(destination.relative_to(ROOT)),
        source_filename=original_name,
        source_url=url,
        sha256=digest(destination),
        bytes=metadata["bytes"],
        published_etag=metadata["etag"],
        source_last_modified=metadata["last_modified"],
        verified_at_utc=datetime.now(timezone.utc).isoformat(),
        reused_read_only=destination.is_relative_to(previous),
        status="complete",
    )


def main():
    (OUT / "catalog").mkdir(exist_ok=True)
    manifest = pd.read_csv(ROOT / "outputs/eight/manifest.csv")
    participants = manifest.participant_id.tolist()
    if len(set(participants)) != 8 or set(participants) != {
        f"sub-pixar{i}" for i in range(123, 131)
    }:
        raise ValueError("Eight original participants must be retained exactly")
    records = []
    for participant in participants:
        specifications = [
            (
                "native_bold",
                f"{participant}/",
                f"{participant}/func/{participant}_task-pixar_bold.nii.gz",
            ),
            ("native_t1", f"{participant}/", f"{participant}/anat/{participant}_T1w.nii.gz"),
            (
                "spm8_bold",
                f"derivatives/fmriprep/{participant}/",
                f"derivatives/fmriprep/{participant}/{participant}_task-pixar_run-001_swrf_bold.nii.gz",
            ),
            (
                "spm8_mask",
                f"derivatives/fmriprep/{participant}/",
                f"derivatives/fmriprep/{participant}/{participant}_analysis_mask.nii.gz",
            ),
            (
                "spm8_anat",
                f"derivatives/fmriprep/{participant}/",
                f"derivatives/fmriprep/{participant}/{participant}_normed_anat.nii.gz",
            ),
            (
                "mriqc",
                f"derivatives/mriqc/reports/{participant}",
                f"derivatives/mriqc/reports/{participant}_task-pixar_bold.html",
            ),
        ]
        for role, prefix, relative in specifications:
            key = PREFIX + relative
            try:
                entries = catalog(prefix)
                if key not in entries:
                    raise ValueError(f"Requested asset absent from full source prefix: {key}")
                record = obtain(participant, role, key, entries[key])
                print(f"{participant} {role}: verified {record['bytes']:,} bytes", flush=True)
            except (requests.RequestException, ValueError, OSError) as exc:
                record = dict(
                    participant_id=participant,
                    role=role,
                    status="unavailable",
                    error=str(exc),
                    source_url=BUCKET + "/" + key,
                )
                print(f"{participant} {role}: {exc}", flush=True)
            records.append(record)
            save_records(records)
    failures = [r for r in records if r["status"] != "complete"]
    print(
        f"{len(records) - len(failures)}/{len(records)} supplementary assets verified", flush=True
    )
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
