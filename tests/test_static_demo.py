"""Offline acceptance of the shipped real-data demo, not new scientific findings."""

import hashlib
import json
from pathlib import Path

import pytest

from tools.build_static_demo import validate_destination, verified_primary_file

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


def test_saved_site_identities_hashes_frames_and_unavailable_result():
    manifest = json.loads((SITE / "data/manifest.json").read_text())
    assert manifest["data_kind"] == "saved_real_results_demo"
    assert manifest["analysis_performed"] is False
    assert [r["participant_id"] for r in manifest["participants"]] == [
        f"sub-pixar{i}" for i in range(123, 131)
    ]
    complete = 0
    for entry in manifest["participants"]:
        content = (SITE / entry["file"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == entry["sha256"]
        data = json.loads(content)
        assert data["participant_id"] == entry["participant_id"]
        assert data["motion"]["original_index"] == list(range(168))
        assert data["motion"]["fd"][0] is None
        assert sum(v is not None for v in data["motion"]["fd"]) == 167
        assert [r["id"] for r in data["regions"]] == list(range(1, 49))
        for row in data["primary_analysis"]:
            keep = data["motion"][f"{row['condition']}_keep"]
            assert not keep[0]
            assert row["retained_volumes"] == sum(keep)
            assert abs(row["retained_minutes"] - sum(keep) * 2 / 60) < 1e-12
        if data["participant_id"] == "sub-pixar124":
            assert entry["comparison"] is None
            assert all(r["status"] == "unavailable" for r in data["primary_analysis"])
            assert "matrices.png" not in entry["exports"]
            assert data["regions"][13]["tutorial"]["constant"]
        else:
            complete += 1
            assert entry["comparison"]["scan_key"] == data["scan_key"]
            assert "matrices.png" in entry["exports"]
        for artifact in entry["exports"].values():
            assert (
                hashlib.sha256((SITE / artifact["file"]).read_bytes()).hexdigest()
                == artifact["sha256"]
            )
        assert "/Users/" not in content.decode()
        assert "/home/" not in content.decode()
    assert complete == 7


def test_static_viewer_is_exact_copy_and_contains_no_source_or_private_documents():
    for original in (ROOT / "fmri_study_check/web").rglob("*"):
        if original.is_file():
            assert (
                original.read_bytes()
                == (
                    SITE / "viewer" / original.relative_to(ROOT / "fmri_study_check/web")
                ).read_bytes()
            )
    assert not [p for p in SITE.rglob("*") if p.suffix in (".nii", ".gz", ".docx", ".pdf", ".env")]
    report = (SITE / "report.html").read_text()
    assert "../docs/" not in report
    assert 'href="data/results_summary.csv"' in report
    assert "171/647" in report and "50/610" in report
    assert "corrupted" not in report.lower()


def test_export_refuses_changed_or_unrecorded_primary_artifacts(tmp_path):
    # Byte-only integrity fixture; it never represents an fMRI observation.
    path = tmp_path / "artifact.txt"
    path.write_text("recorded bytes")
    recorded = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}
    assert verified_primary_file(tmp_path, recorded, path.name) == path
    path.write_text("altered bytes")
    with pytest.raises(ValueError, match="completed marker"):
        verified_primary_file(tmp_path, recorded, path.name)
    with pytest.raises(ValueError, match="completed marker"):
        verified_primary_file(tmp_path, {}, path.name)


def test_export_destination_preserves_nested_sources_and_symlinks(tmp_path):
    site = tmp_path / "public/site"
    # The fixed destination must not contain the supplied research project.
    with pytest.raises(ValueError, match="overlaps scientific"):
        validate_destination(site, site / "data")
    project = tmp_path / "research"
    with pytest.raises(ValueError, match="overlaps scientific"):
        validate_destination(project / "outputs/site", project)
    (project / "data").mkdir(parents=True)
    alias = tmp_path / "aliased-site"
    alias.symlink_to(project / "data", target_is_directory=True)
    with pytest.raises(ValueError, match="overlaps scientific"):
        validate_destination(alias, project)
    assert validate_destination(site, project) == site.resolve()
    site.mkdir(parents=True)
    (site / "viewer").symlink_to(project / "data", target_is_directory=True)
    with pytest.raises(ValueError, match="overlaps scientific"):
        validate_destination(site, project)
