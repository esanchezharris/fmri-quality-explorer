# Coverage supplement: scope, exports and reproduction

This is a separate descriptive audit of the original eight participants, sub-pixar123–130.
It never changes primary source images, confounds, matrices, frame exclusions, participant
selection or statuses. The primary one-person checkpoint and eight-adult bundles are historical
results and remain unchanged.

## Evidence layers

| Layer | What is measured | What it does not establish |
| --- | --- | --- |
| Existing 4-mm tutorial data | Number of parcel voxel centers with at least one stored value differing from the first acquired frame; uncleaned regional means | A brain mask, neural validity, signal-to-noise adequacy or coverage unaffected by int8 quantization/interpolation |
| Archived 2-mm SPM8 BOLD | The same stored-value variation checks on a finer normalized grid | The matching parent of the 2019 fMRIPrep tutorial derivative; native 2-mm acquisition resolution |
| Archived SPM8 analysis mask | Atlas voxel centers known included, excluded or unknown | A validated functional-coverage standard or the exact mask applied by fMRIPrep 1.3 |
| Native images and MRIQC 0.9.6 | Acquisition dimensions/spacing/TR, full native BOLD readability, original QA and header-coordinate FOV illustrations | The missing 2019 fMRIPrep registration/preprocessing HTML QA or a new native-to-MNI registration |

The historical archive folder named `derivatives/fmriprep/` is documented as SPM8/MATLAB,
with 5-mm smoothing and normalization to a 2-mm grid. This lineage is kept separate from
the tutorial's fMRIPrep 1.3.0.post2/masking/downsampling/int8 workflow and the published
SPM12 reanalysis. Atlas placement uses supplied world coordinates; normalization and
template differences remain an interpretive limitation.

## Denominators and missingness

- `grid_*`: denominator is the parcel after nearest-neighbor placement on that image's
  own grid. A tutorial parcel's 82 centers and a 2-mm atlas parcel's 647 centers are not
  interchangeable observations.
- `atlas_*`: denominator is **every voxel of the original 2-mm Harvard–Oxford parcel**,
  including atlas locations outside the rectangular image grid. Outside-grid locations
  have no image samples and therefore no functional support.
- `*_varying_voxels`: exact comparison of stored codes across all acquired frames.
  Finite nonzero NIfTI scaling is checked; exact constancy is preserved by that scaling.
  Temporal variation alone can include artifacts or interpolation effects.
- `*_mask_voxels`: known included locations only; `*_mask_unknown_voxels` counts NaNs.
  Finite mask values must be binary. Infinity, nonbinary finite values, or mismatched grids
  make that scan audit unavailable.
- `*_mask_fraction_lower = known_included / total`;
  `*_mask_fraction_upper = (known_included + unknown) / total`. Point fractions are
  unavailable when unknown locations affect that parcel. No missing mask voxel is filled.
- `archived_mask_cortical_atlas_fraction`: sum of known included cortical atlas locations
  divided by all cortical atlas locations, available only when no cortical atlas location
  has unknown mask membership. This whole-atlas average can hide severe regional gaps.

Sub-pixar130's archived mask contains 152 NaNs, none overlapping the cortical atlas parcels
used here. Their original voxel/world coordinates are exported. The source mask is unchanged.

Review bins of mask overlap below 0.8 and tutorial varying-center fraction below 0.5 are
**project conventions for finding records to inspect**, not validated adequacy thresholds.
No scan or region is automatically dropped. All acquired frames remain in this audit;
no nuisance cleaning, temporal filtering, censoring or connectivity computation is added.

## Files in `investigations/eight-adult-coverage/`

| File | Grain and use |
| --- | --- |
| `participant_summary.csv` | One row per original participant; primary status, whole-atlas counts, review counts and focused regions 14/37 |
| `region_coverage.csv` | Participant × evidence lineage × fixed region ID; 8 × 2 × 48 = 768 rows |
| `issues.csv` | One actionable review finding; evidence, file/region location, rule, possible explanations and recommended action |
| `download_manifest.json` | One public supplementary asset; participant/role, exact URL, local path, SHA-256, published ETag/MD5, byte count and timestamps |
| `settings.json`, `provenance.json` | Definitions, thresholds, dependencies, atlas, code/input hashes and explicit limitations |
| `scans/<participant>/source_roi_timeseries.csv` | Original acquisition index/time and uncleaned source means for regions 14/37 in each lineage; not re-cleaned series or replacement primary results |
| `scans/<participant>/mask_missing_locations.csv` | Original voxel/world coordinates of undefined archived mask values; empty for finite masks |
| `scans/<participant>/scan.json` | Image metadata, native timing, mask missingness and direct-source/primary-export reconciliation error |
| `scans/<participant>/completed.json` | Hash-keyed per-scan cache with derived-artifact digests |
| `report.html`, `REPORT.md`, `coverage_heatmap.png` | Standalone investigator findings and fixed-scale overview |
| `scans/<participant>/coverage.png` | Source BOLD slices and atlas/mask overlays; relative intensity per image |
| `scans/sub-pixar12[34]/native_fov.png` | Acquired slab shown on the native T1 using scanner headers only |
| `completed.json` | Completed supplement artifact hashes and status; the app requires this marker |
| `preservation.json`, `protected-before.json` | Hash evidence for 654 pre-existing protected files |
| `independent_verification.json` | Direct coordinate-indexing counts and visible HTML/CSV reconciliation |
| `UNSENT_SOURCE_REQUEST.md`, `PORTFOLIO_CASE_STUDY.md` | Unsent request and factual portfolio explanation |

The corruption fixture and labeled synthetic unit-test arrays do not enter any coverage
report, cache or export. Missing supplementary assets and scan failures are recorded in
`failures.json`; no participant is silently replaced. The initial plotting failures and
source-mask discovery are retained in `audit-first-attempt.log` and `attempts/`.

## Run and resume

```sh
uv run python investigations/eight-adult-coverage/acquire.py
uv run python investigations/eight-adult-coverage/audit.py
uv run python investigations/eight-adult-coverage/audit.py --verify-only
uv run python investigations/eight-adult-coverage/verify_independently.py
uv run pytest -q
uv run streamlit run app.py --server.address 127.0.0.1 --server.headless true
```

Acquisition is sequential, verifies public S3 ETag/MD5 and byte count, and records SHA-256.
Existing verified files are reused. All new downloads live under this supplement; verified
sub-pixar124 assets in the preceding investigation are reused without edits.

Analysis is sequential and keyed by source/atlas/settings/code hashes. Cached artifacts
must pass their own hash checks before reuse. A report rebuild never changes the primary
analysis cache. App navigation reads completed artifacts and cannot acquire or analyze data.

The independent check uses direct affine indexing and voxel min/max instead of the audit's
Nilearn resampling and metric helpers. It checks all 768 grid-region and 768 full-atlas
counts, special 123/124 findings, unknown-mask handling for 130, and the eight visible HTML
summary rows against the CSV exports. The standard primary-bundle `verify` commands remain
separate checks of the unchanged original workflow.
