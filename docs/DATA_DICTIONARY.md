# Export definitions

All study CSVs use `scan_key` for joins. A scan key uses participant/task and any supplied
session/acquisition/run/echo entities. Indices are zero-based original acquisition indices.

| File | Meaning and important denominators |
| --- | --- |
| `manifest.csv` | One row per acquired scan: identifiers, paths/filenames, SHA-256 hashes, dimensions, volumes, raw header zoom/units, resolved sampling interval and timing source, N×TR duration, confound rows/columns, selected nuisance columns, regions, processing status. Missing run/session stays empty. |
| `participants.csv` | Actual selected phenotype records, matched by participant ID. |
| `issues.csv` | One issue per rule/evidence item: error, review, unavailable, or info; stage; source location; column; original row indices; possible explanations; recommended action. |
| `motion_summary.csv` | Populations: `all_acquired`, `baseline`, `censored`. `n_volumes` is population size; `n_fd_observed` counts finite FD with original index >0. Mean/median/p95/max use those observations. `high_fd_fraction = high_fd_count / n_fd_observed`. |
| `frame_exclusions.csv` | One row per original frame: stored FD, `fd_defined`, optional DVARS, time, initial/NSS/high-FD flags, condition membership and all exclusion reasons. |
| `retention.csv` | Condition status/reason, retained and excluded volumes, acquired denominator, retained minutes/fraction, design rank and nominal residual df. Additional FD exclusions count only frames surviving the common policy. |
| `regions.csv` | Fitted numeric region IDs and names for each scan, verified against the atlas LUT. |
| `regional_timeseries.csv` | Raw and available cleaned regional means in `roi_<id>` columns; original index and acquisition-relative time are preserved. `condition=raw` is explicitly uncleaned. |
| `matrices.csv` | Full numerical Pearson matrices in long form, joined on scan, condition, `region_i`, `region_j`; diagonals remain 1. |
| `comparison.csv` | Unique off-diagonal edge count, mean absolute change, maximum absolute change, mean signed change and edge-vector similarity. Undefined similarity stays empty. Difference direction: censored minus baseline. |

Per-scan files in `scans/<scan_key>/` also contain selected confounds, regional labels,
raw/cleaned time series, square matrix CSVs, exact methods and the report figures.

`settings.json` records the fixed policy and every cleaning parameter. `provenance.json`
records selected participants, acquisition event, available dataset/atlas metadata, sources,
dependency versions and code digest. `source_inspection.json` records actual source headers
and phenotype records. Unknown derivative/preprocessing versions remain explicitly unknown.

`completed.json` contains hashes for a finished report bundle, including partial bundles;
`summary.json` distinguishes complete analyses from partial work. A finished partial bundle
is inspectable but does not authorize the one-to-eight expansion. `attempt.json` tracks the run.
Unexpected failures include `failure.json` with exact exception and traceback.

Files under `tests/fixtures/corrupted/` have `data_kind=fixture` and are never appended to
these real-data study tables. Their five-volume/four-row mismatch is deliberate.
