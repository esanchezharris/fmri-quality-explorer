# Saved real-data findings

These are path-free excerpts from the recorded eight-adult run. They preserve participant identities and scientific status; they are not a fresh analysis or an application-ready cache. Source artifacts and hashes are listed in [provenance.json](provenance.json).

| Participant | Baseline volumes | FD-censored volumes | Primary status |
| --- | ---: | ---: | --- |
| sub-pixar123 | 167 | 166 | complete |
| sub-pixar124 | 167 | 104 | analysis_unavailable |
| sub-pixar125 | 167 | 159 | complete |
| sub-pixar126 | 167 | 164 | complete |
| sub-pixar127 | 167 | 161 | complete |
| sub-pixar128 | 167 | 164 | complete |
| sub-pixar129 | 167 | 167 | complete |
| sub-pixar130 | 167 | 164 | complete |

The primary result bundle is partial: seven paired matrices and an explicitly unavailable result for sub-pixar124. Retention is still reported for unavailable matrix calculations. Each original scan has 168 volumes, a documented TR of 2 seconds, and 167 defined FD values. Both conditions omit frame zero; only FD strictly above 0.5 mm adds exclusions.

## Spatial support is a separate question

Sub-pixar123 had 50/610 included native-atlas centers in independent SPM8 mask region 37: 49/316 left and 1/294 right. Its tutorial image had 20/76 varying region-37 centers. These fractions have different meanings and denominators. Sub-pixar124 had 0/82 varying tutorial region-14 centers and 0/647 included independent-mask centers.

A changed matrix does not establish biological accuracy. Neither variation nor an independent mask certifies coverage in the unavailable matching fMRIPrep parent. Project coverage review bins are not validated inclusion criteria.

## Files

- [report.html](report.html): self-contained investigator-facing saved findings; download and open locally.
- [results_summary.csv](results_summary.csv): participant join, primary status, retained volumes/minutes, FD denominator and descriptive matrix-change measures.
- [coverage_summary.csv](coverage_summary.csv): separate tutorial-grid varying-center and full-native-atlas archived-mask counts for regions 14/37; mask non-finite count.
- [hemisphere_coverage.csv](hemisphere_coverage.csv): original hemisphere counts/bounds and evidence identities; blank mask fields in the tutorial layer remain unavailable.
- [source_hashes.csv](source_hashes.csv): original participant, source filenames and SHA-256 identities, with local paths omitted.
- [provenance.json](provenance.json): exact primary settings/versions and hashes of the source output CSVs from which these excerpts were made.

The original complete reports, source files, local filesystem paths and study caches are not included. Reproduce them with the documented acquisition/analysis commands.
