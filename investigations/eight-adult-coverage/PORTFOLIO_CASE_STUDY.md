# When a valid matrix hides incomplete brain coverage

**Project:** fMRI Study Check — reproducible QA of eight public adult movie-watching scans.

The initial workflow inventoried images and confounds, verified identifiers and frame alignment, and compared fixed nuisance cleaning with and without FD > 0.5-mm censoring. The first participant, sub-pixar123, produced both matrices. Across eight adults, seven paired comparisons were available; sub-pixar124's matrices were unavailable because a regional signal was constant.

That exposed a broader question: **does a computable regional average establish adequate spatial coverage?** A separate audit showed why the answer is no.

## Investigation and evidence

The audit preserved the original data and results and measured two evidence layers separately:

- **Distributed tutorial data:** voxelwise temporal variation on the existing 4-mm grid, plus direct comparison of source-derived regional means with saved raw exports.
- **Independent archived data:** mask overlap against every voxel in the original 2-mm Harvard–Oxford cortical atlas, including locations outside the image grid. The archive's historical “fmriprep” directory is actually documented as SPM8/MATLAB. It is not the matching 2019 fMRIPrep parent dataset.

For **sub-pixar123**, the independent archived mask covered 94.0% of the full cortical atlas overall, yet covered only **171/647 voxels (26.4%)** in region 14, Inferior Temporal Gyrus, anterior division, and **50/610 (8.2%)** in region 37, Temporal Fusiform Cortex, anterior division. In the tutorial source, only **20/76 region-37 voxel centers** varied across the run. All regional averages were still nonconstant, so numerical matrix checks alone did not reveal that sparse support.

For **sub-pixar124**, the independent mask included **0/647 region-14 atlas voxels**, and the tutorial parcel had **0/82 varying voxel centers**. A published reanalysis independently documented poor-coverage exclusions for both 123 and 124. That study used a third preprocessing lineage; its exclusion rule was not imported into the primary results.

The audit also found **152 undefined mask voxels in sub-pixar130**, all outside the cortical atlas parcels used here. Undefined locations remained unknown; the code records lower/upper overlap bounds if unknown values affect a parcel. It never fills the source mask.

## Reproducibility and communication

Python, NumPy, pandas, NiBabel, Nilearn and Matplotlib support sequential processing, identifier-based joins, explicit denominators, original-frame indices and inspectable calculations. Source hashes and content-based per-scan caches make reruns traceable. A separate HTML report, CSVs and a read-only Streamlit view expose the measured findings and uncertainty. The audit reconciled direct regional means with saved exports to within approximately **2.3 × 10⁻¹³** and preserved **654 existing protected files**.

The work adapts established Nilearn acquisition, atlas extraction and confound-cleaning methods. The project adds inventory checks, lineage tracing, coverage measurements, preservation evidence, failure handling and investigator-facing reporting. It does not propose a new neuroimaging estimator.

## Interpretation and limits

The evidence supports upstream coverage concerns, especially in inferior temporal parcels. It cannot isolate acquisition, masking, registration, downsampling and quantization contributions without the matching fMRIPrep parent BOLD, mask and preprocessing report. Mask overlap and temporal variation have different denominators and meanings; neither proves biological signal. The 80% mask and 50% variation bins are descriptive project review conventions, not validated exclusion thresholds. Primary results remain unchanged.

**Practical lesson:** validate what observations contribute to a result, not only whether the software can calculate it.

## Short interview explanation

“I built a reproducible workflow for eight public movie-watching fMRI scans. One scan had a constant regional signal, so I traced the source bytes and preprocessing provenance. I then audited every participant's spatial support separately from the connectivity results. The first participant had valid matrices but very limited coverage in two inferior temporal regions. I documented the actual counts, kept independent preprocessing lineages separate, preserved all original results, and exposed the findings in a report and Streamlit supplement. The remaining limitation is the unavailable matching preprocessing QA.”

## Evidence

- [Public saved findings](../../examples/README.md), [coverage counts](../../examples/coverage_summary.csv), [hemisphere counts](../../examples/hemisphere_coverage.csv), [settings and provenance](../../examples/provenance.json).
- [Archive SPM8 methods](https://s3.amazonaws.com/openneuro/ds000228/ds000228_R1.0.0/uncompressed/derivatives/README).
- [OSF tutorial methods](https://osf.io/wjtyq/).
- [Published coverage exclusions](https://pmc.ncbi.nlm.nih.gov/articles/PMC10635736/).

These are excerpts of the recorded investigation. Original local preservation snapshots and complete reports are retained privately; fresh-checkout snapshots are created by the bootstrap without resetting existing records.
