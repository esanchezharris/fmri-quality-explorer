# Methods and interpretation

## Sample and question

Use `fetch_development_fmri` with one adult and all confounds, then eight adults after the
one-person script, report and checks succeed. Record actual identifiers; selection follows
the fetcher rather than random sampling. The public dataset contains movie-watching data.
This project asks how one stated FD exclusion changes usable data and descriptive correlations.

## Source integrity

Join images and full confound TSVs on participant, task, and available session/acquisition/run/echo
entities; join phenotype by participant ID and confirm adulthood. Absent entities remain absent.
Read complete 4D image data and confirm finite values, valid dimensions/affine, timing and
image/confound alignment. Inspect raw TSV headers before pandas can rename duplicates.
Require six motion parameters, white-matter/CSF means, supplied cosine terms, and FD.
Validate numeric values, finiteness, unexpected missingness and binary non-steady-state flags.

Allow an initial undefined FD, DVARS, standardized DVARS, or derivative term only at original
row zero. Do not fill source values. Unexpected missingness in required inputs blocks analysis;
issues in unused columns are flagged for review. Retain exact evidence and locations.

## Initial frames and masks

Common mask: original index zero plus supplied `non_steady_state_outlier* == 1` frames are
excluded. Censored mask: additionally exclude FD strictly above 0.5 mm. Equality at 0.5 is
retained. Do not remove neighbors or prune segments. Do not use DVARS to select frames.

Even when the derivative stores a zero at row zero, the first between-frame difference has
no predecessor. Retain its stored value in exports but mark it `fd_defined=False` and omit
it from FD/DVARS descriptive denominators. All other finite FD values are summarized with
their observed counts. Report all-acquired, baseline-retained and censored-retained populations
separately. Common and FD exclusions form a union, never a double-counted sum.

Every row retains a zero-based original index. Acquisition-relative time is index × TR.
Acquired duration is N × TR, whereas the final plotted sample occurs at (N−1) × TR. Retained
minutes count retained volumes × TR / 60; this is aggregate usable duration, not a contiguous run.
No task-event model or slice-timing inference is performed.

## Atlas and nuisance regression

Use Harvard–Oxford `cort-maxprob-thr25-2mm` with `symmetric_split=False`. Resample discrete labels
to each image's voxel grid using nearest-neighbor interpolation. Map fitted masker `region_ids_`
through the atlas lookup table and sort by numeric ID; never infer labels from array position.
Missing atlas regions or duplicate IDs make the affected calculation unavailable.

Extract raw regional means with standardize=None, detrend=False, smoothing_fwhm=None,
high_pass=None, low_pass=None, confounds=None, sample_mask=None. Cache only this uncleaned extraction.

Select `load_confounds(strategy=['high_pass','motion','wm_csf'], motion='basic',
wm_csf='basic', demean=False)` and compare its returned columns and numerical values against
the source TSV. Verify its non-steady-state mask against the source indicators. Use the
existing cosine columns at their original timepoints; record their names without inventing
a source high-pass cutoff. No global-signal regression or CompCor is added.

For each condition, pass the full raw regional array, full nuisance array and sorted original
retained indices to `signal.clean`. Set detrend=False, standardize='zscore_sample',
standardize_confounds=True, filter=False, low_pass=None, high_pass=None, ensure_finite=False,
and actual recorded TR. Nilearn applies the same mask to signals and confounds before regression.
Nuisance fitting and final standardization occur separately for each condition. No filtering
is performed on compressed or concatenated time. Runs/participants are never concatenated.

## Guardrails and descriptive matrices

Require retained seconds ≥120 and nominal residual design df ≥20. Design df is retained rows
minus rank([intercept, retained nuisance columns]). Constant nuisance columns need review;
rank, rather than the nominal column count, determines this calculation. These project rules
do not certify reliability or account for temporal autocorrelation.

Reject non-finite, constant or near-zero-variance regional time series before and after
regression. Compute ordinary Pearson correlations. Validate finite square matrices, symmetry,
unit diagonals and bounds. Preserve numerical diagonals in CSVs; mask them only in figures.

Compare the unique upper-triangle off-diagonal edges: mean absolute change, maximum absolute
change, mean signed change, and edge-vector correlation when both vectors vary. An undefined
similarity is unavailable, not zero. Keep region identities fixed; baseline and censored scales
are [-1,1], differences [-2,2]. Do not use clustering to reorder conditions independently.

The official example's default `ConnectivityMeasure` uses shrinkage. This project explicitly
chooses ordinary Pearson for straightforward descriptive interpretation. Fixed methods are
not a claim that this nuisance strategy or censoring rule is optimal for this study.

## Limits

Short, spatially downsampled excerpts, a convenience sample, shared movie stimuli, unequal
retained observations and residual artifacts limit interpretation. Matrix changes do not
prove biological accuracy. There are no group tests, clinical claims or estimates of population effects.
