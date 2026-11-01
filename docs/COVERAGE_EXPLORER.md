# fMRI Quality Explorer

Open the **Coverage Explorer** view in the **fMRI Quality Explorer** app. Select any of the original eight
participants in the sidebar. The explorer links a real full-run mean BOLD image to atlas
selection, hemisphere support, uncleaned regional signals and the saved FD exclusions.

This is the current project name. Archived analysis reports retain their original
fMRI Study Check titles, hashes and provenance; renaming the app does not revise those results.

Use **Slices** to scroll through anatomy and click a parcel to select it. The initial
**Coronal** plane fills the panel; choose Axial, Sagittal or All slices with **Plane**.
**Enlarge** gives the image the full workspace width; **Zoom −/+** changes the view scale.
**Reset** restores scale and focus. **Left** and
**Right** move the crosshair to a real atlas center on that side. **3-D** supports rotation;
the region selector works in both modes. Amber marks the selected parcel. The optional cyan
overlay shows the selected source's support measure. The cursor plays through the original
168 frame positions; the mean brain image remains static. This is not an activation movie.

**Smooth 2-D BOLD display only** switches the background image between linear display
interpolation and its original nearest-neighbor voxel pattern. Categorical atlas/mask
overlays stay nearest-neighbor in 2-D. The option is disabled in 3-D, where NiiVue uses its
default volume-rendering interpolation. These appearance controls do not resample saved
images, change counts, alter regional signals or recalculate results. Evidence JSON records
the display choices separately from scientific settings.

The tutorial source has 4-mm voxels; the independent archive's 2-mm normalized grid has
already been smoothed by 5 mm. Making the panel bigger or interpolating its display cannot
recover finer anatomical detail. The resolution note is visible in the image panel. Sharp
structural anatomy would require a separate, appropriately identified anatomical image;
none is substituted for a participant's BOLD data here.

## Prepare and launch

After the existing eight-adult analysis and separate coverage/hemisphere supplements exist:

```sh
uv sync --locked
uv run python -B -m fmri_study_check.explorer_prepare
uv run streamlit run app.py --server.address 127.0.0.1 --server.headless true
```

Preparation reads preserved inputs, reconciles all 48 regional counts against the sealed
coverage audit, checks regions 14/37 against its hemisphere supplement, and checks raw
tutorial means against the saved primary series. It writes display assets only under
`data/derived/coverage_explorer/`. A repeat command validates and reuses content-based caches.
No cleaning or correlations run. Missing/mismatched inputs are specific failures, not
substitute observations. Old derived versions are kept under that directory's `history/`.

The UI reads completed assets, checks their hashes, and verifies the selected primary
scan/status/retention/frame and signal exports still agree. Stale display assets produce
an error and the preparation command. Switching source, region, hemisphere or time occurs
inside the browser component and does not rerun Python. Source contexts are cached within
the participant. There are no new Python dependencies, CDN calls or deployment requirements.

## Keep these evidence layers separate

| Context | Image and signal | Support numerator / denominator |
| --- | --- | --- |
| 4-mm tutorial | fMRIPrep derivative distributed as scaled int8 | Centers whose stored values change over the full run / parcel centers on the tutorial grid |
| 2-mm SPM8 archive | Independent SPM8 normalization and 5-mm smoothing | Known included mask centers / every center in the original 2-mm atlas parcel, including out-of-grid locations |

The matching tutorial fMRIPrep mask and exact parent QA remain unavailable. Tutorial
variation can reflect masking, interpolation and quantization; it is not a brain mask.
The SPM8 mask is never overlaid on the tutorial image. A 2-mm normalized grid is not a
claim of native 2-mm acquisition resolution. Coordinates use the supplied affines, without
new registration. Native atlas denominators and displayed image-grid parcels can therefore
differ. Hemispheres are strata of the original unsplit atlas, defined by supplied MNI x;
the atlas itself is unchanged. Midline counts appear when nonempty. Unknown mask membership
remains unknown; bounds are retained, and an empty stratum is unavailable rather than zero.

The charts use uncleaned means in each source's own intensity units, which are not comparable
across preprocessing lineages. FD and both condition masks always come from the preserved
tutorial analysis, including while inspecting the SPM8 series. Original time is index × TR;
the initial undefined FD is blank rather than zero, with 167 of 168 defined observations.
The FD condition adds only FD **strictly above 0.5 mm**. These views add no primary exclusions.

Sub-pixar124's constant region 14 and unavailable whole-matrix calculation remain visible
when other regions are selected. Sub-pixar123 has a computable matrix despite sparse
inferior temporal support. Sub-pixar125 is a comparison case, not a validated pass standard.
Neither mask overlap, variation nor computability establishes biological validity.

## Exports

- **Series CSV:** 168 original indices, acquisition seconds, the selected uncleaned source
  mean, defined tutorial FD, preserved stored FD, both saved memberships and exclusion reasons.
- **Evidence JSON:** selected region/lineage, left/right/midline/whole counts and bounds,
  BOLD and mask filenames/hashes (tutorial mask stays null), primary condition statuses,
  atlas identity, exact settings, dependency versions and preparation key/input digests.
- **Standalone HTML:** selected participant's complete explorer, including both contexts,
  the local library and its BSD license. It works as a local file with networking disabled.

The browser needs WebGL2 for the brain panel. If unavailable, a specific message replaces
the image while real measured counts and charts remain usable. No synthetic image is shown.

## Existing methods and added work

Brain rendering, slice navigation and volume picking use **NiiVue 0.69.0**, BSD-2-Clause,
vendored from its published npm archive. The archive's published SHA-512 integrity, local
file SHA-256 hashes, percent-decoding transformation and full license are recorded in
`fmri_study_check/web/vendor/provenance.json` and `NiiVue-LICENSE.txt`. No library code was
modified. Atlas display uses Nilearn nearest-neighbor resampling and NiBabel source scaling.

This project adds the validated display preparation, separate evidence contexts, exact
hemisphere/parcel counts, linked raw-signal and original-time exclusion audit, stale-result
guards, investigator explanations and traceable exports. Primary scientific methods are
documented in [METHODS.md](METHODS.md); no analysis estimator or censoring choice changed.

Verification evidence, screenshots and the read-only review are under
`output/coverage-explorer/`. Automated checks use labeled unit arrays only in tests;
the prepared explorer and browser acceptance cases use real participant observations.
