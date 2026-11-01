# Third-party notices

## NiiVue

The browser component includes `@niivue/niivue` 0.69.0 under the BSD 2-Clause License.
Its complete copyright, conditions and disclaimer are in
[NiiVue-LICENSE.txt](fmri_study_check/web/vendor/NiiVue-LICENSE.txt).
The published npm archive's SHA-512 integrity, local SHA-256 hashes and decoding step
are recorded in [provenance.json](fmri_study_check/web/vendor/provenance.json).
The library code was not modified. Standalone viewer exports include the license text.

## Scientific data and tools

Nilearn, NiBabel, NumPy, pandas, SciPy, Matplotlib, Jinja2 and Streamlit retain their
respective upstream licenses. Dependencies are resolved through `uv.lock`; their source
packages are not copied into this repository.

Public data and the Harvard–Oxford atlas are acquired from their attributed upstream
sources. Downloaded 4-D scans, masks, T1 images and preprocessing QA source files are not
bundled. The `examples/` files contain explicitly labeled excerpts of computed results
with source hashes, scientific limitations and attribution, and the screenshots show
the actual application using those real data.
