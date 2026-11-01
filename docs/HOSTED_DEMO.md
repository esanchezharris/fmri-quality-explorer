# Hosted interactive demo

Open **https://esanchezharris.github.io/fmri-quality-explorer/**. No installation,
account or upload is needed. The browser viewer supports all eight recorded participants,
48 parcels, both independent evidence layers, left/right counts, slice navigation,
3-D rotation, original-frame observations and CSV/JSON downloads. Analysis Comparison
shows the original saved matrices and retention; sub-pixar124 remains explicitly unavailable.

This is a static saved-results edition of the project. GitHub Pages serves HTML,
JavaScript and reviewed derived artifacts; it does not run Python or Streamlit.
The full analysis and Streamlit app are reproduced using the repository instructions.
Navigation never acquires source data or runs a scientific analysis.

## Build from verified local artifacts

After primary results and display preparation are complete, run from this repository:

```sh
uv run python tools/build_static_demo.py --project-root /path/to/completed/research-project
uv run pytest -q tests/test_static_demo.py
python -m http.server 8543 --bind 127.0.0.1 --directory site
```

The exporter verifies prepared hashes and binds each participant to the primary image,
frame, regional-series and retention artifacts through the existing read-only loader.
Every consumed primary table and copied figure/matrix is also checked against the
original completed-output marker before publication. It publishes only derived 3-D display volumes, saved regional observations, selected
CSV/PNG outputs and path-free metadata. It preserves unknown values and independent
lineages. It never downloads, extracts, cleans, changes exclusions or recalculates matrices.
Only the fixed `site/data/` publication directory is replaced; failed staging is retained
for inspection. The viewer under `site/viewer/` is copied byte-for-byte from the package.

Each participant payload is checked against a SHA-256 recorded in the site manifest
before the browser displays it. Payload promises are cached for repeat participant
navigation. Source files, full 4-D scans, private documents and local filesystem paths
are not published. This cache is separate from the original analysis cache.

The Pages workflow uploads the checked-in `site/` directory on changes to that directory.
It never accesses the local research workspace or credentials for acquiring data.
Deployment uses GitHub's Pages environment and pinned official actions.

## Animated preview

`docs/images/scan-rotation.gif` is a 12-second periodic orbit of a real saved mean BOLD
image with four Harvard–Oxford parcel selections. It is not an activation movie.
The GIF renderer and metadata record the source, frame cadence, loop closure and
display parameters. The independent SPM8 derivative already has 5-mm smoothing; a
2-mm normalized display grid is not a claim about native acquisition resolution.

The Harvard–Oxford atlas adaptations and animated preview retain CC BY-SA 4.0;
study and library attribution is included in the demo's Sources page. The locally
bundled NiiVue library retains its full BSD-2-Clause license.
