# Design rehearsal

Applied [Benji Taylor's Stageshow skill](https://github.com/benjitaylor/skills/blob/802057409787133703c56001080169e347305572/stageshow/SKILL.md),
including its craft, framing and rehearsal guidance, to the existing real-data product.
The skill's React illustration kit and faux-data examples are not used in this scientific
viewer. No new dependencies or copied third-party UI assets were introduced.

The focal object is the actual brain image, supported by region counts and motion.
Removed the decorative ring mark and generic slogan. Reduced the introduction's height,
used a plain text wordmark, quiet colors and sentence-case labels, and kept comparison
data in aligned rows. The mobile navigation fits all four views in two columns rather
than cutting off the Report label. The existing viewer has its own stacked compact
layout; it is not a scaled-down desktop screenshot.

The GIF has one action: a periodic camera orbit, with four actual atlas selections fading
in and out. No simulated activation, fake controls, decorative cursor or invented observation.
The 800 × 600 composition keeps the brain as its largest object and every parcel name in
frame. A separate 390px composition reuses the same rendered brain frames with captions
typeset at 12–20px and a two-column parcel key. A 240-frame contact sheet/decoded-frame
review verified the four parcels and seam.
The final renderer separates the canvas from the caption rules; the compact crop
retains the displayed brain through every angle. No divider removal or anatomical
pixel reconstruction is used.
The hidden closure render exactly matches frame zero; encoded seam change is consistent
with its neighbors. The README offers the still poster for reduced-motion preferences.
The live site has no automatic presentation animation; the viewer responds to user input.

Acceptance uses isolated headless Chrome at 1440px and 390px, actual WebGL and exports,
all eight participants/four views, plus a deliberately delayed parent script to exercise
startup ordering. Screenshot evidence and JSON results are kept separately from scientific
artifacts. The reusable source/result hashes are unchanged. Physical touch devices and
Safari are not claimed as checked. Stageshow's React-kit film gate is not claimed: this
is the real viewer with its existing renderer, reviewed using the applicable guidance.
