#!/usr/bin/env python3
"""Render a looping, artifact-backed NiiVue illustration with headless Chrome.

Requires existing prepared real display assets, their original primary bundle,
Playwright, Pillow and Chrome. It does not acquire or analyze imaging data.
The animation rotates a full-run mean and fades actual atlas parcel selections;
it does not animate neural activity. Run from the published repository, e.g.:

    python tools/render_demo_gif.py --display /path/to/coverage_explorer \
        --primary /path/to/outputs/eight --work /path/to/private-render-work

Add --compact to reuse those saved frames in a separately typeset 390px image;
this performs no new browser capture or imaging analysis.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import http.server
import json
import math
import statistics
import sys
import threading
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageStat
from PIL.PngImagePlugin import PngInfo
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fmri_study_check.explorer import load_explorer, render_explorer  # noqa: E402

REGIONS = [1, 7, 21, 48]
WIDTH, HEIGHT = 800, 600
FRAME_COUNT, FRAME_MS = 240, 50
MEDIA_ATTRIBUTION = (
    "Adapted from Harvard-Oxford cortical atlas (Harvard Center for Morphometric Analysis / FSL): "
    "nearest-neighbor resampling to the saved image grid, binary parcel selections, "
    "3-D rendering, color, opacity and camera animation. GIF and poster: CC BY-SA 4.0 "
    "(https://creativecommons.org/licenses/by-sa/4.0/). Atlas license: "
    "https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html . Mean BOLD: Richardson et al. (2018), "
    "OpenNeuro ds000228, sub-pixar125 independent SPM8 archive. Viewer: NiiVue 0.69.0, BSD-2-Clause."
)

STAGE_CSS = """
html,body{margin:0!important;width:800px!important;height:600px!important;
 overflow:hidden!important;background:#102b3b!important;color:#e8f0f3;
 font-family:Arial,Helvetica,sans-serif}
.explorer{display:none!important}
#gif-stage{position:relative;width:800px;height:600px;background:#102b3b}
#gif-header{position:absolute;top:26px;left:32px;right:32px;z-index:2}
#gif-header h1{font-size:27px;letter-spacing:-.6px;line-height:1.2;margin:0 0 7px;font-weight:600}
#gif-header p{font-size:13px;color:#b0c3cd;margin:0}
#gif-canvas{position:absolute;top:70px;left:0;width:800px;height:398px}
#gif-canvas canvas{display:block;width:800px!important;height:398px!important;outline:none}
#gif-footer{position:absolute;left:32px;right:32px;bottom:25px}
#gif-parcels{display:flex;justify-content:space-between;gap:12px;margin-bottom:21px}
.gif-parcel{flex:1;font-size:13px;line-height:1.3;color:#b0c3cd;
 border-top:2px solid #35515f;padding-top:9px;white-space:nowrap}
.gif-parcel span{display:block;font-size:10px;text-transform:uppercase;
 letter-spacing:1px;margin-bottom:4px;opacity:.75}
#gif-caption{font-size:14px;line-height:1.3;margin:0;color:#e8f0f3}
#gif-credit{font-size:11px;color:#8da8b7;margin:7px 0 0}
"""

STAGE_JS = """() => {
 const canvas=document.getElementById('coverage-canvas');
 const stage=document.createElement('section'); stage.id='gif-stage';
 stage.innerHTML=`<header id="gif-header"><h1>Explore a real fMRI scan</h1>
 <p>sub-pixar125 · Independent SPM8 archive · 2-mm normalized grid</p></header>
 <div id="gif-canvas"></div><footer id="gif-footer"><div id="gif-parcels">
 <div class="gif-parcel" data-region="1"><span>Atlas region 1</span>Frontal pole</div>
 <div class="gif-parcel" data-region="7"><span>Atlas region 7</span>Precentral gyrus</div>
 <div class="gif-parcel" data-region="21"><span>Atlas region 21</span>Angular gyrus</div>
 <div class="gif-parcel" data-region="48"><span>Atlas region 48</span>Occipital pole</div>
 </div><p id="gif-caption">Rotating view of the average fMRI image</p>
 <p id="gif-credit">Atlas selections, not activation · Harvard–Oxford cortical atlas · ds000228 · NiiVue 0.69.0</p></footer>`;
 document.body.append(stage); document.getElementById('gif-canvas').append(canvas);
 const nv=window.coverageExplorer.viewer;
 nv.opts.isOrientCube=true; nv.opts.textHeight=.026;
 nv.scene.volScaleMultiplier=1.23;
 nv.resizeListener(); nv.drawScene();
 window.renderGifFrame=async (index,count) => {
   const phase=(index % count)/count;
   const position=phase*4, segment=Math.floor(position), local=position-segment;
   const id=[1,7,21,48][segment];
   const weight=Math.sin(Math.PI*local)**2;
   window.coverageExplorer.selectRegion(id,false);
   // The existing explorer generates this binary mask from its verified atlas.
   nv.volumes[3].opacity=.86*weight;
   nv.setRenderAzimuthElevation(120+360*phase,15+5*Math.sin(2*Math.PI*phase));
   nv.updateGLVolume(); nv.drawScene();
   for(const label of document.querySelectorAll('.gif-parcel')) {
     const active=Number(label.dataset.region)===id ? weight : 0;
     label.style.color=`rgb(${176+Math.round(44*active)},${195-Math.round(39*active)},${205-Math.round(153*active)})`;
     label.style.borderTopColor=`rgb(${53+Math.round(167*active)},${81+Math.round(75*active)},${95-Math.round(43*active)})`;
   }
   // Finish the GPU work, then allow layout, compositor and a second draw.
   await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
   nv.drawScene(); nv.gl.finish();
   return {index,phase,region:id,opacity:nv.volumes[3].opacity,
     selectedVoxels:nv.volumes[3].img.reduce((sum,value)=>sum+(value===1),0),
     meanNonzero:nv.volumes[0].img.some(value=>value!==0),
     azimuth:nv.scene.renderAzimuth,elevation:nv.scene.renderElevation};
 };
 return {volumes:nv.volumes.map(v=>({name:v.name,dimensions:Array.from(v.hdr.dims)})),
   renderer:nv.gl.getParameter(nv.gl.RENDERER)};
}"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_difference(a: Image.Image, b: Image.Image) -> float:
    return statistics.mean(
        ImageStat.Stat(ImageChops.difference(a.convert("RGB"), b.convert("RGB"))).mean
    )


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


@contextlib.contextmanager
def serve(directory: Path):
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(QuietHandler, directory=str(directory))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/viewer.html"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def encode(frames: list[Path], destination: Path, work: Path) -> dict:
    images = [Image.open(path).convert("RGB") for path in frames]
    # One shared palette for all frames. No independently quantized colors or
    # temporal dithering; subsampling only chooses the palette, not the frames.
    samples = images[::10]
    palette_sheet = Image.new("RGB", (200 * 4, 150 * math.ceil(len(samples) / 4)))
    for index, sample in enumerate(samples):
        palette_sheet.paste(sample.resize((200, 150)), ((index % 4) * 200, (index // 4) * 150))
    palette = palette_sheet.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in images]
    destination.parent.mkdir(parents=True, exist_ok=True)
    indexed[0].save(
        destination,
        save_all=True,
        append_images=indexed[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=False,
        disposal=1,
        comment=MEDIA_ATTRIBUTION.encode("utf-8"),
    )
    poster_metadata = PngInfo()
    poster_metadata.add_text("Attribution and license", MEDIA_ATTRIBUTION)
    images[30].save(destination.with_name("scan-rotation-poster.png"), pnginfo=poster_metadata)
    # Eight full-aspect thumbnails, kept outside the published repository.
    sheet = Image.new("RGB", (800, 300), "#102b3b")
    for position, index in enumerate(range(0, FRAME_COUNT, 30)):
        sheet.paste(images[index].resize((200, 150)), ((position % 4) * 200, (position // 4) * 150))
    sheet.save(work / "contact-sheet.png")
    closure = Image.open(work / "closure.png").convert("RGB")
    differences = [
        mean_difference(images[i], images[(i + 1) % len(images)]) for i in range(len(images))
    ]
    with Image.open(destination) as saved:
        durations = []
        unique = set()
        decoded_differences = []
        first_decoded = previous_decoded = None
        for index in range(saved.n_frames):
            saved.seek(index)
            durations.append(saved.info.get("duration"))
            decoded = saved.convert("RGB")
            unique.add(hashlib.sha256(decoded.tobytes()).hexdigest())
            if previous_decoded is not None:
                decoded_differences.append(mean_difference(previous_decoded, decoded))
            else:
                first_decoded = decoded
            previous_decoded = decoded
        decoded_differences.append(mean_difference(previous_decoded, first_decoded))
        peak_highlight_pixels = {}
        for region_id, index in zip(REGIONS, (30, 90, 150, 210), strict=True):
            saved.seek(index)
            pixels = saved.convert("RGB").crop((100, 100, 700, 460)).tobytes()
            peak_highlight_pixels[region_id] = sum(
                red > 115 and green > 90 and blue < red * 0.65
                for red, green, blue in zip(pixels[0::3], pixels[1::3], pixels[2::3], strict=True)
            )
        verification = dict(
            dimensions=list(saved.size),
            saved_frames=saved.n_frames,
            unique_decoded_frames=len(unique),
            duration_ms=sum(durations),
            frame_durations_ms=sorted(set(durations)),
            loop=saved.info.get("loop"),
            global_palette_colors=256,
            palette_dither="none",
            shared_input_palette=all(
                frame.getpalette() == indexed[0].getpalette() for frame in indexed
            ),
            decoded_peak_highlight_pixels=peak_highlight_pixels,
            closure_frame_rgb_mean_absolute_difference=mean_difference(images[0], closure),
            closure_frame_pixel_identical=ImageChops.difference(images[0], closure).getbbox()
            is None,
            adjacent_frame_mean_absolute_difference=dict(
                minimum=min(differences),
                median=statistics.median(differences),
                maximum=max(differences),
                seam=differences[-1],
                first_step=differences[0],
                before_seam=differences[-2],
            ),
            decoded_adjacent_frame_mean_absolute_difference=dict(
                minimum=min(decoded_differences),
                median=statistics.median(decoded_differences),
                maximum=max(decoded_differences),
                seam=decoded_differences[-1],
                first_step=decoded_differences[0],
                before_seam=decoded_differences[-2],
            ),
        )
    if verification["saved_frames"] != FRAME_COUNT or len(unique) != FRAME_COUNT:
        raise RuntimeError("Frame loss or repeated frames in encoded animation")
    if not verification["closure_frame_pixel_identical"]:
        raise RuntimeError("Hidden closure render differs from first render")
    if not all(pixels > 1000 for pixels in peak_highlight_pixels.values()):
        raise RuntimeError("A selected atlas region is not visible in the encoded GIF")
    return verification


def render_compact(work: Path, destination: Path) -> None:
    """Recompose existing real render pixels for a readable phone-width image."""
    if destination.name != "scan-rotation-compact.gif":
        raise ValueError("Compact output must be named scan-rotation-compact.gif")
    evidence = work / "compact"
    evidence.mkdir(parents=True, exist_ok=True)
    frame_paths = [work / "frames" / f"{index:04d}.png" for index in range(FRAME_COUNT)]
    full_gif = ROOT / "docs/images/scan-rotation.gif"
    full_manifest_path = full_gif.with_suffix(".json")
    full_manifest = json.loads(full_manifest_path.read_text())
    full_before = sha256(full_gif)
    if full_before != full_manifest["output_sha256"]:
        raise ValueError("Original GIF no longer matches its recorded provenance")
    # The unchanged header overlays only blank space at canvas top70–85.
    # Begin below that UI text, retaining all displayed brain pixels below126.
    source_crop = (80, 86, 720, 468)
    compact_size, brain_size, brain_top = (390, 534), (390, 233), 92
    background, neutral, secondary = "#102b3b", "#e8f0f3", "#b0c3cd"
    font_dir = Path("/System/Library/Fonts/Supplemental")
    fonts = {size: ImageFont.truetype(str(font_dir / "Arial.ttf"), size) for size in (12, 13)}
    title_font = ImageFont.truetype(str(font_dir / "Arial Bold.ttf"), 20)
    caption_font = ImageFont.truetype(str(font_dir / "Arial Bold.ttf"), 14)
    labels = [
        (1, "Frontal Pole"),
        (7, "Precentral Gyrus"),
        (21, "Angular Gyrus"),
        (48, "Occipital Pole"),
    ]

    def compose(source_image: Image.Image, index: int) -> Image.Image:
        image = Image.new("RGB", compact_size, background)
        draw = ImageDraw.Draw(image)
        draw.text((18, 18), "Explore a real fMRI scan", font=title_font, fill=neutral)
        draw.text(
            (18, 52), "sub-pixar125 · Independent SPM8 archive", font=fonts[12], fill=secondary
        )
        draw.text((18, 70), "2-mm normalized grid · full-run mean", font=fonts[12], fill=secondary)
        position = (index % FRAME_COUNT) / FRAME_COUNT * len(REGIONS)
        segment = math.floor(position)
        weight = math.sin(math.pi * (position - segment)) ** 2
        brain = source_image.crop(source_crop)
        brain = brain.resize(brain_size, Image.Resampling.LANCZOS)
        image.paste(brain, (0, brain_top))
        draw.text((18, 351), "Atlas selections", font=fonts[12], fill=secondary)
        for label_index, (region_id, label) in enumerate(labels):
            active = weight if region_id == REGIONS[segment] else 0
            color = (176 + round(44 * active), 195 - round(39 * active), 205 - round(153 * active))
            x, y = 18 + (label_index % 2) * 181, 374 + (label_index // 2) * 23
            draw.text((x, y), f"{region_id}  {label}", font=fonts[12], fill=color)
        draw.text(
            (18, 434), "Rotating view of the average fMRI image", font=caption_font, fill=neutral
        )
        draw.text((18, 456), "Atlas selections, not activation", font=fonts[13], fill=neutral)
        draw.text((18, 493), "Harvard–Oxford · ds000228", font=fonts[12], fill=secondary)
        draw.text((18, 511), "NiiVue 0.69.0 · CC BY-SA 4.0", font=fonts[12], fill=secondary)
        return image

    images, source_bounds = [], []
    for index, path in enumerate(frame_paths):
        with Image.open(path) as source_image:
            source_image = source_image.convert("RGB")
            # Bound visible source marks without segmenting anatomy or making
            # a scientific brain-coverage judgment.
            mask = (
                source_image.crop(source_crop).getchannel("R").point(lambda r: 255 if r > 45 else 0)
            )
            bounds = mask.getbbox()
            if bounds is None or bounds[0] <= 0 or bounds[2] >= 640 or bounds[1] <= 0:
                raise ValueError(f"Central crop clips visible source pixels in frame {index}")
            source_bounds.append([bounds[0] + 80, bounds[1] + 86, bounds[2] + 80, bounds[3] + 86])
            images.append(compose(source_image, index))
    with Image.open(work / "closure.png") as source_closure:
        closure = compose(source_closure.convert("RGB"), FRAME_COUNT)
    if ImageChops.difference(images[0], closure).getbbox() is not None:
        raise RuntimeError("Compact closure is not pixel-identical to frame zero")
    closure.save(evidence / "closure.png")
    palette_sheet = Image.new("RGB", (390 * 4, 534 * 6), background)
    for index, image in enumerate(images[::10]):
        palette_sheet.paste(image, ((index % 4) * 390, (index // 4) * 534))
    # Reserve neutral shades so quantization cannot add colored patches to the
    # mean BOLD. Adaptive colors retain the source amber and background blends.
    adaptive_palette = palette_sheet.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    palette = Image.new("P", (1, 1))
    gray_ramp = [value for level in range(128) for value in (level * 2,) * 3]
    palette.putpalette(gray_ramp + adaptive_palette.getpalette()[:384])
    indexed = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
    destination.parent.mkdir(parents=True, exist_ok=True)
    indexed[0].save(
        destination,
        save_all=True,
        append_images=indexed[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=False,
        disposal=1,
        comment=MEDIA_ATTRIBUTION.encode("utf-8"),
    )
    poster_metadata = PngInfo()
    poster_metadata.add_text("Attribution and license", MEDIA_ATTRIBUTION)
    images[30].save(destination.with_suffix(".png"), pnginfo=poster_metadata)
    decoded_frames, durations = [], []
    with Image.open(destination) as saved:
        loop = saved.info["loop"]
        for index in range(saved.n_frames):
            saved.seek(index)
            durations.append(saved.info["duration"])
            decoded_frames.append(saved.convert("RGB"))
    unique = len({hashlib.sha256(image.tobytes()).hexdigest() for image in decoded_frames})
    differences = [
        mean_difference(image, decoded_frames[(index + 1) % FRAME_COUNT])
        for index, image in enumerate(decoded_frames)
    ]
    if len(decoded_frames) != FRAME_COUNT or unique != FRAME_COUNT or set(durations) != {FRAME_MS}:
        raise RuntimeError("Compact GIF lost frames, timing, or unique motion")
    peaks = {}
    for region_id, index in zip(REGIONS, (30, 90, 150, 210), strict=True):
        pixels = (
            decoded_frames[index].crop((0, brain_top, 390, brain_top + brain_size[1])).tobytes()
        )
        peaks[region_id] = sum(
            red > 115 and green > 90 and blue < red * 0.65
            for red, green, blue in zip(pixels[0::3], pixels[1::3], pixels[2::3], strict=True)
        )
    if not all(count > 1000 for count in peaks.values()):
        raise RuntimeError("A compact atlas highlight is not visible")
    contact = Image.new("RGB", (390 * 4, 534), background)
    for position, index in enumerate((30, 90, 150, 210)):
        contact.paste(decoded_frames[index], (position * 390, 0))
    contact.save(evidence / "decoded-contact-sheet.png")
    if sha256(full_gif) != full_before:
        raise RuntimeError("The full-sized GIF changed during compact export")
    validation = dict(
        dimensions=list(compact_size),
        frames=len(decoded_frames),
        unique_frames=unique,
        frame_duration_ms=FRAME_MS,
        total_duration_ms=sum(durations),
        loop=loop,
        shared_palette=all(frame.getpalette() == indexed[0].getpalette() for frame in indexed),
        palette_colors=256,
        reserved_neutral_shades=128,
        dithering="none",
        closure_pixel_identical=True,
        decoded_highlight_pixels=peaks,
        all_angles_no_added_crop_clipping=True,
        canvas_to_labels_gap_pixels=351 - (brain_top + brain_size[1]),
        source_pixel_bounds=[
            min(b[0] for b in source_bounds),
            min(b[1] for b in source_bounds),
            max(b[2] for b in source_bounds),
            max(b[3] for b in source_bounds),
        ],
        decoded_transition_mae=dict(
            seam=differences[-1],
            before_seam=differences[-2],
            first_step=differences[0],
            median=statistics.median(differences),
            maximum=max(differences),
        ),
    )
    manifest = dict(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        artifact=destination.name,
        caption="Rotating view of the average fMRI image",
        output_bytes=destination.stat().st_size,
        output_sha256=sha256(destination),
        source_full_gif_sha256=full_before,
        source_full_manifest_sha256=sha256(full_manifest_path),
        source_frame_sha256={path.name: sha256(path) for path in frame_paths},
        crop_source_pixels=list(source_crop),
        resized_brain_pixels=list(brain_size),
        source_canvas_pixels=[0, 70, 800, 468],
        source_footer_rule_y=472,
        inherited_UI_cleanup="None; newly captured canvas ends above the separate footer rule",
        minimum_text_size_pixels=12,
        minimum_at_358_pixel_display=12 * 358 / 390,
        typography="Arial; 20px heading, 12px source/parcel/credit, 14px caption, 13px qualification",
        design_guidance="https://github.com/benjitaylor/skills/tree/802057409787133703c56001080169e347305572/stageshow",
        compact_design="Separate composition; original brain canvas pixels cropped and resized, header/footer typeset anew; no new brain rendering or analysis.",
        scientific_data_change=False,
        source={
            key: full_manifest[key]
            for key in (
                "participant",
                "source",
                "source_filename",
                "source_sha256",
                "preprocessing",
                "grid_mm",
                "atlas",
                "regions",
                "media_license",
                "representation",
            )
        },
        validation=validation,
    )
    destination.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            dict(output=str(destination), bytes=destination.stat().st_size, validation=validation),
            indent=2,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--display", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/images/scan-rotation.gif")
    parser.add_argument(
        "--chrome", default="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    )
    parser.add_argument("--preview", action="store_true", help="Render eight test frames only")
    parser.add_argument("--encode-only", action="store_true", help="Encode already rendered frames")
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Recompose existing frames for a 390px phone-width GIF; no recapture",
    )
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    if args.compact:
        destination = args.output
        if destination == ROOT / "docs/images/scan-rotation.gif":
            destination = ROOT / "docs/images/scan-rotation-compact.gif"
        render_compact(args.work, destination)
        return
    frame_dir = args.work / "frames"
    frame_dir.mkdir(exist_ok=True)
    participant, source = "sub-pixar125", "archive"
    metadata = load_explorer(args.display, participant, primary_bundle=args.primary)
    raw_metadata_path = args.display / "scans" / participant / "metadata.json"
    raw_metadata = json.loads(raw_metadata_path.read_text())
    source_dir = args.display / "scans" / participant
    frame_paths = [frame_dir / f"{index:04d}.png" for index in range(FRAME_COUNT)]
    if not args.encode_only:
        html = render_explorer(
            args.display,
            participant,
            initial_source=source,
            initial_region=1,
            primary_bundle=args.primary,
        )
        (args.work / "viewer.html").write_text(html)
        errors, states = [], []
        with serve(args.work) as url, sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                executable_path=args.chrome,
                args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
            )
            page = browser.new_page(
                viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=1
            )
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(url, wait_until="load")
            page.wait_for_function("document.body.dataset.viewerStatus==='ready'", timeout=90000)
            page.locator("#view-3d").click()
            page.add_style_tag(content=STAGE_CSS)
            runtime = page.evaluate(STAGE_JS)
            runtime["framing"] = page.evaluate("""() => {
                const bounds = id => {
                    const r=document.getElementById(id).getBoundingClientRect();
                    return {left:r.left,top:r.top,right:r.right,bottom:r.bottom};
                };
                return {header:bounds('gif-header'),canvas:bounds('gif-canvas'),
                    parcels:bounds('gif-parcels'),footer:bounds('gif-footer')};
            }""")
            if runtime["framing"]["canvas"]["bottom"] >= runtime["framing"]["parcels"]["top"]:
                raise RuntimeError("Brain canvas overlaps the footer")
            page.evaluate("document.fonts.ready")
            indices = list(range(0, FRAME_COUNT, 30)) if args.preview else list(range(FRAME_COUNT))
            for index in indices + [FRAME_COUNT]:
                state = page.evaluate(
                    "async ([index,count])=>await window.renderGifFrame(index,count)",
                    [index, FRAME_COUNT],
                )
                path = args.work / "closure.png" if index == FRAME_COUNT else frame_paths[index]
                page.screenshot(path=str(path), animations="disabled")
                states.append(state)
                if index % 20 == 0 or args.preview:
                    print(
                        f"Rendered {index}/{FRAME_COUNT}; region {state['region']}; "
                        f"{state['selectedVoxels']} actual atlas voxels",
                        flush=True,
                    )
            browser.close()
        if errors or not all(state["selectedVoxels"] and state["meanNonzero"] for state in states):
            raise RuntimeError(f"Invalid render: {errors}")
        (args.work / "render-state.json").write_text(
            json.dumps(dict(runtime=runtime, frames=states), indent=2) + "\n"
        )
        if args.preview:
            print("Preview frames ready", flush=True)
            return
    validation = encode(frame_paths, args.output, args.work)
    lineage = raw_metadata["sources"][source]
    manifest = dict(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        artifact="scan-rotation.gif",
        caption="Rotating view of the average fMRI image",
        participant=participant,
        source=source,
        source_filename=lineage["source_filename"],
        source_sha256=lineage["source_sha256"],
        mask_source_sha256=lineage["mask_source_sha256"],
        preprocessing=lineage["preprocessing"],
        grid_mm=lineage["grid_mm"],
        mean_intensity_window=lineage["window"],
        metadata_sha256=sha256(raw_metadata_path),
        display_asset_hashes={
            key: sha256(source_dir / path) for key, path in lineage["volumes"].items()
        },
        display_asset_key=metadata["asset_key"],
        primary_binding="verified by load_explorer",
        framing=json.loads((args.work / "render-state.json").read_text())["runtime"].get("framing"),
        atlas=metadata["atlas"],
        media_license={
            "identifier": "CC-BY-SA-4.0",
            "url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "scope": "GIF and poster",
            "atlas_license_source": "https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html",
            "attribution_and_adaptations": MEDIA_ATTRIBUTION,
        },
        regions=[
            {"id": region["id"], "name": region["name"]}
            for region in metadata["regions"]
            if region["id"] in REGIONS
        ],
        viewer=json.loads((ROOT / "fmri_study_check/web/vendor/provenance.json").read_text()),
        representation="Full-run mean BOLD, rotating camera; binary atlas selections fade. Not activation.",
        orientation="Uses saved supplied coordinates and existing NiiVue convention; no reorientation or registration.",
        cyclic_parameterization="phase=(index mod 240)/240; azimuth=120+360*phase; elevation=15+5*sin(2*pi*phase); parcel opacity=.86*sin(pi*local_phase)^2",
        output_sha256=sha256(args.output),
        output_bytes=args.output.stat().st_size,
        validation=validation,
    )
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "bytes": args.output.stat().st_size,
                "validation": validation,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
