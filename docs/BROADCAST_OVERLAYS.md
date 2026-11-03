# Experimental Axelsen–Lee broadcast overlay

`broadcast_analysis.py` implements the complete overlay path on this source-bound
portrait clip. It reuses the existing RF-DETR/ByteTrack observations and the pinned
TrackNetV3 checkpoint. It is a development preview, not a validated match report.

The exported video includes:

- Temporary player IDs and camera-compensated movement trails.
- A court minimap and time-weighted occupancy heatmap, reset at edits and mapping gaps.
- Observed shuttle proposals and short trails, broken at missing observations.
- Proposed contact/shot hints, with paired names and question marks where ambiguous.
- Explicit pauses during close-ups, the complete source footage and original audio.

## Run

First create the player session using the command in
`validation/AXELSEN_LEE_TEST_2026-11-03.md`. Then run:

```powershell
.\.venv\Scripts\python.exe broadcast_analysis.py all --job out\axelsen-lee-session-001 --config configs\axelsen_lee_broadcast.json --out out\axelsen-lee-analysis-new
```

The output path must be fresh. Individual stages are `court`, `shuttle`, `relink`,
`events`, and `render`. Subsequent stages require the source/configuration/player
binding established by the court stage. `--reuse-shuttle PATH` reuses the model's
observations only when the inputs and saved artifact hashes match. Filtering and
rendering do not require repeating model inference.

Output: `Axelsen-Lee-All-Overlays.mp4`, per-frame court/shuttle JSONL, contact/shot
hypotheses, stage bindings and render verification. Media stays local and ignored.
The original session review is never changed; this separate experimental path
cannot grant approved player intervals or verified shuttle/coaching measurements.

## Court and view changes

Setup contains observed line intersections at native frames 0 and 1272. Some outer
corners are clipped, so these are multi-point fits of visible intersections rather
than invented corners. Six additional intersections at the first frame check the
fit independently of those fitting points. They are approximate automated source
inspection, not independent human calibration or physical accuracy measurements.

Optical flow on floor texture, with person regions excluded, estimates incremental
camera homographies. Projected-line consistency gates the resulting map. The
residual is a diagnostic, not a real-world error estimate. Features may drift;
other courts, colours, camera motion and recordings require separate evaluation.

Active-picture change and floor-flow support detect the middle close-up, return to
court, and closing close-up. A native-frame guard catches an edit between 15 FPS
court observations. A return view is enabled only by an explicit new setup; view
detection does not invent a replacement court calibration.

## Movement limitations

Projected positions use box-bottom centres. These can move with jumps, occlusion
and detector jitter. Trails are proposals; occupancy adds observed sample time
only. No distance in metres, ground-contact acceptance or athlete identity is
claimed. No path is joined across missing IDs, excluded views or scene changes.
The heatmap shows the current uninterrupted mapped segment, not match-wide effort.

## Shuttle limitations

Model input is the explicit flight-region crop `[0,420,1080,1630]`, resized to
512×288. The crop is clip-specific, anisotropic and not a generic portrait-camera
solution. Every position is restored to source pixels. Inference is 30 FPS on a
60 FPS source; the display holds the preceding sample for one inference interval.
No interpolation, inpainting or airborne-to-floor projection is performed.

Sequence windows/backgrounds remain inside eligible views, with bounded-memory
256-frame blocks. Raw candidates are preserved. An experimental body-core filter
withholds candidates on the central 60% of a person box between 25% and 90% of its
height. This reduces some clothing activations but can discard real shuttle
observations near bodies. Withheld does not mean absent. The filter does not change
the established TrackNet/production defaults.

## Shot and outcome limitations

Five consecutive linked observations, a direction change and nearby player boxes
produce contact cues. Uninterrupted outgoing image motion plus relative contact
height supplies low-confidence hints such as `Lift / clear?` or `Net / block?`.
Perspective, racket motion and occlusion prevent these rules from determining a
definitive shot class. These are not trained classifier outputs or verified racket
contacts. Ambiguous hitter proximity stays unknown. No complete shot inventory,
contact recall, shot accuracy, point winner, forced/unforced errors or coaching
scores is claimed. These remain data/model validation work.

## Validation

Run the Python regression suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
```

New tests cover camera-coordinate invariance, degenerate fits, static portrait
padding, discontinuous trails, missing outgoing shot evidence, explicit ambiguity,
body-core exclusion and modified cache rejection. The real-video record in
`validation/axelsen_lee_overlay_2026-11-03.json` separately records decoded frames,
stage hashes, observed coverage, sparse source checks and remaining gaps. Passing
software tests does not establish sporting accuracy.

## Shuttle refinement and enlarged maps (3 November)

The updated result is `out/axelsen-lee-analysis-008/index.html`. Open it directly,
or double-click `Start-BroadcastReview.cmd` for a seekable read-only local video
server. **Enlarge movement** and **Enlarge heatmap** open large synchronized maps.
The player selector filters both maps. Clicking a contact time seeks the video;
seeking backwards recomputes occupancy rather than keeping future positions.

`broadcast_refinement.py` adds a bounded development comparison. The model trials
use cutoff 0.35 with the original eight-frame stride or overlapping four-frame
stride. Existing observations are reused; an endpoint guide only selects actual
candidate positions on short gaps (at most six sampled frames). No synthetic
coordinate is emitted. Ambiguous paths, missing candidates, edits and unbracketed
gaps stay missing. Camera-stabilized motion rejects static ads. The initial loose
41-sample recovery trial was rejected after a source-patch audit found false
scoreboard/ad recoveries; that result is not the delivered video.

The selected run recovers 16 source-observed samples, and withholds 24 prior
detections inside the visible scoreboard. Net proposal count therefore falls from
614 to 606; more detections is not the acceptance goal. Source patches for all 16
recoveries were inspected automatically; they are not held-out precision/recall
labels. Long gaps, including the longest flight fragment, remain unresolved.
Possible slow apexes are retained unless background texture is also visible at
the corresponding camera-compensated location before and after the whole run.

Shot hypotheses now require a hitter-proximity margin and at least five outgoing
observations. Low rising cues can be labelled `Lift?`; overhead, descent and lateral
rules otherwise require their own evidence. The generic `Drive / other?` fallback
has been removed. The delivered clip contains five `Lift?` hypotheses and eight
unknown contacts. There is no measured classification-accuracy improvement or
trained shot model claim.

Reproduce the selected model trial and refinement, using fresh output directories:

```powershell
.\.venv\Scripts\python.exe broadcast_analysis.py court --job out\axelsen-lee-session-001 --config configs\axelsen_lee_broadcast_overlap_trial.json --out out\axelsen-lee-analysis-007
.\.venv\Scripts\python.exe broadcast_analysis.py shuttle --job out\axelsen-lee-session-001 --config configs\axelsen_lee_broadcast_overlap_trial.json --out out\axelsen-lee-analysis-007
.\.venv\Scripts\python.exe broadcast_refinement.py --baseline out\axelsen-lee-analysis-004 --trial out\axelsen-lee-analysis-007 --job out\axelsen-lee-session-001 --out out\axelsen-lee-analysis-008
```

Tests: 176 Python tests and five new JavaScript tests pass. Real browser checks
verified the enlarged movement/heatmap dialogs, player filtering and video/map
seeking at 2.90 seconds. The final video decodes 2,256 frames and retains audio.
See `validation/axelsen_lee_refinement_2026-11-03.json` for exact hashes, counts,
the rejected experiment and the unresolved validation work.
