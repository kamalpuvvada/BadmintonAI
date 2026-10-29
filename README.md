# Badminton Academy AI — Technical Prototype

This is a growing **offline-first, fixed-camera badminton coaching platform**. Step 1 establishes repeatable video ingestion and court calibration. Step 2 adds a separately installed person detection and ID-tracking preview. We deliberately do **not** invent shuttle speeds, shot types, or coaching metrics before validation.

## Why start here?

Every court must have a stable camera, known video characteristics, and a calibrated floor plane. If these are wrong, later movement heatmaps and footwork reports will look precise but be misleading.

This prototype can:

1. Inspect a session video (frame rate, resolution, length, video hash, audio presence).
2. Extract nine representative review frames and a contact sheet.
3. Flag capture limitations relevant to badminton tracking.
4. Create a structured `session.json` that the future .NET API can consume.
5. Fit a 2D court calibration from four *visible* doubles-court outer corners and export a preview and homography.

Step 1 does **not** detect players or shuttles. Step 2 introduces experimental person boxes and track IDs, but they must pass a manual accuracy review before any performance scores are derived.

## Prerequisites

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) with `ffprobe`, both accessible on `PATH`
- `pip install -r requirements.txt`

## Step 1A — Analyze a video

From the folder containing `badminton_proto.py`:

```bash
python badminton_proto.py inspect --video "path/to/match.mp4" --out "out/session-001" --camera "court-01"
```

For the supplied example clip, which already contains generated labels and graphics:

```bash
python badminton_proto.py inspect --video "path/to/189733.mp4" --out "out/demo" --camera "demo-court" --processed-overlay
```

Inspect these files:

- `out/session-001/session.json`: video facts, capture warnings, deliberately empty analytics
- `out/session-001/review_contact_sheet.jpg`: visual review of 9 frames
- `out/session-001/frames`: original extracted review frames (scaled to 360 px width)

The input video is **never copied** to the output folder.

## Step 1B — Calibrate a physical court (when all four corners are visible)

Take a full-resolution frame from the **raw fixed camera**. Measure pixel coordinates for the four *outer doubles* court corners in this exact order:

1. `far_left`: far-side baseline at left doubles sideline
2. `far_right`: far-side baseline at right doubles sideline
3. `near_right`: near-side baseline at right doubles sideline
4. `near_left`: near-side baseline at left doubles sideline

Save a JSON file named `corners.json`:

```json
{
  "far_left": [100, 200],
  "far_right": [500, 200],
  "near_right": [590, 900],
  "near_left": [20, 900]
}
```

These numbers are a **fictional example**. You must replace them with actual pixel positions from the camera image.

Then run:

```bash
python badminton_proto.py calibrate --frame "path/to/full_court_frame.jpg" --corners "corners.json" --out "out/court-01" --camera "court-01"
```

This exports `calibration.json` and `court_calibration_preview.jpg`. The matrix converts a point on the court *floor plane* from image pixels into metres. It does not convert the shuttle's airborne image position to metres because the shuttle is not on that plane.

**Important:** the supplied 22-second sample is a composited 720×1280, 30 fps video in which the near baseline is not fully visible. Do not invent the missing corner to calibrate that video. Use a new unedited, landscape, full-court fixed-camera recording for real calibration.

## Tests

```bash
python -m unittest discover -v
```

## Step 2 — Detect and track badminton players (experimental)

Uses [RF-DETR Nano](https://github.com/roboflow/rf-detr) (pretrained COCO person detection) and [ByteTrack](https://github.com/roboflow/trackers) (ID association). Core RF-DETR Nano/Small weights and the tracker package are Apache-2.0 licensed, suitable for investigation in a commercial product. Other model variants may have different terms; audit all dependencies before deployment.

Step 2 has *additional, considerably larger* AI dependencies. On Windows, create an isolated Python environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-vision.txt
```

RF-DETR downloads pretrained model weights on the first run. Processing on CPU may be slow. An NVIDIA GPU can help, but we have **not** benchmarked your machine.

Run a small smoke test on a 10-second section (about 30 sampled frames):

```powershell
.\.venv\Scripts\python.exe player_tracking.py --video "C:\Users\kamal\Downloads\videoplayback.mp4" --out "out\players-001" --camera court-01 --sample-fps 3 --max-seconds 10 --roi "0.03,0.45,0.97,0.95"
```

### Step 2.1 — Diagnose zero player boxes in the example broadcast

Our initial run on the **360×640 portrait broadcast** successfully processed 30 frames, but reported **0 player boxes and 0 track IDs**. That alone does not tell us whether RF-DETR missed the players, the court filter removed them, or ByteTrack discarded the remaining boxes. The updated prototype records **counts for each stage**, and makes a separate detector-only contact sheet.

Run a new comparison into a **new output directory** (so the first result remains available):

```powershell
.\.venv\Scripts\python.exe player_tracking.py --video "C:\Users\kamal\Downloads\videoplayback.mp4" --out "out\players-002" --camera court-01 --sample-fps 3 --max-seconds 10 --model nano --confidence 0.15 --detect-roi "0.00,0.38,1.00,0.98" --roi "0.03,0.45,0.97,0.95" --track-high-confidence 0.20 --track-activation 0.30
```

Parameters in this **diagnostic experiment** are intentionally more permissive. They may increase false-positive boxes and ID switches; they are **not validated production settings**.

- `--detect-roi` crops the image **before** running RF-DETR, making players larger relative to model input. The coordinates of returned boxes are restored to the original image before all later stages.
- `--roi` separately filters **after** detection based on the feet of players, and is the same court filter as before.
- `--confidence` sets the RF-DETR minimum score. `--track-high-confidence` and `--track-activation` control ByteTrack's association and track-start thresholds. Defaults are still 0.25, 0.60, and 0.70 respectively.

Check `out\players-002\tracking_summary.json` → `diagnostics.stage_totals`:

| Counter | Interpretation if zero |
| --- | --- |
| `raw_person_detections` | RF-DETR didn't recognize people; inspect crop and model performance |
| `court_filtered_person_detections` | Person detections were outside the allowed `--roi`/court calibration |
| `tracker_output_boxes` | ByteTrack rejected the candidate detections |
| `boxes_with_track_id` | Boxes exist, but ByteTrack hasn't assigned usable IDs |

Open `out\players-002\detector_review\review_contact_sheet.jpg`. Grey boxes are **all person detections**, orange boxes are those remaining **after court filtering**. Yellow rectangle is the filter ROI; cyan rectangle is the detector crop. This view helps spot a detector/ROI problem even if no tracked boxes are displayed in the main preview.

Send back `tracking_summary.json` and **both** contact sheets (or zip the entire `players-002` directory). Do not assess player movement speed, shot quality, or footwork until the boxes and IDs are confirmed visually.

### Step 2.2 — Fix pretrained COCO person label mapping

The `players-002` diagnostic run reported **329 total object detections but zero person detections** in 30 frames. Our previous code assumed `class_id=0` meant person. That is incorrect for pretrained RF-DETR checkpoints using **sparse COCO category IDs**, where `class_id=1` means person. Fine-tuned models may instead use zero-based class IDs, so the fix preferentially uses the model's `detections.data["class_name"]` field, falling back to `class_id=1` only when class names are absent. The numeric fallback can be overridden with `--person-class-id` for a different model.

After pulling the latest code, run the **same comparison** with a fresh output directory:

```powershell
.\.venv\Scripts\python.exe player_tracking.py --video "C:\Users\kamal\Downloads\videoplayback.mp4" --out "out\players-003" --camera court-01 --sample-fps 3 --max-seconds 10 --model nano --confidence 0.15 --detect-roi "0.00,0.38,1.00,0.98" --roi "0.03,0.45,0.97,0.95" --track-high-confidence 0.20 --track-activation 0.30
```

Check `tracking_summary.json` for `diagnostics.class_id_histogram`, `raw_person_detections`, `court_filtered_person_detections` and `boxes_with_track_id`. Inspect the two contact sheets. Real inference verification on a different landscape clip is recorded below. Low-resolution broadcasts and camera cuts still cause genuine detection misses and ID switches even with the correct label mapping.

### Step 2.3 — Landscape clip verification (29 October 2026)

The local `Videos/videoplayback.mp4` is a **different source** from players-001/002: 640×360, 30 FPS, 790.848 seconds; SHA-256 `f317baacdd4ccce45febd9f416ba44b6c90071ebf3380c22f877b03376101130`. All runs below use only its first 10 seconds, RF-DETR Nano 1.11.2 and trackers 2.6.1. These counts cannot be used as an accuracy comparison against the old portrait clip.

| Run | FPS | Confidence | All objects | Raw people | Court kept | Tracker boxes | Boxes with IDs | Unique IDs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| players-003 (old portrait regions) | 3 | 0.15 | 432 | 96 | 78 | 78 | 64 | 8 |
| players-004 (landscape regions) | 3 | 0.15 | 516 | 196 | 63 | 63 | 52 | 8 |
| players-005 (landscape regions) | 15 | 0.15 | 2429 | 946 | 307 | 307 | 298 | 2 |
| players-006 (landscape regions) | 15 | 0.30 | 667 | 501 | 299 | 299 | 297 | 2 |

Visual inspection of both sheets showed that the old crop truncates the far player and retains a side official. The landscape crop fixes those issues in the reviewed frames. At 3 FPS the player IDs fragment; at 15 FPS IDs 0 (near) and 1 (far) persist through this short segment. With confidence 0.15, four frames contain extra unconfirmed boxes. Raising confidence to 0.30 removes these extras, at the cost of one missed person box at 5.867 seconds. There are two boxes in 149/150 frames; the first frame has unconfirmed IDs, and subsequent boxes retain IDs 0/1. Their near/far foot-position ordering remains consistent across every frame with both IDs. These are diagnostic observations, **not measured precision/recall or validation of the full match**.

Reproduce the cleaner short baseline into a fresh output directory:

```powershell
.\.venv\Scripts\python.exe player_tracking.py --video "Videos\videoplayback.mp4" --out "out\players-006-repeat" --camera court-01 --sample-fps 15 --max-seconds 10 --model nano --confidence 0.30 --detect-roi "0.10,0.20,0.90,1.00" --roi "0.15,0.25,0.85,1.00" --track-high-confidence 0.20 --track-activation 0.30
```

These regions apply only to this clip's opening camera view. Defaults remain unchanged. Review later rallies, camera cuts, and coach-labeled frames before movement analysis or coaching scores. Landscape contact sheets now use compact rows matched to image aspect ratio. Local videos and generated outputs remain excluded from Git.

### Step 2.4 — Review later windows without copying the video

Use `--start-seconds` together with `--max-seconds` (the window's duration). For example, this processes source seconds 60–70:

```powershell
.\.venv\Scripts\python.exe player_tracking.py --video "Videos\videoplayback.mp4" --out "out\players-007-repeat" --camera court-01 --start-seconds 60 --max-seconds 10 --sample-fps 15 --model nano --confidence 0.30 --detect-roi "0.10,0.20,0.90,1.00" --roi "0.15,0.25,0.85,1.00" --track-high-confidence 0.20 --track-activation 0.30
```

`--start-seconds` defaults to zero, must be finite and non-negative, and must precede the end of the video. A window reaching the end stops there. `sampling.start_seconds` records the offset. JSONL `frame_index` starts at zero for each run; `timestamp_seconds` and contact-sheet labels refer to the **source video**, approximately `start_seconds + frame_index / fps`. The preview starts at time zero for the selected window. Schema version is now 0.2.3. The tracker uses elapsed window time internally and begins afresh for each run: an ID in one output directory does not identify the same athlete in another run.

Review each window's camera view before interpreting its court filter or foot positions. Broadcast close-ups and replays do not share the original court geometry. Candidate-cut handling is described below; it still requires manual review and does not validate camera continuity.

Real-model checks on the same landscape source and settings above (150 frames per window):

| Run / source window | Raw people | Court kept / tracker boxes | Boxes with IDs | Unique IDs | Review finding |
| --- | --- | --- | --- | --- | --- |
| players-007 / 60–70 s | 376 | 299 | 297 | 2 | Both players visible in the nine review frames, consistent near/far IDs; one frame has only one box |
| players-008 / 300–310 s | 376 | 255 | 246 | 7 | Multiple broadcast close-ups interrupt court tracking; IDs fragment |
| players-009 / 720–730 s | 416 | 286 | 277 | 6 | Starts with a cropped close-up, then returns to court; far-player IDs fragment and extra boxes occur |

Both detector-only and tracked sheets were inspected for each window. These are diagnostic counts and sampled visual checks, not exhaustive ground-truth accuracy measurements. The later failures mean whole-match movement analysis remains blocked on camera-view handling and further tracking validation. Do not join trajectories across these camera changes. Coaching metrics remain unavailable. Sixteen tests pass, including a two-color video check that verifies seeking extracts the requested content and an end-to-end check of source-relative timestamps.

### Step 2.5 — Break tracking at candidate camera cuts

The tracker now compares consecutive full-frame 64×36 RGB thumbnails. A mean absolute channel change of at least `--scene-cut-threshold` (default `0.15`, on a 0–1 scale) starts a new scene. This is an experimental edit-boundary heuristic, not a court-view classifier. Fast motion, lighting changes or graphics can trigger false cuts; gradual edits and similar views can be missed, especially at low sampling rates. Use `--scene-cut-threshold 0` to disable resets for comparison.

At each candidate cut the program resets ByteTrack, clears drawing trails, and assigns new exported track IDs so IDs are not reused across scenes within the same run. IDs still do not persist across separate runs. A camera cut **invalidates the supplied homography for the rest of that run**: subsequent boxes omit `foot_court_xy_m`, including when the broadcast returns to a similar view. Start a separately reviewed run with the correct calibration to resume court coordinates. Detector crop and rectangular ROI remain configured as before and can be inappropriate for the new view; their boxes are candidates requiring review.

Schema 0.3.0 adds JSONL `scene_id` (starts at zero), `scene_cut` (true on the first sampled frame after a candidate cut), and `court_view_status` (`requires_review` initially, `invalidated_after_cut` thereafter). Summary `scene_review` records the method, threshold, candidate boundary timestamps, scores and manual-review status. Neither status certifies a usable court view. The initial view may already be a close-up. Both contact sheets include the sampled frames immediately before and after candidate cuts, in addition to the usual nine frames, and the tracked preview labels the scene. Coaching metrics remain unavailable.

Recheck the broadcast edits using the Step 2.4 command with `--start-seconds 300`, a fresh output directory and optionally explicit `--scene-cut-threshold 0.15`. Continue using 15 FPS for this short experiment; do not infer that a higher scene count or more unique IDs means more athletes.

Verified on 29 October 2026 with the same source and 15 FPS / confidence 0.30 settings:

| Run / source window | Candidate cut timestamps (source seconds) | Result |
| --- | --- | --- |
| players-010 / 60–70 s | None | Control keeps one scene and IDs 0/1; same 299 boxes and 297 boxes with IDs as players-007 |
| players-011 / 300–310 s | 301.267, 302.733, 306.333, 308.000 | Five scenes; all exported ID sets disjoint across scenes; 255 boxes, 245 with IDs |
| players-012 / 720–730 s | 720.400, 725.200, 725.933 | Four scenes; all exported ID sets disjoint across scenes; 286 boxes, 276 with IDs |

Both expanded sheets were inspected in the two edit windows. Before/after frames show broadcast edits at the flagged boundaries. This verifies these sampled examples, not scene-detection recall for the whole video. Close-up truncation, low-confidence extras, and within-scene ID fragmentation remain. Resets can reduce the number of boxes with IDs because each new scene must establish tracks again. Eighteen tests pass, including a three-scene synthetic video that checks unique IDs, cleared trails, homography invalidation on both cuts, and the disabled-reset comparison. No real court calibration was available for the broadcast; calibration invalidation was tested synthetically.

### Step 2.6 — Exclude mismatched views and suppress duplicate boxes

For a fixed camera, supply a **visually reviewed, full-resolution wide-court reference** with `--court-reference`. The program compares full-frame 64×36 RGB thumbnails against this image. Frames exceeding `--court-view-threshold` (default 0.08 mean absolute channel difference) retain raw detector diagnostics but supply **zero boxes to ByteTrack and export no player tracks**. Entering or leaving a matching view resets tracking, even when candidate-cut detection is disabled. The initial frame is checked too, so a run beginning in a close-up is excluded immediately. The reference hash and rejection count are recorded for audit.

This is an opt-in similarity gate for an established camera view, not a general court classifier. Similar-looking incorrect views may pass, while lighting, camera movement, and different broadcast layouts can reject valid court footage. Review before using it on another camera or match. A match does not restore calibration after a boundary or prove court geometry is valid.

Person boxes also pass score-ordered non-maximum suppression (`--person-nms-iou`, default 0.5; 0 disables). Lower-score boxes overlapping a higher-score box above that IoU are removed before tracking. Overlapping real people can be suppressed too, so evaluate doubles and occlusions separately. This does not force a two-player count or fabricate missing boxes. Detector confidence remains independently configurable; the opening segment's missed 0.288-confidence box is recovered at confidence 0.25 without the low-confidence extras seen at 0.15.

Create the reference from the reviewed wide-court frame in this source, then run a new comparison:

```powershell
ffmpeg -hide_banner -loglevel error -y -ss 3 -i "Videos\videoplayback.mp4" -frames:v 1 "out\court-reference.jpg"
.\.venv\Scripts\python.exe player_tracking.py --video "Videos\videoplayback.mp4" --out "out\view-review-repeat" --start-seconds 300 --max-seconds 10 --sample-fps 15 --confidence 0.25 --detect-roi "0.10,0.20,0.90,1.00" --roi "0.15,0.25,0.85,1.00" --track-high-confidence 0.20 --track-activation 0.30 --court-reference "out\court-reference.jpg"
```

Schema 0.4.0 adds `court_view_review` in the summary (reference filename/hash, threshold and rejected-frame count). JSONL `court_view_status` is `reference_match` or `reference_mismatch` when enabled; without a reference the prior statuses remain. `court_view_difference` records the normalized difference or null when disabled. `court_geometry_status` independently states whether the supplied homography is available but unverified, absent, or invalidated. `scene_cut` now marks any tracking boundary, including reference-view transitions; each summary boundary records its reason. New stage counts `deduplicated_person_detections` and `view_accepted_person_detections` distinguish duplicate removal from view rejection. Detector sheets show all raw people in grey and **final tracker input** in orange, labeled TRACK INPUT; rejected close-ups therefore show only grey boxes. The preview's scene/status badge now has a dark background for readability.

`--roi` is an **optional normalized rectangle** `(left, top, right, bottom)` in the range 0–1. It keeps detections whose approximate *foot position* is inside the rectangle, helping exclude spectators/officials. The example coordinates suit the sample broadcast's first few seconds, **not all camera setups**. Omit `--roi` initially for a new fixed-camera video, then inspect its generated review sheet and add an appropriate region.

Generated files:

- `out/players-001/tracked_preview.mp4`: annotated video, played at the original time scale but sampled to 3 FPS in this example
- `out/players-001/player_tracks.jsonl`: one row per sampled frame, with boxes, approximate foot pixels, confidence, and track IDs
- `out/players-001/tracking_summary.json`: model versions, diagnostic counts, warnings, and file names
- `out/players-001/review_contact_sheet.jpg`: nine-frame annotated QA sheet (when enough frames are available)
- `out/players-001/review_frames/`: individual annotated review images
- `out/players-001/detector_review/review_contact_sheet.jpg`: **pre-tracker** diagnostic sheet for inspecting raw versus court-filtered person boxes
- `out/players-001/detector_review/frame_*.jpg`: the separate diagnostic review frames

**Crucial limitations:** ID 1, ID 2, etc. mean *temporary track IDs*, not known athletes or sides of court. A camera cut or occlusion can trigger ID switches. Until a coach validates output, do not interpret the count of unique IDs as the number of players. The system does not yet count rallies or shots, or produce coaching recommendations.

If you calibrated a full-court camera in Step 1B, optionally append `--calibration "out\court-01\calibration.json"`. It will filter foot positions to the mapped court and add approximate court coordinates in metres. The calibration must use a frame from the exact same camera and video resolution.

For better player tracking, use original, fixed-camera footage with the full court visible; low-resolution broadcast edits are useful for testing the program but not for accuracy claims.

## Next milestones

| Step | Deliverable | Validation |
| --- | --- | --- |
| 1 (this package) | Reliable video ingestion and court calibration | Metadata matches source; geometric corner re-projection; manual visual QA |
| 2 (this package) | Experimental person detection, ID tracking, foot-position candidates | Coach-labeled frames; identity switches; detection precision and recall |
| 3 | Shuttle trajectory and shot-event timing | Annotated shuttle locations and contact timestamps |
| 4 | Shot types, rally segmentation, and court heatmaps | Shot-level precision/recall and class-wise confusion matrix |
| 5 | Academy dashboard and verified coaching reports | Coach review, correction flow, consistency across sessions |

## Product design decisions

- **Inference:** specialized local computer-vision models (not Ollama).
- **Reports:** a local LLM through Ollama may later explain **verified numeric observations** and suggest coach-reviewed drills.
- **Backend:** ASP.NET Core for academy/court/player/session management; Python inference as a separate worker.
- **Storage:** retain calibration by physical camera and session, and record model versions and confidence with every future shot event.
- **Quality:** offline batch processing first, real-time overlays later. Show uncertainty rather than manufacturing certainty.
- **Privacy:** do not upload minors' footage without appropriate guardian consent; institute access controls and retention policies before deployment.
