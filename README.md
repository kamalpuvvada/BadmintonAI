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

### Step 2.7 — Optional second detector crop for jumps

A far player's jump can rise above the detector crop or the foot-region filter. Lowering confidence alone cannot fix a truncated view. An optional `--secondary-detect-roi` runs the **same model** on a taller crop with its own `--secondary-confidence` (default 0.35), restores both passes to original coordinates, and combines their detections before person selection, court filtering and NMS. The second pass runs only on reference-matching frames when a reference is supplied. NMS must be enabled. This costs additional inference time and remains an experimental per-camera configuration.

Example for this landscape broadcast's reviewed view (create the reference as in Step 2.6):

```powershell
.\.venv\Scripts\python.exe player_tracking.py --video "Videos\videoplayback.mp4" --out "out\jump-review-repeat" --start-seconds 720 --max-seconds 10 --sample-fps 15 --confidence 0.25 --detect-roi "0.10,0.20,0.90,1.00" --secondary-detect-roi "0.10,0.05,0.90,1.00" --secondary-confidence 0.50 --roi "0.15,0.20,0.85,1.00" --person-nms-iou 0.40 --track-high-confidence 0.20 --track-activation 0.30 --court-reference "out\court-reference.jpg"
```

Schema 0.4.1 records the optional secondary crop and confidence, plus `secondary_all_class_detections`. `all_class_detections` and raw person counts include **both passes before duplicate suppression**, so they are not counts of unique physical objects. The secondary counter is a subset of the all-class total, not an additional sequential stage. No boxes are synthesized. Bounding-box bottoms during jumps are not floor-contact measurements: movement and coaching metrics remain unavailable.

When a person box touches an internal inference-crop edge within two pixels, duplicate suppression also removes it if a larger, higher-confidence person box covers at least 65% of its area. A covering non-edge box can also replace a crop-edge fragment whose confidence is up to 0.10 higher. This handles crop-edge lower-body fragments that have low IoU with a full-body box. Boxes touching only the original image boundary do not receive this flag. The rule can still suppress overlapping real people, so it requires separate doubles/occlusion validation. Disabling person NMS disables this rule as well. RF-DETR's crop-specific `source_image` and `source_shape` metadata are restored to the original frame before merging; a regression test uses the actual installed Supervision merge implementation to verify this compatibility.

The example's secondary confidence 0.50 is specific to this reviewed clip. Experiments showed that simply lowering confidence admitted weak partial-body boxes, a taller crop alone changed which frames were missed, and changing only the rectangular court filter did not recover the two jump misses. Keeping the original pass plus a higher-confidence taller pass recovered those detections; crop-edge suppression is necessary to prevent both the fragment and the full person appearing together. These experiments do not establish production settings for other cameras.

Final comparison on 29 October 2026 uses the **same Step 2.7 settings** in every window, RF-DETR Nano 1.11.2 and trackers 2.6.1:

| Run / source window | Sampled frames | Excluded view frames | Accepted frames with two boxes | Boxes with IDs | Unique IDs |
| --- | --- | --- | --- | --- | --- |
| players-044 / 300–310 s | 150 | 47 | 103/103 | 200 | 6 (two per court scene) |
| players-045 / 720–730 s | 150 | 17 | 133/133 | 262 | 4 (two per court scene) |
| players-046 / 0–10 s | 150 | 0 | 150/150 | 298 | 2 |
| players-047 / 60–70 s | 150 | 0 | 150/150 | 298 | 2 |
| players-048 / 480–490 s (additional comparison window) | 150 | 0 | 150/150 | 295 | 5 (fragmentation remains) |

All 686 accepted frames contain two boxes; all 64 excluded frames export no tracks and skip secondary inference. Both contact sheets were visually reviewed for all five windows. The known two jump misses and duplicate-box examples are resolved under this configuration. Stage totals match sums of JSONL counts, exported ID sets do not overlap between scenes, and coaching metrics remain unavailable. Twenty-four tests pass, including native Supervision metadata compatibility, crop coordinate restoration, edge-fragment suppression, reference-view rejection (including an initial close-up), scene resets and calibration invalidation.

**Historical finding before Step 2.8:** the 480–490 second window retains both detections but changes IDs within the same camera view. Follow-up inspection found timeline edits in that window, as described below. Do not derive continuous movement paths across those edits. Two boxes per frame plus sampled visual checks are not ground-truth precision/recall, bounding-box localization accuracy, or proof of identity continuity. Full-match, doubles, occlusion and different-camera validation are still required. The second inference pass remains optional and is slower; the default path is a single crop.

### Step 2.8 — Detect timeline edits within a matching court view

Source-frame inspection around the apparent ID breaks in players-048 shows same-camera jump cuts at approximately **483.933 s** and **488.000 s**. At the first cut a player abruptly changes from a dive to standing and the score graphic changes; at the second both players abruptly relocate. The largely unchanged background made these edits too small for the original 0.15 full-frame cut threshold. Loosening tracker association did not resolve the actual timeline discontinuity and is not the selected fix.

The optional `--court-cut-threshold` enables a more sensitive mean RGB change check **only when both adjacent sampled frames match the court reference**. It requires `--court-reference`, defaults to 0 (disabled), and leaves the original global cut threshold unchanged. For this source at 15 FPS, append `--court-cut-threshold 0.009` to the Step 2.7 command. A boundary resets ByteTrack and trails, assigns fresh exported IDs, invalidates the homography, and adds before/after frames to both review sheets. It does not reject otherwise matching court frames or change detector boxes.

Schema 0.4.2 records `scene_review.matching_view_cut_threshold` and the candidate boundary reason `matching_view_jump_cut`. The reason identifies an experimental candidate, not a verified edit for every video. Fast motion, lighting, graphics or another sampling rate can cause false positives or missed edits; this low threshold needs camera-specific review.

Decoded-source image auditing found mean changes of 0.01165 and 0.00939 at the two edits. No additional matching-view boundaries at 0.009 were found in the four comparison windows (0–10, 60–70, 300–310 and 720–730 seconds); their maximum matching-view changes were 0.00812, 0.00762, 0.00625 and 0.00334. These are source-frame checks, not four new RF-DETR inference runs. Native ByteTrack replay of the previously exported model boxes produced two local IDs per court scene and no within-scene near/far ID changes across all five windows after applying the edit boundaries. Near/far ordering is a diagnostic for these reviewed singles windows, not an athlete identity system.

Real inference in **players-050**, using the Step 2.7 settings plus `--start-seconds 480 --court-cut-threshold 0.009`, verified both boundaries, two boxes in every one of 150 frames, 294 boxes with confirmed IDs, and three scenes. Exported IDs are `{0,1}`, `{2,3}`, and `{4,5}` in the respective scenes, with no cross-scene reuse. Both expanded contact sheets were inspected. The six unique IDs are intentional scene-local tracks for two visible players, not six athletes. There are two unconfirmed boxes when each scene starts. A comparison at threshold 0.010 (players-049) caught only the first edit, illustrating the sensitivity of this experimental setting. Twenty-six tests pass. Continuous movement must still be restricted to reviewed uninterrupted segments, and coaching metrics remain unavailable.

### Step 2.9 — Record scene review before movement analysis

Create a local review packet from existing tracking output; this does not rerun inference or change boxes and IDs:

```powershell
.\.venv\Scripts\python.exe tracking_review.py prepare --run "out\players-050" --expected-players 2
```

Open `out/players-050/tracking_review.html`. It includes the sampled tracked images, a playable preview, links to individual scenes, original-video timestamps, temporary track IDs, review flags and verification status. The HTML and JSON stay beside their source artifacts and work offline. The scene links use preview-relative time; visible timestamps remain source-relative. Choose the expected player count explicitly (1–4), or omit it to leave count checks unspecified. Unique track IDs are never automatically treated as athlete counts.

All scenes begin **pending**. After actually reviewing a complete scene, record who checked its boxes, ID continuity, camera view and boundaries. For example (replace the reviewer name and notes):

```powershell
.\.venv\Scripts\python.exe tracking_review.py decide --run "out\players-050" --scene 0 --decision accepted --reviewer "Reviewer name" --reviewer-kind human --boxes-verified --ids-verified --view-verified --boundaries-verified --notes "Reviewed the complete scene"
.\.venv\Scripts\python.exe tracking_review.py verify --run "out\players-050"
```

Use `--decision rejected` to reject a scene, or `--decision pending` to revoke its current acceptance. Rejected decisions require a reviewer; verification flags are required only for acceptance. Each decision is retained in that scene's history. Re-preparing an unchanged packet preserves decisions and history; it does not approve anything. Choose the expected player count when first preparing the packet; an existing packet with a different count is rejected rather than silently reconfigured.

`tracking_review.json` schema 0.1.0 binds decisions to SHA-256 hashes of **both** `tracking_summary.json` and `player_tracks.jsonl`. Changed source files invalidate the review; stale packets are preserved and cannot be approved or verified. Scene facts are recomputed from source, not trusted from editable derived fields. Excluded views, missing confirmed tracks, reused/duplicate IDs and expected-count/ID-count inconsistencies block acceptance. A count check cannot detect every ID swap: the reviewer must still inspect continuity.

`verify` returns freshly computed `approved_intervals` only for accepted scenes. These contain inclusive local frame indices, source timestamps and temporary track IDs. `approval_scope` is explicitly `tracking_in_image_pixels`: this review does not validate court calibration or floor contact. Unconfirmed frames are omitted, gaps split intervals, and intervals never cross a scene boundary or a change in track-ID set. Pending and rejected scenes contribute no intervals. Future movement consumers must call verification and key tracks by the source binding, scene and track ID; they must not blindly trust stored interval fields or join equal numeric IDs across runs. This local review log is **not an authenticated signature or authorization system**; server-side access control remains future RACKETEK integration work.

This step produces no movement measurements or coaching scores, even after all scenes are accepted. Five local packets were prepared for players-044, 045, 046, 047 and 050: all 14 scenes remain pending, with zero approved intervals. The report for players-050 was checked in a browser: all 13 review images and the 10-second preview loaded successfully. Thirty-six tests pass, including approval checks, stale-source rejection, decision preservation/revocation, excluded/fragmented scene blocking, escaped report text and interval separation at unconfirmed frames and scene boundaries.

### Step 2.10 — Export reviewed image positions

Export a small, offline JSON contract for future backend consumption:

```powershell
.\.venv\Scripts\python.exe tracking_export.py export --run "out\players-050"
.\.venv\Scripts\python.exe tracking_export.py verify --run "out\players-050"
```

`reviewed_tracks.json` schema 0.1.0 includes the source hashes, a hash of the validated review snapshot, image dimensions, sampling rate, approved intervals and per-track segments. Each sample retains its local frame index, original-video timestamp, bounding box, approximate box-bottom pixels and detector confidence. Coordinates use the original image's top-left origin. Positions are copied without interpolation or conversion to metres. Out-of-image accepted samples and invalid confidence values fail export instead of being silently clamped.

Only explicitly accepted scenes contribute data. Unconfirmed frames are omitted; gaps and scene boundaries split segments. Identify a track using **both source hashes, scene ID and track ID**; `interval_index` distinguishes separated segments. Equal IDs in another run do not imply the same athlete. Export reads and validates one source/review snapshot, recomputing eligibility instead of trusting saved interval fields. `verify` rebuilds the current export and rejects changed data, changed review decisions (including revocation), or modifications to the exported JSON. A saved file cannot revoke itself: consumers must verify against the current run before use, and a future backend must enforce its own access control and invalidation.

Pending or rejected-only runs export `status: no_approved_intervals` with empty segments. This is a successful gated result, not missing inference. Even accepted exports keep `movement_metrics` and `coaching_metrics` unavailable: image-space review does not establish ground contact, calibration, distance, speed or coaching quality.

Initial export verification on 30 October 2026 exercised players-044, 045, 046, 047 and 050 while all 14 scenes were pending, correctly producing zero segments. At that checkpoint, 42 tests passed and accepted-export behavior had only been exercised with synthetic fixtures. The subsequent automated visual review below records actual image-space decisions and exercises the accepted exports on real tracking output.

Local cleanup removed 41 superseded `players-*` experiment folders (98,012,352 bytes), preserving their summaries and summary hashes in `out/tracking_cleanup_2026-10-30.json`. The five final runs above, `out/session-001`, `out/dive-transition-review`, the reference image and JSON audits remain. Earlier run names in this README are historical experiment records; their artifacts may have been deleted. Generated data and the cleanup inventory stay local and are not committed.

### Step 2.11 — Completed automated visual review

At the user's request to proceed autonomously, Codex inspected **every one of the 750 sampled frames** in the five final windows on 30 October 2026. Source frames were decoded again with the original sampling settings and overlaid with the saved JSON boxes; 30 dense sheets cover all frames. Source SHA-256 is `f317baacdd4ccce45febd9f416ba44b6c90071ebf3380c22f877b03376101130`. Inspection found no gross missing-player, duplicate-player or within-scene ID-swap error in these windows, including jumps and dives. Box placement remains approximate; this is not a ground-truth IoU or precision/recall benchmark, a full-match evaluation, or a review of unsampled source frames.

The four close-up scenes were explicitly rejected for downstream analysis. All ten court scenes were explicitly accepted for sampled image-space tracking after inspecting their frames, IDs, views and boundaries. The same-camera cuts at 483.933 and 488.000 seconds are visible discontinuities and correctly reset IDs. No inference output or track ID was changed to obtain these decisions.

| Run | Accepted scenes | Excluded scenes | Exported segments | Position samples |
| --- | --- | --- | --- | --- |
| players-046 | 0 | None | 2 | 298 |
| players-047 | 0 | None | 2 | 298 |
| players-044 | 0, 2, 4 | 1, 3 | 6 | 200 |
| players-050 | 0, 1, 2 | None | 6 | 294 |
| players-045 | 1, 3 | 0, 2 | 4 | 262 |

All 14 scenes now have decisions: **10 accepted, 4 rejected, 0 pending**. Accepted exports contain 20 separate track segments and 1,352 position samples, excluding the 64 close-up frames and the first unconfirmed frame in each accepted scene. Each accepted interval stays within one scene. All five exports were regenerated and verified against the current sources and decisions. Local evidence is in `out/full-frame-audit/audit.json`, the 30 sheets, and `out/full-frame-audit/completed_review.json`; each decision records the evidence audit hash. `out/scene-review/index.html` provides scene clips and review notes.

Review decisions and history now retain optional `reviewer_kind`: `automated`, `human`, or `unspecified`. The CLI option is `--reviewer-kind`; its default and legacy records are `unspecified`, never an inferred human review. Reports display the type; verification and exports expose `review_provenance.accepted_by_reviewer_kind`. Codex recorded the initial decisions with kind **automated**. Kamal subsequently explicitly confirmed all four checks for **A (players-046 scene 0)** and **B (players-047 scene 0)** in chat. Those two decisions now record **Kamal / human**, preserving the earlier automated decisions in history. Final accepted-scene provenance is **2 human, 8 automated**; the four excluded scenes retain automated decisions. Counts and automated checks alone never accept scenes; these decisions followed explicit visual inspection. As with other local review fields, the kind is a recorded claim, not authenticated proof.

Forty-five tests pass, including preservation of automated provenance in history, HTML and exports, legacy handling, unknown-kind rejection, and invalidation when review provenance changes. A compact result with source/output hashes and scene decisions is versioned in `validation/tracking_review_2026-10-30.json`; the video and frame evidence remain local. Human confirmation covers A and B only; coach validation is not recorded. Movement measurements, court-calibrated metrics and coaching scores remain unavailable.

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

## Step 3.1 — Experimental shuttle candidate diagnostics

Run the first shuttle baseline on a **reviewed player run**, using the matching original video:

```powershell
.\.venv\Scripts\python.exe shuttle_candidates.py --video "Videos\videoplayback.mp4" --player-run "out\players-046" --out "out\shuttle-repeat"
```

The output directory must be new. The tool revalidates the player review and source identity, uses the original video rate (30 FPS for this source), and processes the same time window as the player run. A higher-rate frame is eligible only inside an accepted player interval's timestamp bounds. This inherits player-view eligibility, **not shuttle approval or review of every higher-rate frame**. Rejected views, unconfirmed starts and time outside accepted bounds produce no candidates. The player source hashes, review-snapshot hash and full video SHA-256 are recorded for audit; outputs are snapshots, not a live authorization system.

The baseline finds tiny bright components that change against both adjacent frames, subtracts a scene-local median background, masks player boxes from neighboring player samples, and applies an optional normalized `--roi` (default `0.10,0.08,0.90,1.00`). These are experimental pixel thresholds, recorded in the summary, for the 640×360 source. It uses no trained shuttle classifier. A conservative linker starts only from a single candidate, requires consecutive frames, and terminates on ambiguity, excessive motion, missing observations, rejected views or a scene change. It never interpolates across gaps. The median uses up to 31 frames from the same scene with offline look-ahead; no temporal image context or background is shared across cuts.

Generated artifacts:

- `shuttle_candidates.jsonl`: every sampled frame, candidates, optional unverified proposal and explicit missing/excluded/ambiguous status. `verified_shuttle_position_px` is always null.
- `shuttle_summary.json`: source/review hashes, sampling, thresholds, diagnostic counts and limitations. `verified_shuttle_positions` stays zero; shot events and coaching metrics stay unavailable.
- `candidate_preview.mp4`: native-rate preview; orange circles are candidate components, cyan boxes are linked proposals. They are **not verified shuttle detections**.
- `candidate_contact_sheet.jpg`, nine native-resolution `review_*.jpg` images, and `index.html`: offline inspection evidence.

On 31 October 2026, four final runs processed **1,200 frames** after an initial comparison exposed white-marking/graphic flicker. Adding the scene background contrast check is covered by a regression test. It reduced some proposal clutter but does not establish an accuracy improvement without ground truth.

| Final run | Player run / source window | Frames with linked proposals | Proposal segments | Verified shuttle positions |
| --- | --- | --- | --- | --- |
| shuttle-005 | players-046 / 0–10 s | 60 | 30 | 0 |
| shuttle-006 | players-045 / 720–730 s | 33 | 24 | 0 |
| shuttle-007 | players-050 / 480–490 s | 51 | 27 | 0 |
| shuttle-008 | players-044 / 300–310 s | 54 | 25 | 0 |

All four previews decode to 300 frames each; status totals match JSONL rows; rejected frames contain no candidates; every proposal segment is consecutive and confined to one scene. The 198 linked-proposal frames and 106 segments are **diagnostic counts, not recall, detected shots, rallies or valid trajectories**. All four final contact sheets were inspected. Initial single-frame spot checks at 4.267, 728.067, 484.400 and 303.500 seconds incorrectly classified all four proposals as racket heads. **Correction from Step 3.2:** enlarged temporal context shows the first, third and fourth are shuttles moving independently of the racket; the 728.067 s example remains a local racket-head negative. The validation record retains the previous classifications with explicit corrections. Player masking also deliberately loses near-body shuttle visibility. Blur, low contrast, court-line overlap and slow/static motion cause additional misses. Racket confusion and fragmentation remain, so the baseline is retained for diagnostics and not promoted to verified trajectories.

Results, hashes and the four failure labels are recorded in `validation/shuttle_candidates_2026-10-31.json`. Local source visibility sheets and a four-example failure sheet are in `out/shuttle-visibility`. Fifty-two tests pass, including real FFmpeg pipeline coverage, source mismatch, output preservation, player eligibility bounds, scene and gap resets, ambiguity, stationary/ghost rejection, masks and background flicker. Installed dependencies pass `pip check`.

The existing pretrained person model is not treated as a shuttle detector. No shuttle trajectories, contact timing, shot types or coaching scores are verified by this baseline.

## Step 3.2 — Labeled development benchmark and measured color experiment

`validation/shuttle_benchmark_seed_2026-10-31.json` contains 16 selected source frames: **12 visible shuttle locations, one local racket-head region, and three additional uncertain frames**. All labels are attributed to Codex / automated visual review. Six consecutive enlarged court-background frames and three colored-blur frames were inspected directly; three candidate-selected examples were corrected from the earlier misclassifications after inspecting their temporal context. The corrected examples are not independent ground truth. Centers of tiny components or streaks are approximate image locations, not physical cork positions.

Prepare native-resolution source images from the hash-matched video, then evaluate candidate output:

```powershell
.\.venv\Scripts\python.exe shuttle_benchmark.py prepare --seed "validation\shuttle_benchmark_seed_2026-10-31.json" --video "Videos\videoplayback.mp4" --out "out\benchmark-repeat"
.\.venv\Scripts\python.exe shuttle_benchmark.py evaluate --benchmark "out\benchmark-repeat\benchmark.json" --prediction "shuttle-005=out\shuttle-005" --prediction "shuttle-006=out\shuttle-006" --prediction "shuttle-007=out\shuttle-007" --prediction "shuttle-008=out\shuttle-008"
```

The packet's `index.html` shows green point labels, orange local distractor regions and provenance. Its `benchmark.json` binds each unmarked source image to its SHA-256. Evaluation rejects altered/escaping images, mismatched source video or dimensions, changed sampling, malformed rows, duplicate or contradictory labels, invalid coordinates and proposals that do not reference a candidate. Reports bind the benchmark and prediction bytes to hashes. Default point tolerance is four native pixels (`--tolerance-px`). An uncertain frame is excluded from point scoring; a local racket label **does not mean the entire frame is shuttle-absent**. These selected, correlated development examples are not a held-out accuracy benchmark or full-video precision/recall test. They must not be used as evidence of generalization.

An opt-in `--allow-tinted-streaks` mode in `shuttle_candidates.py` relaxes color thresholds only while retaining motion, scene-background contrast, player masks, ROI and cut/gap rules. It recovers color-tinted shuttle blur that the white-component test misses. It can also admit more distractors; the default remains unchanged. To reproduce a trial, add that flag to the Step 3.1 command and choose a new output folder. For evaluation, map the original benchmark source name to the trial output directory, for example `--prediction "shuttle-005=out\shuttle-009"`.

The corrected local packet is `out/shuttle-benchmark-002` (draft 001 is explicitly superseded). Four full-window tinted trials, shuttle-009 through 012, processed another 1,200 frames. At four-pixel tolerance:

| Selected development cases | Default white components | Opt-in tinted streaks |
| --- | --- | --- |
| Visible points with a nearby candidate | 10/12 | 12/12 |
| Visible points with a nearby linked proposal | 10/12 | 9/12 |
| Local racket regions containing a candidate | 1/1 | 1/1 |
| Local racket regions selected as a linked proposal | 1/1 | 0/1 |

The experimental mode recovers both red-background observations but increases ambiguity and reduces linked point hits. Failure to select the racket is an abstention, not evidence that it learned racket identity. Across the four full windows, linked-proposal frames increased from 198 to 335 and segments from 106 to 152; these are not accuracy gains. Both modes remain unverified, and the trial is **not promoted as better tracking**. A separate extraction-only ablation on these labeled cases compared temporal lag 1 against lag 3: white hits stayed 10/12 and tinted hits 12/12, with the racket candidate remaining in every condition. Wider temporal comparison was not adopted.

The versioned evaluation record is `validation/shuttle_benchmark_evaluation_2026-10-31.json`; local source/context images, packet labels and comparisons are retained for inspection. All four trial contact sheets were inspected. Sixty-one tests pass, including native source preparation, exact tolerance boundaries, unknown-frame handling, source/image integrity, contradictory labels, color-mode opt-in and retained masks. No verified trajectories, shot events or coaching scores are enabled. The next milestone is a broader independently reviewed dataset and shuttle-specific model evaluation; this seed now provides reproducible checks for that work.

## Step 3.3 — TrackNetV3 model evaluation

The first learned shuttle baseline uses the **tracking module** from
[TrackNetV3](https://github.com/qaz812345/TrackNetV3), pinned at upstream commit
`6eda442ada1740573f200f836d93edc9a541ee86`. The author's
[MIT license](https://github.com/qaz812345/TrackNetV3/blob/6eda442ada1740573f200f836d93edc9a541ee86/LICENSE)
explicitly includes pretrained checkpoints. The small vendored architecture, license
and provenance are in `third_party/tracknetv3`; InpaintNet is omitted. No training
dataset or video is downloaded. The author-linked tracking checkpoint stays in ignored
local storage. Initial setup requires internet; inference thereafter is offline.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-model-download.txt
.\.venv\Scripts\python.exe download_tracknet.py --out "out\models"
.\.venv\Scripts\python.exe shuttle_tracknet.py --video "Videos\videoplayback.mp4" --player-run "out\players-046" --out "out\tracknet-repeat"
```

The download helper verifies the archive and tracking file against the recorded
fingerprints, reads only the named tracking member, and preserves a mismatched
existing checkpoint instead of overwriting it. Checkpoint loading uses PyTorch's
restricted `weights_only=True` mode without an unrestricted fallback. The evaluated
checkpoint SHA-256 is `df867641a02712b021f04548ff4b1208ddfdb47f629ab2094ceb978667e83b1a`.
The optional downloader dependency is not needed for offline inference with an
already-downloaded checkpoint.

This checkpoint takes eight RGB frames with an RGB median background prepended
(27 channels at 512×288). The adapter runs on CPU (`--threads`, default 4), processes
only consecutive frames inside accepted player intervals, estimates a median from
up to 31 frames in each eligible group, and uses non-overlapping eight-frame blocks
plus an overlapping tail. Heatmaps for shared tail frames are averaged before
thresholding at 0.5. The adapter never pads a short group, shares model context across
a cut/rejected interval, inpaints a trajectory, or interpolates a missing position.
Groups shorter than the checkpoint sequence are explicitly unprocessed. Source RGB
channel order and native coordinate scaling are tested. It uses full-frame model
input rather than the heuristic's player-box masks. Model heatmap activation is
uncalibrated and does not certify shuttle identity.

Outputs use the existing candidate JSONL/summary and review artifact conventions,
so `shuttle_benchmark.py evaluate` can compare them directly. Orange circles mark
thresholded heatmap components and cyan boxes mark conservative linked proposals.
Every `verified_shuttle_position_px` remains null. Output summaries retain checkpoint,
source and player-review hashes, model/version details, window settings and timing.
This modified inference protocol is not the author's published temporal ensemble
and inpainting evaluation; no upstream accuracy numbers are inherited.

Five real runs, tracknet-001 through 005, processed **1,500 native-rate frames** from
players-046, 045, 050, 044 and 047 respectively. Each preview decodes to 300 frames.
Rejected player intervals contain no heatmap observations, and all 57 linked proposal
segments stay consecutive and scene-local. There are 983 frames with linked proposals;
this is a diagnostic count, not an accuracy or recall measurement. Local CPU times
were approximately 30–37 seconds per 10-second window including decoding, annotation
and preview encoding but excluding model loading; this is an offline baseline.
All five contact sheets were inspected; missing observations and occasional ambiguous
heatmaps remain.

An additional nine-case packet from the 60–70 second window was labeled **before
observing TrackNet predictions**: six visible shuttle points at 67.000–67.167 seconds
and three uncertain frames. Its versioned seed is
`validation/shuttle_control_seed_2026-10-31.json`; the prepared local packet is
`out/shuttle-control-benchmark-001`. The extra labels are automated, approximate and
correlated, from the same match and camera. They are not an independent held-out
dataset; overlap between the source video and upstream training footage is unknown.

| Selected labeled observations, four native-pixel tolerance | Default heuristic | TrackNetV3 adapter |
| --- | --- | --- |
| Original development points matched by candidates | 10/12 | 11/12 |
| Original development points matched by linked proposals | 10/12 | 11/12 |
| Local racket region containing a candidate / linked proposal | 1/1 / 1/1 | 0/1 / 0/1 |
| Additional-window points matched by candidates | 2/6 | 6/6 |
| Additional-window points matched by linked proposals | 2/6 | 6/6 |

The model misses the labeled blurred observation at 5.100 seconds. The small comparison
supports continued model evaluation, not a general accuracy claim or automatic
trajectory acceptance. The heuristic remains available and no existing default is
changed. Model results, source/output fingerprints, control-label provenance and
limitations are recorded in `validation/tracknet_evaluation_2026-10-31.json`.
Sixty-eight tests pass, covering preprocessing, coordinate restoration, restricted
checkpoint loading, exact-member archive handling, short-context behavior and scene/gap
isolation, including native FFmpeg integration. Installed dependencies pass `pip check`.
Shot events, court-calibrated metrics and coaching scores remain unavailable.

### Step 3.4 — Overlapping-window experiment

The TrackNet adapter also accepts an optional `--window-stride` from 1 through
the checkpoint sequence length (eight for the pinned model). For example:

```powershell
.\.venv\Scripts\python.exe shuttle_tracknet.py --video "Videos\videoplayback.mp4" --player-run "out\players-046" --out "out\tracknet-overlap-repeat" --window-stride 4
```

This evaluates complete eight-frame windows every four frames, adds a final
covering window when needed, and averages all observed heatmaps for each frame
before applying the existing 0.5 threshold. It uses a uniform mean; it does not
implement the upstream weighted ensemble. Inputs and scene-local backgrounds
remain isolated by cuts and rejected intervals. Short groups are left unprocessed,
and missing detections remain missing. Omit the option to retain the existing
non-overlap behavior. Summaries record both the window mode and effective stride.

The overlap experiment was motivated by the known 5.100-second development miss;
its results are development tuning, not held-out validation. All proposals remain
unverified. No shot events or coaching scores are enabled.

Across the four development windows, stride four still matches **11/12** labeled
points as both candidates and linked proposals; the 5.100-second miss remains.
The labeled racket region contains no candidate or linked proposal, matching the
default result. This experiment does not justify changing the default. Seventy
tests pass, including overlap averaging before thresholding and isolation across
both gaps and scene boundaries.

Five real overlap runs (`out/tracknet-overlap-001` through `005`) processed
another **1,500 frames**. The additional-window control remains **6/6** linked
point matches. All previews decode to 300 frames, every linked segment stays
consecutive within a scene, and excluded intervals match the default exactly.
There are 960 linked-proposal frames across 52 segments versus the default's
983 across 57; neither count establishes accuracy. All five contact sheets were
inspected with automated review; full trajectories have not been certified.
The comparison is in `out/tracknet-overlap-evaluation/index.html`, with source,
model and output fingerprints in
`validation/tracknet_overlap_evaluation_2026-10-31.json`. Some runs shared CPU time,
so their elapsed times are not a controlled speed comparison. The next validation
milestone is longer continuously annotated shuttle intervals, including difficult
motion and distractors, before accepting trajectories or detecting shot events.

### Step 3.5 — Continuously annotated shuttle intervals

The benchmark evaluator now accepts `sequences`: named inclusive frame intervals
with `source_run`, `start_frame` and `end_frame`. Every frame in each interval must
have exactly one visible or uncertain annotation. Missing frames, overlapping
intervals and annotations outside the declared coverage are rejected. Continuous
evaluation also rejects proposal segment IDs reused across a gap or scene cut.
Existing sparse packets retain their previous evaluation behavior.

`validation/shuttle_continuous_seed_2026-10-31.json` labels **180 native-rate
frames**: opening frames 180–269 (6.000–8.967 seconds) and control-window frames
30–119 (61.000–63.967 seconds). Both intervals cover three seconds of sampled
video, substantially longer than the earlier six-frame point checks. Codex
reviewed all frames in 30 enlarged unmarked temporal sheets, then checked visible
labels in enlarged local crops before evaluating this packet. There are **111
visible points and 69 uncertain frames**. Approximate component/streak centers
were placed visually; some were refined with local bright-pixel measurements.
No evaluated model output or interpolation supplied locations. Earlier sparse
model previews had already been seen, so this is automated development labeling,
not blind, independently reviewed or held-out ground truth. Broadcast repeats,
correlated frames and unknown upstream training-video overlap limit interpretation.

```powershell
.\.venv\Scripts\python.exe shuttle_benchmark.py prepare --seed "validation\shuttle_continuous_seed_2026-10-31.json" --video "Videos\videoplayback.mp4" --out "out\continuous-repeat"
.\.venv\Scripts\python.exe shuttle_benchmark.py evaluate --benchmark "out\continuous-repeat\benchmark.json" --prediction "opening=out\tracknet-001" --prediction "control=out\tracknet-005"
.\.venv\Scripts\python.exe shuttle_benchmark.py evaluate --benchmark "out\continuous-repeat\benchmark.json" --prediction "opening=out\tracknet-overlap-001" --prediction "control=out\tracknet-overlap-005"
```

Per-sequence results distinguish visible frames without proposals from proposals
displaced beyond the point tolerance. They report localization error on visible
proposed frames, longest consecutive matches within one proposal segment, missing
observation runs, and segment changes between adjacent point matches. Uncertain
labels break all matching/missing runs and never become absence labels or false
positives. Matching nearby points does not certify shuttle identity or a physical
trajectory. A displaced point can reflect annotation uncertainty or localization
error as well as a wrong object.

| Continuous development check | Default stride 8 | Optional stride 4 |
| --- | --- | --- |
| Visible point matches at 4 native pixels | 94/111 | 98/111 |
| Opening matches / visible labels | 32/43 | 34/43 |
| Control matches / visible labels | 62/68 | 64/68 |
| Longest same-segment matching run, opening / control | 8 / 24 frames | 16 / 26 frames |
| Visible frames with no proposal | 0 | 0 |
| Visible frames with a proposal beyond 4 pixels | 17 | 13 |
| Uncertain frames, excluded from localization scoring | 69 | 69 |

Tolerance sensitivity matters for tiny blurred objects and approximate labels:
both modes match 62/111 at two pixels and 109/111 at six pixels. At 6.567 seconds,
the default proposal is 42.63 pixels from its label; overlap reduces that error to
7.41 pixels but still fails the four-pixel check. Both modes also miss the
approximate label at 7.567 seconds by more than six pixels. These examples were
inspected with the source and both predictions together. The default remains
unchanged; no trajectory is accepted on the basis of this comparison.

The local packet is `out/shuttle-continuous-benchmark-001`. Review
`out/shuttle-continuous-evaluation/index.html` for two complete 90-frame annotated
clips, error examples, every frame's result and links to unmarked source images.
Green circles are automated labels, cyan squares default predictions and pink
squares overlap predictions. Both clips were decoded back to 90 frames.
`validation/shuttle_continuous_evaluation_2026-10-31.json` records the source-image
packet, label, evaluator and prediction fingerprints plus full four-pixel timelines
and tolerance sensitivity. Seventy-four tests pass, including complete annotation
coverage, missing/displaced/uncertain accounting, gap/cut isolation, all-uncertain
intervals and native video integration. Shot events and coaching scores remain
unavailable. The next milestone is a broader set of complete rally intervals and
independent label checks before trajectory acceptance or shot-event timing.

### Step 3.6 — Complete broadcast-point context and motion diagnostics

`validation/shuttle_rally_seed_2026-10-31.json` annotates every frame from
720.000–725.167 seconds: **156 native-rate frames**, including the service
close-up, wide-view exchanges, a dive and post-point context. This covers a complete
broadcast point, with camera changes at source frames 12 and 155; it is not an
uninterrupted fixed-camera rally. Codex inspected 26 enlarged unmarked temporal
sheets and local crops, including the full-frame bottom strip after the dive.
There are **36 visible approximate locations and 120 uncertain frames**. Earlier
sparse model previews had already been seen. No evaluated prediction or
interpolation supplied labels; this remains automated development review from
the same match, without independent human or held-out validation.

Optional per-frame `view` labels (`wide_court`, `closeup`, `unknown`, or
`unspecified`) now split results by source view. Visible close-up labels stay in
the overall totals; a view breakdown never silently removes difficult examples.
The continuous evaluator records prediction status counts and both inherited
scene changes and annotated view changes. It also measures the error between
predicted and labeled frame-to-frame displacement vectors, using only adjacent
visible observations within the same proposal segment, scene and annotated view.
Uncertainty, missing proposals, segment resets and view changes break this motion
comparison. Matching-run and missing-run statistics also stop at view changes.
These are native image-pixel diagnostics, not physical velocity or contact events.

```powershell
.\.venv\Scripts\python.exe shuttle_benchmark.py prepare --seed "validation\shuttle_rally_seed_2026-10-31.json" --video "Videos\videoplayback.mp4" --out "out\rally-repeat"
.\.venv\Scripts\python.exe shuttle_benchmark.py evaluate --benchmark "out\rally-repeat\benchmark.json" --prediction "rally-720=out\tracknet-002"
.\.venv\Scripts\python.exe shuttle_benchmark.py evaluate --benchmark "out\rally-repeat\benchmark.json" --prediction "rally-720=out\tracknet-overlap-002"
```

| Broadcast-point development check, four-pixel tolerance | Default stride 8 | Optional stride 4 |
| --- | --- | --- |
| Linked point matches across all visible labels | 25/36 | 25/36 |
| Linked matches on labeled wide-court observations | 25/34 | 25/34 |
| Visible frames without a linked proposal | 4 | 6 |
| Visible frames with a displaced proposal | 7 | 5 |
| Longest same-segment matched run | 14 frames | 14 frames |
| Adjacent visible proposed pairs, same segment and view | 28 | 26 |
| Mean displacement-vector error on those pairs | 1.1182 px | 1.1396 px |

The default's four missing observations are two intentional service-close-up
exclusions and two wide-view boundary frames outside the approved player
interval. Overlap additionally loses a blurred observation at frame 54 and
abstains on ambiguous components at frame 110. At two-pixel tolerance both modes
match 21/36 linked points; at six pixels the default matches 31/36 and overlap
30/36. These small approximate-label comparisons do not establish general accuracy
or support changing the default. The 120 uncertain frames, including overlaps and
flight beyond the enlarged court crop, cannot establish full-rally recall,
absence, false positives or verified continuous trajectories.

Review `out/shuttle-rally-evaluation/index.html` for the full 156-frame comparison
video, view/boundary examples and every frame's result with its raw source image.
The video decodes back to 156 frames. Source-label, evaluator, model and prediction
fingerprints, per-view results, full timelines and tolerance sensitivity are in
`validation/shuttle_rally_evaluation_2026-10-31.json`. Re-evaluation of the previous
continuous packet preserves its 94/111 default and 98/111 overlap matches.
Seventy-six tests pass, including motion comparisons across uncertainty, missing
observations, cuts and view changes; visible close-up labels remain counted.
The default is unchanged. No trajectory acceptance, shot-event timing, line
decisions or coaching scores are enabled.

### Step 3.7 — Offline shuttle label correction

`shuttle_annotation.py` prepares a source-only editor and imports corrections
into a **new** benchmark packet. The editor shows the complete source frame,
adjacent frames, existing labels and view type. Zoom or enter native-pixel
coordinates to locate a visible shuttle; keep unclear frames uncertain.
Locations are never copied between frames. No model predictions are displayed.

```powershell
.\.venv\Scripts\python.exe shuttle_annotation.py prepare --benchmark "out\shuttle-rally-benchmark-001\benchmark.json" --out "out\shuttle-label-review-new"
```

Open the generated `index.html` locally. Prepared workspaces also exist at
`out/shuttle-rally-corrected-001/editor/index.html` (156 frames) and
`out/shuttle-annotation-continuous-002/index.html` (180 frames).
Previous/Next, Next uncertain and Next unchecked navigate the packet. Confirm
an unchanged label explicitly; Reset this frame removes its draft correction.
Label editing and confirmation stay unavailable until the current source image
loads. A delayed image from a previous frame cannot unlock the current frame.
Browser drafts are bound to the exact original packet. Download corrections
after entering the reviewer name and explicitly choosing human or automated
review. This is declared provenance, not authenticated reviewer identity.

Save the downloaded correction file at the path used below, then import it:

```powershell
.\.venv\Scripts\python.exe shuttle_annotation.py apply --benchmark "out\shuttle-rally-benchmark-001\benchmark.json" --corrections "out\shuttle-corrections.json" --out "out\shuttle-rally-corrected-new"
.\.venv\Scripts\python.exe shuttle_benchmark.py evaluate --benchmark "out\shuttle-rally-corrected-new\benchmark.json" --prediction "rally-720=out\tracknet-002"
```

The importer rejects changed source packets/images, duplicate or unknown frames,
out-of-bounds points, uncertain labels with locations, contradictory local
negative labels and attempts to change source metadata. It preserves previous
labels and reviewer details in revision history. Existing output directories
are never overwritten. Source images and packet hashes bind each revision.

Verification: 81 Python tests and four JavaScript tests pass. Application-level
JavaScript tests use DOM doubles to cover delayed/failed image loads, navigation,
draft restoration, native-coordinate edits, reviewer requirements and reset;
they do not exercise a browser layout engine.
All 336 copied source images match their original hashes. Live browser testing
could not complete because the local preview connection timed out; visual
layout and browser interaction remain unverified. The editor has not created
any new human reviews or changed the measured model results. Label corrections
do not approve shuttle trajectories, shot events or coaching scores.

Automated follow-up review inspected full, unmarked source frames 18–23 of the
broadcast-point packet. All six retain uncertainty: no reliable shuttle center
could be identified. The versioned correction file is
`validation/shuttle_rally_uncertainty_review_2026-10-31.json`; importing it produces
`out/shuttle-rally-corrected-001/benchmark.json` with the original labels preserved
in history. Both default and overlap predictions still match 25 of 36 visible
labels within four pixels. The updated workspace hashes and evaluation results
are in `validation/shuttle_annotation_followup_2026-10-31.json`. These are six
automated review notes, with no new visible labels or human approvals.

### Step 3.8 — Full-frame temporal context and upper-flight review

Generate unmarked full source frames to inspect shuttle flight above and around
the court. This command preserves every source pixel, including the margins;
it adds frame numbers and source timestamps outside each image. The companion
`context.json` binds the sheet to the exact packet and source-image hashes.
No labels or predictions are drawn, and missing source frames are rejected.

```powershell
.\.venv\Scripts\python.exe shuttle_annotation.py context --benchmark "out\shuttle-rally-corrected-001\benchmark.json" --source-run "rally-720" --start-frame 53 --frames 6 --columns 2 --scale 2 --out "out\full-context-new"
```

Each sheet supports 1–12 consecutive frames, 1–4 columns and 1–3× enlargement.
Existing output directories are rejected. Tests compare the entire image area
against source pixels, check temporal ordering and timestamps, and reject gaps
and invalid requests. All 83 Python tests and four JavaScript tests pass.

Full-frame and enlarged upper-frame inspection identifies a visible rising
streak in frame 57 (721.900 s), above the official against the red board.
Its approximate center is `(331.54, 21.29)` native pixels. The local brightness
refinement used raw frame 53 as background inside a visually chosen region,
before inspecting evaluated predictions. Initial center estimates in frames
55–56 were discarded because official-arm overlap confounds the streak center;
those frames remain uncertain. Frame 58 also remains uncertain. No points were
interpolated, and no human approval is claimed.

The correction is `validation/shuttle_upper_flight_corrections_2026-10-31.json`,
bound to the preceding corrected packet. The resulting local packet is
`out/shuttle-rally-corrected-003/benchmark.json`, with editor at
`out/shuttle-rally-corrected-003/editor/index.html`. It preserves both earlier
review history and the original source images. Review the unmarked sheet at
`out/shuttle-full-context-001/context.png`.

Both model variants have no heatmap detection in frame 57. This increases
visible-label coverage from 36 to 37 frames, with 119 uncertain frames retained;
both still match **25/37** visible labels within four pixels. At two pixels both
match 21/37; at six pixels default matches 31/37 and overlap 30/37. This reveals
a supported miss, not a model improvement. Source bindings, measurement method
and results are recorded in
`validation/shuttle_upper_flight_evaluation_2026-10-31.json`. These remain
approximate automated development labels from one previously seen broadcast.
The inference default is unchanged; verified trajectories, shot events and
coaching scores remain unavailable. Live browser interaction remains unverified.

### Step 3.9 — Activation cutoff experiment

`shuttle_tracknet.py` accepts an optional `--heatmap-threshold` strictly between
zero and one. The default remains **0.5**. The cutoff applies after averaging
all sequence-window heatmaps for a frame; changing it does not change the
checkpoint, inputs, scene/view gates or linking rules. Each output row also
records `heatmap_peak`, the largest averaged activation, or `null` when the
frame was excluded or lacked sufficient sequence context. This activation is
uncalibrated and is not a probability of a correct shuttle detection.

```powershell
.\.venv\Scripts\python.exe shuttle_tracknet.py --video "Videos\videoplayback.mp4" --player-run "out\players-045" --out "out\tracknet-cutoff-trial-new" --heatmap-threshold 0.35
```

The single 0.35 trial was selected to investigate the known upper-flight miss,
before measuring the trial results. This is development tuning, not held-out
validation. Tests check invalid cutoffs before any source/model work, averaging
before the cutoff, native-coordinate decoding, diagnostic peaks and unchanged
gap/view boundaries. All 85 Python tests and four JavaScript tests pass.

Five real trial runs (`out/tracknet-threshold035-001` through `005`) processed
1,500 frames. A further 300-frame real run at the unchanged default reproduces
every previous rally prediction row exactly, apart from the new diagnostic
field. All six previews decode to 300 frames each. Trial exclusions match the
baseline, linked segments stay consecutive within a scene, and every verified
shuttle position remains null. All five trial contact sheets and selected
changed-result examples were inspected with automated diagnostic review.

| Linked point matches, four native-pixel tolerance | Default 0.5 | Trial 0.35 |
| --- | --- | --- |
| Original development points | 11/12 | 11/12 |
| Additional control points | 6/6 | 6/6 |
| Continuous development packet | 94/111 | 95/111 |
| Revised broadcast-point packet | 25/37 | 27/37 |

These packets overlap; their counts must not be added as independent samples.
The known local racket negative remains clear in both variants. No broad
false-positive rate is established from the uncertain frames.

The target upper-flight observation at frame 57 has peak activation `0.423221`.
The trial locates it at `(331.875, 17.5)` pixels, about 3.8 pixels from the
approximate development label: a match at four pixels, but not two. Other
results regress: the opening interval drops from 32/43 to 31/43, and extra
candidates make rally frames 14–16 ambiguous for linking. At two pixels,
continuous matches fall from 62/111 to 60/111 and rally matches from 21/37 to
20/37. At six pixels they fall from 109/111 to 108/111 and 31/37 to 29/37.
These mixed results do **not** justify changing the default.

Review `out/tracknet-threshold-evaluation-001/index.html` for all five trial
clips and 17 changed labeled-result comparisons with unmarked source images.
Comparison titles sit outside the full frame so upper-flight points remain
visible. Exact model/output bindings, tolerance sensitivity, target diagnostics
and the decision to retain the default are recorded in
`validation/tracknet_threshold_evaluation_2026-10-31.json`. The trial remains
opt-in. No verified trajectories, shot events or coaching scores are enabled.

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
