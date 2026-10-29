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

Check `tracking_summary.json` for `diagnostics.class_id_histogram`, `raw_person_detections`, `court_filtered_person_detections` and `boxes_with_track_id`. Inspect the two contact sheets. This code correction is verified with mock-model tests, **not yet with RF-DETR inference on the user's PC**. Low-resolution broadcasts and camera cuts still cause genuine detection misses and ID switches even with the correct label mapping.

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
