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

It does **not yet** detect players or shuttles, infer shots, or produce performance scores. Those are subsequent verified milestones.

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

`--roi` is an **optional normalized rectangle** `(left, top, right, bottom)` in the range 0–1. It keeps detections whose approximate *foot position* is inside the rectangle, helping exclude spectators/officials. The example coordinates suit the sample broadcast's first few seconds, **not all camera setups**. Omit `--roi` initially for a new fixed-camera video, then inspect its generated review sheet and add an appropriate region.

Generated files:

- `out/players-001/tracked_preview.mp4`: annotated video, played at the original time scale but sampled to 3 FPS in this example
- `out/players-001/player_tracks.jsonl`: one row per sampled frame, with boxes, approximate foot pixels, confidence, and track IDs
- `out/players-001/tracking_summary.json`: model versions, diagnostic counts, warnings, and file names
- `out/players-001/review_contact_sheet.jpg`: nine-frame annotated QA sheet (when enough frames are available)
- `out/players-001/review_frames/`: individual annotated review images

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
