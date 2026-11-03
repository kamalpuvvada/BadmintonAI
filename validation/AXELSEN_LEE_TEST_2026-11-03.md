# Axelsen / Lee real video test

Tested the complete user-supplied portrait broadcast clip, SHA-256
`7c9ab079c59488a8ab7f69b1015a608e00fc8ee37ff6753e76523608e51a40b2`.
Source: 1080 × 1920, 60 FPS, 2,256 decoded frames (37.6 video seconds).
The container duration including audio is about 37.686 seconds.

Inference used RF-DETR Nano + ByteTrack at 15 sampled FPS, 564 samples,
four CPU threads. The detector/tracker stage took 163.121 seconds.
Inference code commit: `3b9c9c62cf271cc1e0acec114820696694201d82`.

| Diagnostic | Result |
|---|---:|
| Court-view samples before 35.5 s | 533 |
| Exactly two boxes in that interval | 516 / 533 (96.8%) |
| One box in that interval | 16 |
| Three boxes in that interval | 1 |
| Temporary IDs observed in that interval | T0, T1 |
| Sparse visible torso points covered | 12 / 12 |
| Native preview frames decoded | 2,256 |

At six source-only spot checks (0, 8, 16, 24, 32, 35 seconds), the near
white-shirt player and far dark/red-shirt player each have a separate box.
After the initial unconfirmed frame, the inspected anchors match T0 and T1,
respectively. This is encouraging, but does not verify every intervening
frame, measure detection accuracy, or certify identity continuity. The source
scoreboard/names were not used as verified track-to-athlete assignments.

## Failure found

The broadcast changes to a close-up around 35.52 seconds. Automatic scene
detection reports no cuts (one scene), so it fails this example. The retained
court crop then admits background people. From 35.6 seconds onward, 26 of 30
samples contain more than two boxes; the full run produces 12 temporary IDs.
The approximate transition sample is kept outside both interval summaries.

Tracking review remains pending and shuttle processing is withheld. No shot
types, winners, forced errors, physical speed or coaching scores are approved.
No production defaults or source code were changed for this test.

Next fix to evaluate: detect the camera/view change using the active broadcast
picture rather than allowing portrait padding and static title graphics to
dilute the change signal. Test the proposed gate against both real clips and
ordinary camera pans before accepting it.

## Local results and reproduction

- Playable result: `out/axelsen-lee-session-001/test-result.html`.
- Full native preview: `out/axelsen-lee-session-001/render-001/preview.mp4`.
- Visual summary: `out/axelsen-lee-session-001/test-highlights.jpg`.
- Full diagnostic counts, bindings, hashes and sparse anchors:
  `validation/axelsen_lee_video_test_2026-11-03.json`.

The complete source is preserved in the preview below a separate 64-pixel
header. Proposals are held between sampled observations, with no interpolation;
they are not fresh detections on every native frame. Detector and tracked
contact sheets were inspected. Local media remains ignored and is not uploaded.

With `$clip` set to the original full Downloads path:

```powershell
.\.venv\Scripts\python.exe session_runner.py run --video $clip --out out\axelsen-lee-session-001 --camera broadcast-portrait-01 --sample-fps 15 --expected-players 2 --roi '0,0.51,1,0.86' --detect-roi '0,0.40,1,0.86'
.\.venv\Scripts\python.exe validation\summarize_axelsen_lee_2026_11_03.py
```

The evidence script requires the local run, source inspection JPEGs and
`out/axelsen-lee-inspection/anchor-seed.json`. Their source frame indices,
coordinates and hashes are recorded in the versioned JSON. Anchor seed points
were selected from source images before inspecting model predictions.
