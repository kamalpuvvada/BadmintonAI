# Real portrait doubles video test

Tested the complete local 22-second Downloads clip identified by SHA-256
`2c08a15287f336a5a9711f7199b15bb03c8347a0bb5e1dfa3c1d79b61b793232`.
Code at inference: `99d55e1fa60b5caa0e1b215fbea7b89bdc3d595d`.
This is a processed reference clip: embedded shot, speed, distance and P1–P4
graphics are not independent labels or BadmintonAI measurements.

| Diagnostic | Existing 15 FPS | New 30 FPS |
|---|---:|---:|
| Sampled frames | 333 | 665 |
| Frames containing exactly four boxes | 266 (79.9%) | 525 (78.9%) |
| Unique temporary IDs | 12 | 14 |
| Contiguous ID tracklets | 21 | 27 |
| Detector/tracker stage, CPU seconds | 148.851 | 306.222 |
| Visible torso anchors covered | 23/23 | 23/23 |

These are diagnostics, not detection accuracy or identity metrics. An ID
tracklet here is a contiguous sequence of sampled rows containing that ID;
it is not a verified actor trajectory. The six sparse anchor frames were
inspected by Codex, not Kamal, and are not blind or held-out labels. One
partially occluded anchor was excluded. Baseline sample alignment is
approximate; see the machine-readable method and per-anchor candidates.

## Decision and remaining errors

Keep the 15 FPS setting. Doubling the sampling rate alone did not establish
an improvement. No detector/tracker defaults or athlete assignments changed.
Both runs retain missed/merged detections during far-side overlaps and
unstable temporary IDs. At the sparse inspected anchors, the far dark-shirt
player's proposed ID changes from T3 to T8 at 15 FPS and from T3 to T6 to T10
at 30 FPS. This supports investigating identity association, but does not
measure the exact failure time or prove which intervening assignments are
correct. Near-player IDs also change late in the clip.

The new full-frame preview decoded successfully: 665 frames, source image
preserved below a separate header. Player review remains pending; the runner
correctly withheld shuttle processing because no player intervals are accepted.
No shot, point, error attribution or coaching results were granted.

Next targeted experiment: continuously label a short overlap interval, then
compare duplicate/merged box handling and ID association on those same frames.
A higher four-box count alone must not qualify a change as verified.

## Local evidence and reproduction

- Baseline: `out/academy-session-004`.
- New real inference: `out/academy-session-005`.
- Visual comparison: `out/academy-session-005/test-comparison/index.html`.
- Full preview: `out/academy-session-005/render-001/preview.mp4`.
- Versioned evidence: `player_fps_comparison_2026-11-03.json` and
  `player_anchor_seed_2026-11-03.json` in this directory. These bind source,
  model, configurations, observations and raw anchor image hashes.

Run the real pipeline with the local source path in `$clip`:

```powershell
.\.venv\Scripts\python.exe session_runner.py run --video $clip --out out\academy-session-005 --processed-overlay --sample-fps 30 --expected-players 4 --court-polygon '0,0.64;0.55,0.535;1,0.535;1,1;0,1' --detect-roi '0.00,0.38,1.00,1.00' --secondary-detect-roi '0.38,0.45,1.00,0.64'
```

With both local runs and the raw PNGs in
`out/academy-native-fps-test-2026-11-03`, regenerate and check the evidence:

```powershell
.\.venv\Scripts\python.exe validation\compare_player_fps_2026_11_03.py
```

The raw PNGs are native source frames decoded with OpenCV at the indices in
the anchor seed. The comparison checks their hashes, identical source/model
bindings, configurations differing only in sample FPS, one-to-one anchor
matching and full preview decoding. Local videos and generated images remain
ignored; the source clip and model weights are not uploaded to GitHub.
