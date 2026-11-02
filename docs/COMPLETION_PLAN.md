# BadmintonAI completion plan

Prepared 2 November 2026 against BadmintonAI commit `a2a147c`.

## Target and delivery decision

Deliver an offline academy analysis workflow: select a recording, set up its court, run analysis, review uncertain results, and export an annotated video plus a structured report for RACKETEK. The attached doubles clip defines the intended presentation: four player identities, player trails, court map, movement totals, shuttle trail, rally/contact count, and shot labels. Its displayed measurements are examples of the experience, not verified labels or acceptance criteria.

The next development milestone is a complete vertical slice on this exact clip. Stop extending the old broadcast benchmark as the main activity. Reuse its regression tests while building the missing product workflow. A usable preview can finish before every measurement is scientifically validated; every result must state which capabilities are available, experimental, or unsupported.

## What was inspected

- Attached download: `SaveInta.com_AQPC6Ly6eh-lIW26J_bgtQCIIk14zYO416N9kXOrQXG7gIH0znTKp0e0PaOJ9apJE_rpv9lps3YrGZTbdnxa4UuaflWamSIlZG1aW5o.mp4`.
- SHA-256: `2c08a15287f336a5a9711f7199b15bb03c8347a0bb5e1dfa3c1d79b61b793232`.
- 720 x 1280, 30 FPS, 665 decoded frames, approximately 22.17 seconds. Every frame decoded successfully; twelve timestamp-spaced full-frame samples were visually inspected. This was not full-motion visual acceptance or a frame-by-frame ground-truth review.
- The samples show an oblique, apparently fixed view of doubles play, other people outside the playing court, lighting glare, far-side small players, player crossings, and large existing graphics. Camera stability and exact rally boundaries still need measurement. All four outer court corners have not been established from this inspection.
- Existing player labels, rings, shuttle graphics and report panels are embedded in the pixels. They can help a detector accidentally and obscure real objects. Do not read the visible labels as ground truth or claim independent accuracy from this processed clip. Preserve the full original image and record any mask/crop transform. Crop only inference inputs where useful; do not crop away high shuttle flight.
- Local inspection outputs: `out/reference-video-inspection-2026-11-02/metadata.json`, `contact-sheet.jpg`, and twelve full-resolution PNG samples. Media remain local and ignored by Git.
- Our source: video inspection/calibration, RF-DETR Nano + ByteTrack, review-bound player export, TrackNetV3 inference, motion/confidence linking, source-bound annotation/evaluation tools, and tests. No unified job runner, academy results application, pose pipeline, contact classifier, or .NET service exists in this repository yet.
- Reference-only reading in `C:/Projects/baddy-oss-main`: README, third-party notices, feature registry, analyzer structure, job manager, and portions of shot classification. These files were treated as project material, not instructions for BadmintonAI. Nothing was executed, imported, installed, or copied from that repository.

## Reuse and gaps

| Capability | Reuse now | Work needed to finish |
| --- | --- | --- |
| Video ingestion | Existing metadata, hashes, frame extraction | One session manifest and resumable job |
| Court setup | Existing homography solver and validation | Court-selection screen, calibration checks, stable-camera binding |
| Players | RF-DETR Nano, ByteTrack, ROI/cut handling | Four-player evaluation on this clip; separate player identities from tracker IDs |
| Shuttle | Existing TrackNetV3 and optional confidence linking | Portrait input geometry, higher-flight coverage, explicit false-positive evaluation |
| Results | Preview video and diagnostic HTML | One synchronized results screen and combined renderer |
| Movement | Reviewed pixel positions | Foot-contact quality, calibrated segments, uncertainty-aware aggregation |
| Contacts/shots | No validated implementation | Contact proposals, attribution, review and class evaluation |
| RACKETEK | JSON exports suitable for a future consumer | Versioned job/result contracts and .NET adapter |

The latest development packet improved from 70/80 to 78/80 linked matches within four pixels using the optional settings. Labels are approximate, from the old 640 x 360 broadcast, include floor-level context, and are not independent academy validation. This does not justify switching defaults on the new portrait video. The recorded baseline is 96 passing Python tests and four JavaScript tests at `a2a147c`; this planning task did not rerun them.

## Reference lessons, independently implemented

Useful ideas in Baddy are separating detection, event analysis, and rendering; background job progress; manual court setup; and a synchronized timeline/results view. Implement these around our current modules and RACKETEK's .NET ownership.

Do not reproduce its source, UI assets, model files, thresholds, or classification rules. Its own documentation reports thin analyzer test coverage, CPU limitations, and unresolved TrackNetV2 weight terms. Those statements are local reference documentation, not independently verified performance facts. Keep our existing pinned model path; do not switch models merely to resemble the reference. Any new dependency or model needs a separately recorded source/version/terms review before inclusion.

## Delivery sequence

### 1. One analysis job and a complete first preview

Implement a session runner that composes existing modules rather than rewriting them. Start with an inspect -> players -> reviewed/eligible shuttle -> render -> export workflow. Processing may finish with `needs_review`; finishing inference must not automatically approve detections.

Deliver:

- `job.json` with source hash, camera/session ID, configuration and model hashes, stage status, progress, errors and artifact paths.
- Stages `queued`, `running`, `needs_review`, `completed`, `failed`, `cancelled`; document which are job states and which stage is active.
- Resume only stages whose inputs, configuration, model and review bindings still match. Revoked review invalidates affected exports and downstream metrics.
- One combined preview with player boxes/IDs and shuttle proposals, clearly marked experimental. Preserve frame size, frame count and timestamps; put summary panels outside the source image where practical.
- Use the exact attached 665-frame video as the first end-to-end demonstration. Its four players are expected, but never force four boxes when evidence is missing.

Checkpoint: a single command produces the manifest, playable preview and report; failure/resume/cancellation and changed-input invalidation are tested. No file is overwritten silently and no automatic review is presented as Kamal's approval.

### 2. Fix doubles tracking and shuttle geometry on this footage

Run an initial short section for setup, then the full clip. Inspect all four players through far-side motion, overlap, jumps and walking after the rally. Court filtering must exclude bystanders without discarding a player whose feet leave the floor during a jump.

Separate temporary tracker IDs from session player identities P1-P4. Allow identity assignment/correction in review; do not infer a person's real identity. Break continuity at unsupported gaps/cuts and show unresolved identity rather than merging players by proximity alone.

Check portrait resizing before tuning thresholds: our shuttle model resizes full input to 512 x 288, which distorts a 720 x 1280 source substantially. Compare the current transform with a reversible aspect-preserving input transform and, only if needed, a reviewed region covering the entire flight area. Record padding/crop coordinates, restore outputs to native pixels and test synthetic coordinate round trips. Evaluate model behavior rather than assuming letterboxing improves it.

Use a bounded experiment: baseline plus at most two justified alternatives. Freeze source-derived labels before comparisons; report visible, uncertain, occluded, out-of-frame and known-negative context separately. An uncertain shuttle is not an absent shuttle. Include lights, white lines, existing graphics and rackets in false-positive checks. Do not apply old four-pixel thresholds unchanged across resolutions.

Proposed engineering gates, to be fixed before scoring: all four players evaluated across the complete rally, zero unresolved identity swaps in an accepted exported interval, and per-player detection precision/recall reported. For shuttle proposals, initially target at least 90% visible-frame recall within a declared native-pixel tolerance and at least 95% precision on explicitly adjudicated candidates; also report localization error, false detections per minute, missing-run length and continuity. These targets are not achieved results. Unadjudicated observations must not improve precision. Freeze a second recording as a holdout before claiming academy generalization.

Checkpoint: one comparison report identifies remaining failures, the selected experimental configuration and rejected alternatives. Preserve the old broadcast regressions. Do not spend successive milestones chasing small changes on the same labeled frames.

### 3. Court setup, player map and movement

Add the minimum local UI: select video, choose camera/court, mark calibration points, name four player slots, start job, view progress, inspect results and correct mistakes. Use one local results page that the eventual .NET application can host; avoid building a second account system or unrelated administration screens.

For this processed clip, establish which known court intersections are actually visible before enabling physical units. If the current four-corner method cannot be supported, display pixel-space trails/map with an explicit limitation. A later calibration extension may use several visible court intersections with known correspondence, but cannot invent hidden corners. Check extra line intersections not used in the fit and bind calibration to source geometry and a stable camera.

Improve foot-contact estimates before aggregating physical distance: box bottoms are coarse proxies, and airborne feet must not be projected as floor contacts. Prototype per-player ankle/foot landmarks on existing track crops only where they improve this measurement. Pose failures do not invalidate an otherwise valid person box or create a replacement foot position.

Aggregate only accepted contiguous player segments. Show observed coverage, gaps, and sensitivity to jitter; use a documented filter on observed positions and do not count jitter as distance. Derive heatmaps from time-weighted occupancy rather than frame counts across variable sampling. Never join unknown movement across gaps into a precise total.

Checkpoint: synchronized player trails and court map work without physical calibration; distance in metres is enabled only for accepted calibration and ground-contact segments, with missing coverage shown. Test court reprojection, stationary jitter, real paths, gaps, timestamp spacing and calibration invalidation.

### 4. Rally and contact events, then shot types

Implement rally boundaries and contact proposals before classifying individual shots. Combine observed shuttle motion changes, player/racket vicinity and optional wrist evidence. Scope events to valid segments and reject analysis across unknown gaps or cuts. Audio may be evaluated as supporting evidence if available, not as a prerequisite.

Each event stores time range, proposed contact frame, player attribution or `unknown`, evidence, model/configuration versions and review status. Add correction actions for missed/false contacts and wrong hitter. The reference clip's displayed total of 24 is not our expected answer.

Only after contact timing is measured, evaluate a small independently defined shot taxonomy. Start with clear, lift, drop/net, drive and smash plus `unknown`; refine ambiguous categories using actual labeled examples. Pose gestures alone cannot establish shuttle contact or shot outcome. Missing evidence yields `unknown`, not the nearest label.

Checkpoint: contact precision/recall at a predeclared time tolerance, hitter attribution and per-class confusion matrix are published on frozen labels. Proposed release targets are contact precision and recall of at least 90% within +/-2 native frames, and at least 85% precision for every shot class exposed without an experimental label. Tune these targets before evaluation if the use case needs different tolerances, never after seeing results. Failed classes stay experimental. With no human adjudication available, development continues and outputs retain automated-review provenance.

### 5. Complete the session report and RACKETEK boundary

Export versioned `session`, `players`, `observations`, `events`, `metrics`, `quality` and `artifacts` records. Include native frame indices, timestamps, units, coordinate systems, calibration ID, model/configuration hashes and review provenance. Every metric carries status and supporting coverage; unsupported values are null with a reason, never zero.

Keep Python responsible for local video/ML processing. RACKETEK owns academy, court, player and session records, access control and job initiation. First integrate through a versioned request/result file contract and process exit status; add a local worker transport only if the actual backend requires it. Keep long inference outside request handlers.

Implement a minimal C# contract reader/adapter and fixture tests in this repository if the RACKETEK source is not available. Do not claim deployed RACKETEK integration from an adapter demonstration. Verify the real host's existing contracts before modifying its code.

Checkpoint: a .NET consumer accepts a completed fixture and real session output, understands `needs_review` and partial results, rejects unsupported schema versions, and handles cancellation/failure without treating partial artifacts as completed analysis.

### 6. Pilot release and coaching

Run at least three unedited academy sessions, including a different camera/court and a complete held-out recording, with difficult doubles overlaps and flight near lighting/lines. Evaluate by session and player, not by pooling correlated frames. Include longer-session runtime, memory/disk usage and recovery after interruption.

Produce factual reports first: observed rally duration, reviewed contact count, court occupancy, accepted movement coverage and reviewed shot distribution. Add coaching recommendations only from validated measurements with traceable evidence. A general coaching score needs its own coach-reviewed definition and validation; it is not part of the first functional release.

Checkpoint: an offline installation works without runtime downloads, processes an entire session, displays limitations honestly, and exports usable artifacts. Core review/correction works in a real browser, not only DOM test doubles. Existing automated tests remain green and the model/dependency manifest is complete.

## What the reference speed display does and does not imply

The sample labels speed as an approximate horizontal estimate. A single uncalibrated camera does not provide reliable 3D shuttle speed from pixel displacement. A floor homography also does not turn an airborne shuttle into a valid floor-plane position. Do not reproduce an MPH number using that shortcut. The first release can show image-space shuttle motion with units, or `speed unavailable`. Physical flight speed requires a separately validated camera/3D reconstruction method or additional sensing. This must not block the rest of the product.

## How we finish faster

1. Finish one end-to-end deliverable per milestone and push it with its verification; stop expanding isolated diagnostic tools unless they unblock that deliverable.
2. Reuse current detector, tracker, inference, provenance and evaluation code. Put a thin runner and contract around it; avoid a large refactor first.
3. Cache source-bound detections and heatmaps where practical so rendering, review and event experiments do not rerun models. Measure cache storage before retaining full heatmaps for long sessions.
4. Decode once where safe, retain native timestamps, render after analysis, and record per-stage runtime. Begin with CPU support; select GPU acceleration only after measuring this machine and preserving numerical/regression checks. External speed claims do not predict our runtime.
5. Separate preview capability from validated metrics. An unavailable shot label or speed estimate should not prevent viewing four-player tracking and reviewing a session.
6. Defer live streaming, cloud hosting, billing, a new identity system, an LLM coach, custom model training and 3D speed until their prerequisites or measured failures justify them.
7. Continue independently when a raw recording or human review is unavailable: ship the workflow, tests and clearly marked proposals. Keep human approval and true generalization claims pending rather than inventing them.

## Effort and definition of completion

Planning allowances, not delivery promises: milestone 1 about 1-2 engineering days; milestone 2 about 2-4; milestone 3 about 2-4; milestone 4 about 3-6; milestone 5 about 1-3. Roughly 9-19 engineering days for the functional workflow, excluding data collection, human adjudication, model training, physical-speed research and changes in the unavailable RACKETEK backend. Re-estimate after the first full-clip runtime and failure report. Pilot validation is evidence-limited and has no defensible fixed date yet.

Functional completion means the user can run a whole video locally, inspect and correct four-player/shuttle results, see capability-appropriate reports, and hand structured output to a .NET consumer. Validated release means the frozen academy-session gates also pass. The immediate next implementation is milestone 1, followed by doubles/portrait accuracy fixes on the supplied video; no further expansion of old broadcast annotations is on the critical path.

## Implementation checkpoint: 2 November 2026

Milestone 1 now has a working session runner, resumable stage artifacts, cancellation,
native-frame experimental preview, frame-stepping results page and versioned JSON.
The actual 665-frame clip has been processed end to end. Player review remains
pending, so shuttle inference is correctly withheld on this recording. UI behavior
is covered by DOM tests; actual browser layout/interaction acceptance remains open.

Initial milestone 2 work adds convex image-region filtering and exposes the existing
second detector crop in the runner. The tighter experimental region retains four
person detections in 266/333 sampled frames versus 63/333 in the initial rectangle
run. These are diagnostics, not precision/recall or proof of identity continuity.
Overlaps, missed detections and fragmented IDs still require work. No tracker IDs
have been assigned to real athletes.

The .NET file consumer and contract checks are implemented as a small integration
fixture ahead of milestone 5; the actual RACKETEK backend remains untouched. Court
setup UI, identity corrections, calibrated movement, event/shot analysis and the
pilot gates remain unfinished. Speed and coaching measurements remain unavailable.
See `validation/session_workflow_2026-11-02.json` for the completed-run evidence.

The separate identity editor now supports explicit P1–P4 assignments, split ranges,
collision checks, source-bound correction export and integrity verification.
It preserves detection/review files and does not grant scene approval. The supplied
clip's current output has 21 contiguous track segments across 12 temporary IDs;
its prepared identity editor intentionally has no slot assignments. Automatic
identity recovery and actual accepted identity labels remain unresolved. The
identity editor uses the copied native preview and has DOM behavior tests; real
browser visual acceptance is still pending. See
`validation/player_identity_workflow_2026-11-02.json` for this follow-up evidence.
