# Shot, rally and point analysis: implemented workflow

This is an offline review workflow with experimental automatic suggestions.
It can publish reviewed shot/outcome records and factual reports. It does not
claim validated automatic recognition or enable coaching scores.

## Use the Windows app

1. Double-click `Start-BadmintonAI.cmd`, analyze a recording or resume its saved job.
2. Select **Review rallies and shots**. The editor opens the latest valid analysis
   revision for that job, or creates a new source-bound workspace.
3. Set player slots' sides, optional scoring format and any known initial scores.
   Slots are reviewer-chosen; original video labels and temporary T IDs are not
   automatic athlete identification. Leave unknown scores blank.
4. Mark rally bounds, kind, coverage, game, server/receiver and review status.
   Mark replays, practice, incomplete ends and lets explicitly.
5. Add contacts or missed attempts using native frame indices. Record hitter,
   shot family, direction, execution and evidence. Shot class and contact each
   have their own review status; a missed attempt is not counted as a contact.
6. Record the point winner and terminal evidence separately from finish attribution.
   Choose direct winner, forced error, unforced error, other fault or undetermined.
   Forced/unforced acceptance requires an explicit skill/context rubric.
7. Link final contact, terminal error, direct winning shot and preceding pressure
   shot separately. For smash -> failed block, the block can be the terminal error
   while the smash is the preceding pressure shot; this is not two point awards.
8. Enter reviewer name/type and download changes. Select **Apply shot/outcome
   corrections** in the launcher. The backend validates the entire revision and
   opens its recomputed report. Invalid/stale changes do not replace the base.

Drafts are stored locally against the exact base hash. Undo/redo operates on the
current draft; published revisions retain the parent JSON and correction bytes.
Removing a contact/boundary invalidates dependent decisions. A class-only edit
preserves a separately confirmed point; changing the error rubric requires new
forcedness review. Changing side membership clears stale outcome assignments.

Reports include a per-rally “which shot won or lost the point” table, actual
contact/attempt counts, direct-winner and terminal-error rates with denominators,
player error/pressure contributions, unknown responsibility, annotated score
timeline and linked video clips. The report is the last validated revision;
local edits are not silently included. A completed review snapshot is not a claim
that the whole recording, every rally or the model is accurate.

## Automatic suggestions

**Contacts:** new workspaces use existing current, hash-bound shuttle output
only when player-review eligibility exists. Five consecutive observed points,
a direction-change cue and player-box proximity can propose a contact. Gaps,
cuts and link segments are not bridged. Nearby T IDs are not asserted as hitter
identities. Candidate intervals are fragments, not automatically complete rallies.
No accepted point can be exported from a fragment until its coverage is reviewed.

**Point outcomes:** the launcher's outcome button can suggest the opposing side
from an explicitly annotated terminal net/out/body/service-fault execution. It
does not detect those faults from video, prove a line call or confirm the point.
It preserves accepted outcomes and never infers forcedness from shot name.
Misses/disappearances without terminal evidence do not produce a point guess.

**Shot classes:** a locally trained development model may propose classes from
short observed shuttle windows. Low score/margin returns unknown. Existing
accepted classes/outcomes are preserved. No pretrained academy shot classifier
is supplied, no model is downloaded by this workflow, and no real classifier
was trained from the supplied clip's generated overlays.

The current supplied-video workspace is
`out/academy-session-004/match-review-002/index.html`. Its player review remains
pending, so automatic contact cues are unavailable and its event ledger starts
empty. Zero annotations do not mean zero actual shots. This is deliberate
abstention, not an automatic analysis accuracy result.

## Command-line equivalents

```powershell
.\.venv\Scripts\python.exe contact_proposals.py --job out/academy-session-004 --out out/match-workspace-new
.\.venv\Scripts\python.exe match_analysis.py apply --job out/academy-session-004 --base out/match-workspace-new --patch path/to/match-analysis-corrections.json --out out/match-reviewed-new
.\.venv\Scripts\python.exe match_analysis.py verify --job out/academy-session-004 --out out/match-reviewed-new
.\.venv\Scripts\python.exe match_analysis.py suggest-outcomes --job out/academy-session-004 --base out/match-reviewed-new --out out/outcome-suggestions-new
```

Every output is new; old evidence is retained. Media can share an immutable local
hard link between revisions on a supported filesystem, otherwise it is copied.
Treat preview files as immutable. Manifests bind their exact bytes; modifying any
linked copy invalidates every snapshot sharing those bytes.

## Train and evaluate a development shot baseline

Collect accepted contact/shot labels first. Feature export requires a reviewed
ledger and current eligible shuttle observations; no interpolation fills missing
windows. Reports record skipped uncertain/missing contexts.

```powershell
# Export from at least two different training recordings and a separate held-out recording:
.\.venv\Scripts\python.exe shot_model.py export --job path/to/train-job-1 --analysis path/to/reviewed-analysis-1 --split train --out out/train-1-features.json
.\.venv\Scripts\python.exe shot_model.py export --job path/to/train-job-2 --analysis path/to/reviewed-analysis-2 --split train --out out/train-2-features.json
.\.venv\Scripts\python.exe shot_model.py export --job path/to/test-job --analysis path/to/reviewed-test-analysis --split test --out out/test-features.json
.\.venv\Scripts\python.exe shot_model.py combine --input out/train-1-features.json out/train-2-features.json out/test-features.json --out out/shot-dataset-new.json
.\.venv\Scripts\python.exe shot_model.py train --dataset out/shot-dataset-new.json --out out/shot-model-new
.\.venv\Scripts\python.exe shot_model.py propose --job path/to/job --analysis path/to/analysis --model out/shot-model-new --out out/class-suggestions-new
```

The NumPy baseline uses ten normalized image-motion features, fixed optimization
settings and abstention thresholds. The default minimum is 20 training examples
per class, at least two training source videos, and a disjoint test source. These
minimums make the development exercise runnable; they are not evidence that the
sample is adequate for deployment. Duplicate physical contacts, nonfinite data,
unknown feature versions and source/session leakage across splits are rejected.
Automated test labels require explicit `--allow-development-labels`; they never
become human labels. Re-encoded excerpts still need grouping by their original recording; file hashes alone do not prove independence. Scores are uncalibrated. Every model remains development-only
even when its numerical test metrics are high. Synthetic separability tests are
not badminton accuracy evidence.

## Compare annotations and proposals

```powershell
.\.venv\Scripts\python.exe analysis_evaluation.py --job path/to/job --prediction path/to/prediction-analysis --truth path/to/adjudicated-analysis --out out/analysis-evaluation-new.json
```

Use `--truth-job` if the reference snapshot belongs to another job of the same
source recording. Source/timebase, side slots and skill rubric must align. Rally
IDs must share the annotation task's namespace. Reference labels require human
review unless an explicitly marked development fixture is used.

Evaluation uses order-preserving one-to-one contact matching, maximizing matches
before minimizing time error. Duplicate predictions reduce precision; missing
contacts reduce recall. Unknown shot/point predictions count as failures where
the reference is known, with coverage reported. Temporally uncertain reference
events are not known negatives. It reports contact timing, hitter accuracy,
shot confusion/macro F1, point accuracy/coverage, error-player and pressure links,
and finish-label agreement/kappa. No label variability yields undefined kappa;
independent raters and reference completeness are not assumed or authenticated.
No evaluation automatically approves release or coaching.

## Score and contract limits

The score engine supports explicit custom target/margin/cap/best-of profiles,
unknown initial scores, score anchors, lets, separate administrative points,
game/match endings and correction consistency. Unknown rallies and unconfirmed
no-point events break continuity. Duplicate/ambiguous same-time awards and play
after a known game/match ending are rejected. Exact timing assumes a constant frame rate; variable-frame-rate input needs separate timing validation. Service sequence, official rule
profiles, scoreboard OCR and recording completeness are not yet validated.

Analysis exports are a separate **2.0.0** contract (`analysis.json`, `report.json`,
preview, editor, integrity manifest and optional parent/correction files). Existing
1.0.0 session results keep their meaning and unavailable coaching fields.

```powershell
Push-Location integration/dotnet
dotnet run -- --analysis ../../out/academy-session-004 ../../out/academy-session-004/match-review-002
Pop-Location
```

The .NET reader verifies current job/result identity, analysis artifact hashes,
native frame count, exact source preview, player-observation hashes, reviewer
provenance and nonnegative reviewed counts. Its import is a local integration
fixture, not a deployed RACKETEK endpoint or authenticated coach-review service.

Remaining release work includes actual browser/desktop acceptance, reviewed
multi-session academy labels, reliable tracking/hitter/contact/class recognition,
terminal-event evidence, coach rubric agreement, calibration-dependent maps and
movement, service/scoreboard validation, held-out pilots and real backend hosting.
Coaching scores and physical shuttle speed remain unavailable.
