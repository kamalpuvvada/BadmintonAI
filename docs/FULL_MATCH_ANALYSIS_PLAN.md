# Full match, shot and point analysis plan

Prepared 2 November 2026. Extends `COMPLETION_PLAN.md`; this is planned behavior,
not a claim that the current application already recognizes shots or errors.

## Product outcome

For every rally, explain:

1. Who served and received, which players hit each shot, and what each shot was.
2. Where the players were and which direction/area the shuttle was played toward.
3. Which side won the point, what observable event ended play, and what evidence supports the decision.
4. Whether the finish was a direct winner, a forced error, an unforced error, a different fault, or unresolved.
5. Which final attacking shot and earlier sequence preceded the result, without presenting a tactical hypothesis as proven causation.
6. What patterns recur across a player, pair, game and session, with counts, coverage and linked video examples.

The user should be able to click any reported winner, error or tactical pattern
and watch the supporting rally, inspect the event frames and correct the label.
Start with both singles and doubles. In doubles, score belongs to a side; player
credit or responsibility is recorded separately and can remain unknown.

## Current starting point and dependencies

The repository currently provides offline inspection, experimental person and
shuttle proposals, source-bound player scene review, identity correction, a
session runner, preview/results page, Windows launcher and a .NET result reader.
It does not yet produce validated contact events, shot labels, rally outcomes,
line decisions or coaching scores. The supplied processed clip still has ID
fragmentation and missed/overlapping detections; its embedded labels and counter
are not independent ground truth. A short edited demonstration cannot establish
whole-match scorekeeping or academy generalization.

Maintain separate capability gates: identity, camera geometry, shuttle visibility,
contact timing, hitter, shot class, terminal event, scoring, and error attribution.
An accepted player box does not automatically validate all downstream fields.
Conversely, a manually confirmed point winner can be recorded when shot type is
unknown; a known shot must not imply a known rally winner. Show partial analysis
instead of making every report depend on the hardest capability.

## The result model: facts, judgments and tactical hypotheses

| Layer | Examples | Required treatment |
| --- | --- | --- |
| Recorded observations | Frame/time, visible contact, player box, visible shuttle endpoint | Source-bound evidence; unknown gaps preserved |
| Rule-based result | Awarded side, score update, let/no-point | Configured rules and confirmed terminal/official evidence |
| Shot interpretation | Smash, clear, lift, block; forehand/backhand | Per-field prediction and review status |
| Error interpretation | Forced versus unforced; missed reasonable opportunity | Explicit academy rubric, context and coach adjudication |
| Tactical hypothesis | A short lift invited attack; partner rotation exposed a space | Link the sequence; do not claim causal proof from occurrence |

All records carry `proposed`, `accepted`, `rejected` or `unknown` at the relevant
field/event level, with model/configuration version, evidence references, review
history and reviewer type. Model confidence is not automatically a calibrated
probability. A disputed error label must not erase an otherwise confirmed score.

## 1. Match setup and identity

Capture match/session IDs, game boundaries, singles/doubles, side A/B membership,
player slots, optional roster IDs, initial known score, server/receiver when known,
court/camera configuration and scoring-profile version. Player identities remain
attached to people when they change ends; image left/right or near/far is not a
permanent team identity. In doubles, partners can rotate and exchange court areas.

Record partial recordings explicitly: unknown initial score, missing rallies,
camera cuts, replays, breaks and incomplete rally starts/ends. Never reconstruct
missing points by assuming the visible rallies are consecutive. A training drill
may use custom scoring or no scoring and must not be treated as a full match.

Use a configurable rules engine with effective/versioned profiles, rather than
hardcoding one points-to-win format. Verify the applicable competition rules
when implementing/releasing a profile. This plan checked the BWF-hosted 2025
laws for rally/fault/let concepts; that historical document alone does not establish
the appropriate rules for every recording or future event.

## 2. Shot-by-shot event ledger

Make an editable event ledger before building automatic shot statistics. A coach
can create the first complete analysis manually while automatic proposals improve.
This gives the product useful verified output without waiting for perfect models.

For each contact record:

- Session/game/rally ID, stroke order, native contact frame or uncertainty interval,
  source timestamp, clip bounds, camera scene and observation evidence IDs.
- Hitter player slot and side; unknown hitter allowed. Store server/receiver roles
  where known. A swing or attempted miss without shuttle contact is a separate
  `attempt` event, not a manufactured contact/stroke.
- Shot family, optional subtype, forehand/backhand or unknown, overhead/underarm/
  sidearm where supported, contact zone, intended/observed direction and destination.
- Execution state such as continued play, net failure, observed out, body contact
  or unknown. An unsuccessful contacted return is still a contact event even if
  it never crosses the net.
- Per-field evidence, confidence and review status. Editing a contact invalidates
  dependent stroke order, shot attribution, patterns and summaries until recomputed.

Start with this hierarchy; avoid an uncontrolled list of overlapping labels:

| Family | Initial optional refinements |
| --- | --- |
| Serve | Low/short, flick/high, drive; unknown subtype |
| Clear | Attacking/defensive if reliable; unknown subtype |
| Drop | Fast/slow; sliced/reverse-sliced only after dedicated validation |
| Smash | Full/half or stick only if distinguishable; unknown subtype |
| Drive | Flat attacking/defensive context as separate tags |
| Net shot | Straight/cross; tight/tumbling only when visible |
| Lift | Straight/cross; short/deep destination when endpoint supports it |
| Block | Defensive block; distinguish from net shot using incoming context |
| Push | Separate from drive or net shot when taxonomy agreement supports it |
| Kill | Net/forecourt finishing action; success is recorded separately |
| Other / unknown | Preserve uncertainty; do not force a class |

Shot name, direction, execution quality and point result are separate fields.
Do not call every overhead action a smash or every downward shot a winner.
Use a short temporal window around contact with player/racket or wrist motion,
incoming/outgoing shuttle observations and game context. Physical shuttle speed
is not a prerequisite; do not derive MPH from a floor-plane homography.

## 3. Rally boundaries and the observable finish

Identify service/start, contacts in order, end-of-play event, pauses, lets, replays
and incomplete/abandoned segments. Exclude broadcast replays from event counts.
Do not detect the end solely from a tracking disappearance or a player's reaction.

Record `terminal_observation` independently of who receives the point:

- Shuttle first contacts the floor: contact area and side, in/out/uncertain.
- Contacted shot fails to cross the net, if visible.
- Shuttle touches a body or another prohibited object, if reliably observed.
- Officially indicated fault, including service/receiver/player faults.
- Let/replay, stoppage, abandonment or unknown terminal event.
- Administrative/penalty score adjustments as separate non-stroke events.

Court boundaries depend on singles/doubles and service/rally phase. Lines belong
to their defined court area under the cited BWF laws. For any automated line
decision, require the actual first floor-contact evidence and a calibrated court,
with a near-line uncertainty band. An airborne shuttle's image point is not a
floor landing point. A feather/blur centroid must not be treated as precise
contact geometry. A single oblique low-resolution view will leave some cases
unresolved; this application is not an officiating line-call system.

The precise service-height, double-contact or net-touch judgment may be invisible
from this camera. Support official/manual annotation; do not invent those faults
from a generic pose or shuttle gap. An out-looking shot does not decide the point
if a prior fault had already ended play.

## 4. Award the point and keep the score

Use an evidence ledger for `point_awarded_to`: accepted terminal/rule evidence,
an official/scorekeeper entry, or a reconciled scoreboard change. Keep evidence
sources distinct and display conflicts. Player celebration, the next server,
body language and OCR alone are proposals or consistency checks, not proof.

The engine must handle:

- One awarded point per ordinary scored rally; zero for a let, replay or unresolved
  outcome. Unknown outcomes do not silently update the score.
- A known point winner with unknown cause, and a known score adjustment with no
  observed rally. Do not attach a penalty point to a fictional shot.
- Game/match boundaries, configured deuce/cap behavior, end changes, server and
  receiver sequence, doubles service positions and custom academy formats.
- Initial score unknown, scoreboard absent, OCR disagreement, missed footage and
  manual correction. Preserve score discontinuities instead of filling them in.

Store `score_before`/`score_after`, awarded side, score source, rules-profile version
and validation state. Corrections recompute subsequent known scores and recheck
service consistency. Every displayed score must reconcile with reviewed evidence;
unresolved differences stay visible and prevent a confirmed match total.

## 5. Direct winners and forced/unforced errors

These are analytical labels, not interchangeable with a physical terminal event.
The following is a proposed academy rubric, to be refined with coaches and versioned.
It is not a claim that BWF laws define a universal forced/unforced taxonomy.

| Finish attribution | Proposed operational meaning | Evidence/review needed |
| --- | --- | --- |
| Direct winner | A successful shot ends the point without an opponent's successful return; the rubric credits the shot rather than a routine missed opportunity | Terminal result, hitter and opponent opportunity reviewed |
| Forced error | A failed return/attempt occurs under material pressure supported by the preceding play | Incoming shot, reach/timing/body position and coach judgment |
| Unforced error | A failed return/attempt occurs from a reasonably playable opportunity for the agreed skill/context rubric | Opportunity, balance/time and coach judgment; not inferred merely from a miss |
| Other fault / administrative | Confirmed fault or score event not appropriately attributed as a shot execution error | Rule/official evidence; no invented hitter blame |
| Undetermined | Point winner may be known but attribution is ambiguous | Preserve unknown; request adjudication in review queue |
| Not applicable | Let, replay or incomplete/unscored practice segment | No winner/error count |

An untouched shuttle landing in can establish the rally winner when visible,
but does not automatically distinguish a clean winner from a routine airshot or
an unreachable forced miss. Coaches must agree how those cases are categorized.
Use one primary finish attribution per scored rally; optional pressure tags do
not create a second winner/error count.

Pressure evidence can include short available response time, displacement,
late/low contact, a stretch/dive, being off balance, a body-targeted shot, a
rapid change of direction, or an observed short defensive return. Record missing
evidence. No single indicator decides the label. In particular:

- Not every error after a smash is forced.
- An error after a drop or net shot can be forced.
- A smash can be an unforced error by the attacker.
- A late or awkward contact can reflect previous positioning, not solely the last incoming shot.
- Difficulty depends on level and context; freeze that rubric before evaluation.
- A clear shot class does not establish the opponent's intent or ability.

Keep `error_player` separate from losing side. In doubles, an ambiguous failure
between partners is `responsibility_unknown` or an explicitly reviewed coordination
issue. Do not blame the nearest person or both partners automatically.

Two coaches independently label the initial winner/forced/unforced set. Record
disagreement, adjudicated decisions and ambiguity. Introduce automatic forcedness
only after agreement is sufficient; otherwise provide a review suggestion and
evidence panel. Do not hide human disagreement behind a high model score.

## 6. Answer “which shot gained or lost the point?” correctly

Keep distinct links rather than one misleading `winning_shot` field:

1. `last_contact_event_id`: the last observed racket contact, possibly an error by the losing side.
2. `terminal_error_event_id`: a failed shot/attempt responsible for the finish, if accepted.
3. `direct_winner_event_id`: the credited winning shot for a direct-winner attribution, otherwise null.
4. `preceding_pressure_event_id`: the opponent shot associated with a forced error, if reviewed.
5. `setup_sequence_event_ids`: earlier shots/context preceding the finish; tactical interpretation remains separately reviewed.

Hypothetical example, not a label from the supplied video:

| Stroke | Side/player | Shot | Immediate result |
| --- | --- | --- | --- |
| 1 | A1 | Low serve | Rally continues |
| 2 | B1 | Net return | Rally continues |
| 3 | A2 | Net shot | Rally continues |
| 4 | B2 | Lift | Rally continues |
| 5 | A1 | Smash | B2 attempts a return |
| 6 | B2 | Block | Shuttle fails to cross the net |

If accepted, side A wins one point. B2's block is the final-contact/error shot.
A1's smash is the preceding pressure shot. Forced versus unforced remains a
reviewed judgment. A2's net shot may be tagged as part of the setup sequence,
but is not automatically awarded an additional point or proven causal credit.
If A1's smash had instead landed out, side B would win and the smash would be
the terminal error; its shot family alone does not imply attacking success.

## 7. Reports and tactical analysis

### Essential player/pair report

- Shot distribution and contact count, including serve and explicit unknowns.
- Confirmed direct winners, forced errors, unforced errors and unresolved finishes,
  split into committed and induced where justified by event links.
- Error execution categories: net, long, wide, failed contact, service fault,
  body/net/player fault, other or unknown; keep fault evidence separate.
- Serve/receive outcomes, return-of-serve patterns and early-rally performance.
- Shot direction and observed destination maps, forehand/backhand tendencies,
  and shot-to-return response matrices.
- Rally length/duration, pressure sequences, front/mid/rear-court involvement,
  and doubles rotation/coverage issues when player identity and geometry support them.
- Video playlists for each error, winner, class, player, pattern and unresolved case.

### Rates with clear denominators

- `terminal_error_rate(type)` = accepted terminal errors on that shot type /
  eligible classified attempts of that type. State how failed-contact attempts
  with unknown intended type are excluded and report their count separately.
- `direct_winner_rate(type)` = accepted direct winners of that type /
  eligible classified contacts of that type.
- `rally_win_rate_after(type)` = won rallies in which the side played that type /
  decided rallies in which it played that type, counting each rally once per side/type.
  This is an association, not the probability that a shot caused a point.
- `induced_forced_errors` requires an accepted preceding-pressure link. Report
  the number of judgments reviewed and the number still undetermined.
- Per-contact analyses and per-rally analyses are separate. A rally with five
  smashes must not count as five independent wins in a rally-level success rate.

Always show denominators, analyzed coverage, unknown/unreviewed counts and sample
sizes. Do not pool doubles teammates' shared point as two team points. Avoid
ranking players from a tiny or systematically missing sample. Match phase and
score-pressure comparisons require a known score and sufficient examples.

### Later coaching layer

Generate evidence-linked suggestions such as “review these accepted short-lift
errors” or “this serve/return pattern preceded these losses.” Separate factual
counts from tactical hypotheses and coach advice. No general quality score,
fatigue/psychology claim, injury assessment or causal benefit estimate follows
automatically from these labels. Physical speed, player distance and recovery
time retain their existing calibration/ground-contact gates.

## 8. Review experience

Add an editable rally timeline with frame stepping and a contact list. Clicking
a row plays context before and after that event. Give coaches controls to:

- Add/delete/move contacts; record a missed attempt without inventing contact.
- Change hitter, side, class, direction or visibility; split and merge rallies.
- Mark an incomplete rally, replay, let or non-rally score adjustment.
- Set point winner and score evidence, then separately label finish attribution.
- Select error player, preceding pressure shot and optional setup sequence.
- Compare evidence and model alternatives, leave comments, accept or leave unknown.

Show independent status badges for outcome, class and forcedness. Preserve edits,
reviewer type, rubric/model version and revisions. Provide undo/supersession and
recompute only affected summaries. A review queue prioritizes score conflicts,
ID switches, missing terminal evidence and disputed forcedness. Do not prefill
an authenticated human review from an automated run.

## 9. Data, models and validation

Use unedited academy videos with the complete court and rally start/end visible;
retain native frames/timestamps and camera setup. Add another view where practical
for terminal events and occlusion. Higher frame rate may help fast contacts, but
does not by itself solve identity, line calls or forcedness. Optional audio and
scoreboard data must have alignment and provenance checks.

Bootstrap the manual event ledger on the current clip as development work, then
collect multi-session, multi-player singles/doubles footage. Proposed first
annotation batch: 200 complete rallies across several sessions, deliberately
including overlaps, misses, failed returns, lets, faults and incomplete clips.
This is a starting workload, not a claim of statistical adequacy for all classes.
Stratify to obtain enough examples of each reported class/error; rare categories
may need further collection and should remain unavailable if evidence is sparse.

Freeze a taxonomy and annotation guide, contact/time tolerance, skill-level error
rubric, scoring profiles and session-level splits before evaluation. Keep players,
sessions and at least one camera/court out of tuning. Never split adjacent frames
from the same rally across train and test. Record coach agreement separately from
model agreement; adjudicate the initial outcome/error test set independently.

Public research can inform event schemas and model comparisons. ShuttleSet is
a human-annotated singles resource, so it does not establish accuracy for academy
doubles. Check dataset/model access, terms and transfer suitability before use;
no download or incorporation is authorized by this planning document alone.
Start with current observations and a bounded contact/shot baseline, then choose
whether adaptation/training is justified by measured failures. Do not rebuild
tracking merely to adopt a research architecture.

Proposed gates, fixed before the corresponding experiment:

| Capability | Measurement and initial gate |
| --- | --- |
| Contacts | One-to-one event matching; precision and recall >=90% within +/-2 native frames on declared footage FPS; also milliseconds and timing-error distribution |
| Hitter | >=95% correct player attribution among matched, adjudicated contacts; report unknown/missed contacts separately |
| Shot family | Macro F1 >=0.85 and >=85% precision for every class exposed as accepted automatic output; publish confusion matrix and abstention/coverage |
| Point winner | >=99% accuracy on complete adjudicated rallies, with coverage reported; every accepted session total must reconcile, unresolved rallies are not scored |
| Terminal cause | Per-category precision/recall and near-line abstention; unsupported service/net/body faults remain manual |
| Forced/unforced | Coach agreement first; publish Cohen's kappa and class agreement. Proposed kappa >=0.70 before using a shared rubric, then >=85% per-class precision for any automatic accepted label with useful, declared coverage |
| Event links | Correct references, team consistency, temporal order and no cross-rally/gap links; zero duplicate point awards in tests |
| End-to-end | Per-session score reconciliation, reviewed report reproduction, correction/revocation propagation, runtime and long-recording recovery |

These are engineering proposals, not achieved results or guarantees. Tighten or
revise them before seeing test results if coaches' use case demands it. Evaluate
singles/doubles, skill levels and cameras separately, report confidence intervals
using session/rally clusters, and retain difficult/unknown cases in coverage
reports. High selective accuracy with near-zero coverage is not useful automation.

## 10. Implementation sequence

| Phase | Concrete deliverable | Completion check |
| --- | --- | --- |
| A | Versioned match/rally/contact/attempt/outcome ledger, side membership, taxonomy and rules-profile schema; correction UI | A complete manually labeled rally exports consistently, with unknowns and undo/history |
| B | Contact and hitter proposals from bounded shuttle/pose windows; optional audio support | Contact timing/identity gates, gaps/cuts isolated, failures reviewable |
| C | Hierarchical shot family classifier and linked clip playlists | Per-class evaluation on frozen academy test sessions |
| D | Terminal-event evidence, manual/scoreboard reconciliation and score engine | Exactly one/zero point as appropriate; lets, missing footage, doubles and edits tested |
| E | Coach rubric, forcedness suggestions, preceding-shot/setup links | Independent coach agreement, adjudicated error evaluation, no double counting |
| F | Player/pair reports, pattern explorer and .NET event/report contract | Every reported count and clip traces to current accepted events and correct denominators |
| G | Multi-session pilot and targeted coaching summaries | Held-out session gates, actual browser/desktop review, coach sign-off for coaching use |

Build A first so coaches can produce useful complete analysis immediately; add
automatic proposals without changing the ledger's meaning. Reuse the existing
runner/cache and identity/review bindings. Keep render/report work separate from
model inference so corrections do not require rerunning detectors.

Add new `events`, `rallies`, `outcomes`, `score_events`, `analysis_rubric`,
`review_history` and `coverage` structures to a versioned consumer contract.
Do not silently reinterpret existing 1.0.0 unavailable metrics as accepted events;
update the Python exporter, .NET reader and fixtures together. Keep job completion,
point confirmation and coaching eligibility independent.

The critical path is A -> B -> C/D -> E -> F -> G. C and D share the event ledger
but need separate validation. Forced-error automation and final release dates
cannot be estimated responsibly until initial coach agreement, footage coverage
and class-specific failure rates are measured. Re-estimate after Phase A and the
first adjudicated annotation batch rather than promising full automatic analysis
from the current short demonstration.

## Sources consulted for this plan

- [BWF Laws of Badminton, version 5.0 (2), in force 26 April 2025](https://extranet.bwf.sport/docs/document-system/81/1466/1470/Section%204.1%20-%20Laws%20of%20Badminton%20-%2026%20April%202025%20V5.0%20(2)%20.pdf): rally, stroke, court-line, scoring, fault and let concepts. Historical version; select the applicable event profile before implementation.
- [ShuttleSet: A Human-Annotated Stroke-Level Singles Dataset for Badminton Tactical Analysis](https://arxiv.org/abs/2306.04948): research precedent for expert-annotated shot-level records and tactical analysis; singles scope, not evidence of our doubles performance.
- [Automated Hit-frame Detection for Badminton Match Analysis](https://arxiv.org/abs/2307.16000): research context for separating contact detection from downstream shot/movement analysis; no performance transfer claim or code adoption.

Sources accessed 2 November 2026. The forcedness rubric, proposed acceptance
targets, product design and implementation sequence above are BadmintonAI design
proposals, not official laws or conclusions asserted by those papers.
