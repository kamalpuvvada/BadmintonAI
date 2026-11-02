# Offline worker contract, version 1.0.0

The worker's job folder is the unit of transfer. Copy the complete folder after
publication; do not transfer a partially written video while processing runs.
The original recording remains at its source path and is not copied by the job.

`job.json` is replaced atomically after state changes. It contains the session ID,
job status, active stage, UTC timestamps, source/model/configuration/code/package
bindings, and stage attempts. Each finished stage binds its artifacts by SHA-256.
Previous attempt folders remain available for diagnosis. Expected player count
is a review expectation, not a detector output constraint.

Job states:

| State | Meaning for a consumer |
| --- | --- |
| running | Analysis or refresh is underway; do not import the public snapshot |
| needs_review | Preview/results are available, but player scene review is pending |
| completed | Processing finished under current decisions; exclusions and unavailable capabilities remain possible |
| failed | A stage failed; inspect error and stage log |
| cancelled | Cancellation acknowledged; resume may reuse completed stages |

Stages are inspect, players, review, shuttle, render and export. Stage state can
also be pending or not_available. `review=completed` means every scene has a
decision, including rejection. It does not mean every detection is correct.
`export=completed` means the public report has been written and hashed. Snapshots
with a different job/result state or without this export are not importable.

`result.json` is the current consumer snapshot. Its status may additionally be
`stale` after refresh begins or fails. It includes:

- `schema_version`, `session_id`, `status`, source hash and video metadata.
- `players`: diagnostic counts, experimental status, and exact review verification
  snapshot. Temporary tracker IDs are not athlete identities.
- `shuttle`: experimental availability and zero verified positions until a separate
  shuttle validation workflow exists. Pending player review grants no eligibility.
- `metrics`: status, nullable value and reason for every unavailable measurement.
- `preview`: source/preview geometry, FPS, decoded frame count, sample display
  semantics, timestamp assumptions and sample images.
- `artifacts`: paths relative to the job folder, with null for absent artifacts.

Public JSON is written without NaN/Infinity. The current renderer assumes a
constant frame rate, preserves the native frame sequence and adds a separate
64-pixel header (plus an even-dimension pad when required by the encoder). Boxes
are sampled and held for less than one sample interval, never interpolated.
Preview audio is omitted and the source remains unchanged.

Existing tracking review is authoritative for player eligibility. Change a
decision through that workflow, then resume the same job to refresh downstream
results. Snapshot files do not automatically monitor future revocation. The
runner rechecks review during rendering/publication and never upgrades
automated review to human approval.

The .NET reader verifies schema, current manifest/result status, exact result
hash, source/session binding, preview decode counts and local artifact existence.
It rejects path traversal and linked artifact paths. The reader is intentionally
read-only. A production RACKETEK host must add its own academy/session access
rules and worker lifecycle handling; no backend database has been changed here.
