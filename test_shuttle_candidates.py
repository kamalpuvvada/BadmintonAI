import json
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from badminton_proto import sha256_file
from player_tracking import _encode_preview
from shuttle_candidates import ProposalLinker, frame_context, moving_candidates, run
import test_tracking_review as review_tests
from tracking_review import prepare


def candidate(x, y=40):
    return {"center_px": [x, y]}


class ShuttleCandidateTests(unittest.TestCase):
    def test_motion_rejects_static_lines_and_ghosts_but_keeps_current_bright_dot(self):
        previous = np.zeros((100, 100, 3), dtype=np.uint8)
        previous[20:22, 10:90] = 255
        current, following = previous.copy(), previous.copy()
        previous[40:43, 30:33] = 255  # Previous-frame ghost is not a current bright object.
        current[60:63, 50:53] = 255
        following[65:68, 55:58] = 255
        found = moving_candidates(previous, current, following, (0, 0, 1, 1), [])
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["center_px"], [51, 61])

    def test_roi_and_player_masks_exclude_bright_body_fragments(self):
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        current = blank.copy()
        current[40:43, 40:43] = 255
        self.assertFalse(moving_candidates(blank, current, blank, (.6, 0, 1, 1), []))
        self.assertFalse(moving_candidates(blank, current, blank, (0, 0, 1, 1), [[38, 38, 45, 45]]))
        current[50:80, 50:80] = 255
        found = moving_candidates(blank, current, blank, (0, 0, 1, 1), [])
        self.assertEqual(len(found), 1)  # Large bright regions are not tiny proposals.

    def test_scene_background_suppresses_flickering_white_graphics(self):
        background = np.zeros((100, 100, 3), dtype=np.uint8)
        background[20:23, 20:23] = 240
        previous, following = background.copy(), background.copy()
        previous[20:23, 20:23] = 220
        following[20:23, 20:23] = 220
        current = background.copy()
        current[20:23, 20:23] = 255
        current[50:53, 50:53] = 255
        raw = moving_candidates(previous, current, following, (0, 0, 1, 1), [])
        filtered = moving_candidates(previous, current, following, (0, 0, 1, 1), [], background)
        self.assertEqual(len(raw), 2)
        self.assertEqual([c["center_px"] for c in filtered], [[51, 51]])

    def test_tinted_streak_mode_is_opt_in_requires_background_and_respects_masks(self):
        background = np.full((100, 100, 3), [200, 30, 40], dtype=np.uint8)
        current = background.copy()
        current[40:46, 50:53] = [235, 120, 130]
        self.assertFalse(moving_candidates(background, current, background, (0, 0, 1, 1), [], background))
        found = moving_candidates(background, current, background, (0, 0, 1, 1), [], background, True)
        self.assertEqual(len(found), 1)
        self.assertFalse(moving_candidates(background, current, background, (0, 0, 1, 1), [[48, 38, 55, 48]], background, True))
        with self.assertRaisesRegex(ValueError, "require a scene background"):
            moving_candidates(background, current, background, (0, 0, 1, 1), [], allow_tinted_streaks=True)

    def test_linker_never_bridges_gaps_or_scene_boundaries(self):
        linker = ProposalLinker()
        first = linker.update(0, 0, [candidate(10)])
        second = linker.update(1, 0, [candidate(15)])
        self.assertEqual(first["proposal_segment_id"], second["proposal_segment_id"])
        self.assertIsNone(linker.update(2, 0, []))
        after_gap = linker.update(3, 0, [candidate(20)])
        after_cut = linker.update(4, 1, [candidate(25)])
        skipped_frame = linker.update(6, 1, [candidate(30)])
        self.assertEqual([p["proposal_segment_id"] for p in (first, after_gap, after_cut, skipped_frame)], [0, 1, 2, 3])

    def test_ambiguity_and_large_motion_end_a_proposal(self):
        linker = ProposalLinker(max_step_px=20, ambiguity_margin_px=8)
        self.assertIsNone(linker.update(0, 0, [candidate(10), candidate(30)]))
        linker.update(1, 0, [candidate(10)])
        self.assertIsNone(linker.update(2, 0, [candidate(9), candidate(11)]))
        linker.update(3, 0, [candidate(10)])
        self.assertIsNone(linker.update(4, 0, [candidate(90)]))
        self.assertEqual(linker.update(5, 0, [candidate(90)])["verification_status"], "unverified")
        with self.assertRaises(ValueError):
            ProposalLinker(float("nan"))

    def test_context_does_not_extrapolate_approval_or_mask_across_cut(self):
        summary = {"sampling": {"fps": 15, "start_seconds": 480}}
        rows = [{"scene_id": i // 3, "persons": [{"bbox_xyxy_px": [i, 1, i + 1, 2]}]} for i in range(6)]
        intervals = [{"scene_id": 0, "start_frame": 1, "end_frame": 2}]
        self.assertFalse(frame_context(480, summary, rows, intervals)["eligible"])
        self.assertTrue(frame_context(480 + 2 / 15, summary, rows, intervals)["eligible"])
        self.assertFalse(frame_context(480 + 2.5 / 15, summary, rows, intervals)["eligible"])
        context = frame_context(480 + 3 / 15, summary, rows, intervals)
        self.assertFalse(context["eligible"])
        self.assertEqual(context["player_boxes"], [[3, 1, 4, 2], [4, 1, 5, 2]])

    def test_native_video_pipeline_preserves_gates_boundaries_and_unverified_status(self):
        fixture = review_tests.TrackingReviewTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.summary["sampling"]["start_seconds"] = 0
        fixture.summary["video"] = {"width_px": 100, "height_px": 100}
        for i, row in enumerate(fixture.rows):
            row["timestamp_seconds"] = round(i / 15, 3)
        frames = fixture.run / "native"
        frames.mkdir()
        for i in range(12):
            im = np.zeros((100, 100, 3), dtype=np.uint8)
            im[70:73, 35 + i * 3:38 + i * 3] = 255
            Image.fromarray(im).save(frames / f"frame_{i + 1:06d}.jpg")
        video = fixture.run / "source.mp4"
        _encode_preview(frames, video, 30)
        fixture.summary["session_id"] = "test-" + sha256_file(video)[:12]
        fixture.write_source()
        prepare(fixture.run, 2)
        fixture.accept(0)
        fixture.accept(1)
        output = fixture.run / "candidates"
        summary_path = run(video, fixture.run, output)
        summary = json.loads(summary_path.read_text())
        rows = [json.loads(line) for line in (output / "shuttle_candidates.jsonl").read_text().splitlines()]
        self.assertEqual(summary["sampling"], {"fps": 30, "start_seconds": 0, "frames": 12})
        self.assertTrue(all(row["verified_shuttle_position_px"] is None for row in rows))
        self.assertEqual(summary["coaching_metrics"]["status"], "not_available")
        self.assertEqual(summary["verified_shuttle_positions"], 0)
        self.assertEqual(sum(summary["frame_status_counts"].values()), 12)
        self.assertEqual([row["frame_index"] for row in rows if row["status"] == "excluded_player_review"], [0, 1, 5, 6, 7, 11])
        self.assertTrue((output / "candidate_preview.mp4").is_file())
        saved = summary_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "new output"):
            run(video, fixture.run, output)
        self.assertEqual(summary_path.read_bytes(), saved)
        fixture.summary["session_id"] = "wrong-source"
        fixture.write_source()
        # Re-prepare in a new packet rather than weakening stale-review checks.
        (fixture.run / "tracking_review.json").unlink()
        prepare(fixture.run, 2)
        with self.assertRaisesRegex(ValueError, "does not match"):
            run(video, fixture.run, fixture.run / "wrong")


if __name__ == "__main__":
    unittest.main()
