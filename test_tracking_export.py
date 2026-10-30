import json
import unittest

import test_tracking_review as review_tests
from tracking_export import EXPORT_FILE, build_export, export, verify_export
from tracking_review import REVIEW_FILE, decide, prepare


class TrackingExportTests(unittest.TestCase):
    def setUp(self):
        # Reuse the synthetic two-scene fixture, never a real review decision.
        self.fixture = review_tests.TrackingReviewTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.run = self.fixture.run
        self.fixture.summary["video"] = {"width_px": 100, "height_px": 100}
        self.fixture.write_source()
        prepare(self.run, 2)

    def test_pending_and_rejected_scenes_export_no_positions(self):
        export(self.run)
        result = build_export(self.run)
        self.assertEqual(result["status"], "no_approved_intervals")
        self.assertEqual(result["segments"], [])
        self.assertEqual(verify_export(self.run)["status"], "valid")
        decide(self.run, 0, "rejected", "Test reviewer", "Synthetic", {})
        self.assertEqual(build_export(self.run)["segments"], [])

    def test_accepted_tracks_preserve_pixels_times_and_scene_scope(self):
        self.fixture.accept(0)
        self.fixture.accept(1)
        result = build_export(self.run)
        self.assertEqual([(s["interval_index"], s["scene_id"], s["track_id"])
                          for s in result["segments"]], [(0, 0, 0), (0, 0, 1), (1, 1, 2), (1, 1, 3)])
        first = result["segments"][0]["samples"]
        self.assertEqual([s["frame_index"] for s in first], [1, 2])
        self.assertEqual(first[0]["source_seconds"], 480.067)
        self.assertEqual(first[0]["box_bottom_px"], [20, 50])
        self.assertEqual(first[0]["bbox_xyxy_px"], [10, 10, 30, 50])
        self.assertEqual(result["movement_metrics"]["status"], "not_available")
        self.assertEqual(result["coaching_metrics"]["status"], "not_available")

    def test_unconfirmed_gaps_split_segments_without_interpolation(self):
        (self.run / REVIEW_FILE).unlink()
        for i, row in enumerate(self.fixture.rows):
            row["scene_id"], row["scene_cut"] = 0, False
            for n, person in enumerate(row["persons"]):
                person["track_id"] = None if i in (0, 3) else n
        self.fixture.summary["scene_review"]["scenes_observed"] = 1
        self.fixture.write_source()
        prepare(self.run, 2)
        self.fixture.accept()
        result = build_export(self.run)
        self.assertEqual([[s["frame_index"] for s in segment["samples"]]
                          for segment in result["segments"]], [[1, 2], [1, 2], [4, 5], [4, 5]])

    def test_revocation_and_changed_source_invalidate_saved_exports(self):
        self.fixture.accept()
        path = export(self.run)
        saved = path.read_bytes()
        decide(self.run, 0, "pending", "", "Needs review", {})
        with self.assertRaisesRegex(ValueError, "stale"):
            verify_export(self.run)
        self.assertEqual(path.read_bytes(), saved)
        export(self.run)
        self.assertEqual(verify_export(self.run)["segments"], 0)
        with (self.run / "player_tracks.jsonl").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "stale"):
            export(self.run)

    def test_modified_export_and_forged_stored_intervals_are_rejected(self):
        path = self.run / REVIEW_FILE
        review = json.loads(path.read_text())
        review["approved_intervals"] = [{"start_frame": 0, "end_frame": 1000}]
        path.write_text(json.dumps(review))
        self.assertEqual(build_export(self.run)["segments"], [])
        output = export(self.run)
        data = json.loads(output.read_text())
        data["segments"] = [{"track_id": 999}]
        output.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "modified"):
            verify_export(self.run)

    def test_invalid_accepted_sample_cannot_replace_existing_export(self):
        self.fixture.accept()
        path = export(self.run)
        saved = path.read_bytes()
        (self.run / REVIEW_FILE).unlink()
        self.fixture.rows[1]["persons"][0]["confidence"] = float("nan")
        self.fixture.write_source()
        prepare(self.run, 2)
        self.fixture.accept()
        with self.assertRaisesRegex(ValueError, "outside"):
            export(self.run)
        self.assertEqual(path.read_bytes(), saved)
        self.fixture.rows[1]["persons"][0]["confidence"] = .9
        self.fixture.rows[1]["persons"][0]["bbox_xyxy_px"][2] = 200
        self.fixture.write_source()
        (self.run / REVIEW_FILE).unlink()
        prepare(self.run, 2)
        self.fixture.accept()
        with self.assertRaisesRegex(ValueError, "outside"):
            export(self.run)

    def test_export_retains_automated_provenance_and_detects_changed_provenance(self):
        from tracking_review import CHECKS
        decide(self.run, 0, "accepted", "Codex", "Synthetic", dict.fromkeys(CHECKS, True), "automated")
        export(self.run)
        result = build_export(self.run)
        self.assertEqual(result["review_provenance"], {"accepted_by_reviewer_kind": {"automated": 1}})
        self.assertEqual(result["coaching_metrics"]["status"], "not_available")
        decide(self.run, 0, "accepted", "Human test reviewer", "Synthetic", dict.fromkeys(CHECKS, True), "human")
        with self.assertRaisesRegex(ValueError, "stale"):
            verify_export(self.run)


if __name__ == "__main__":
    unittest.main()
