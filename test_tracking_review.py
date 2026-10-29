import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from tracking_review import CHECKS, REVIEW_FILE, decide, prepare, verify


class TrackingReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.rows = []
        for i in range(6):
            scene = i // 3
            self.rows.append({"frame_index": i, "timestamp_seconds": round(480 + i / 15, 3),
                "scene_id": scene, "scene_cut": i == 3, "court_view_status": "reference_match",
                "persons": [{"track_id": None if i in (0, 3) else scene * 2 + n,
                    "bbox_xyxy_px": [10 + 50 * n, 10, 30 + 50 * n, 50],
                    "foot_px": [20 + 50 * n, 50], "confidence": .9} for n in range(2)]})
        self.summary = {"session_id": "court-01-example", "sampling": {
            "fps": 15, "start_seconds": 480, "frames_processed": 6},
            "scene_review": {"scenes_observed": 2}, "artifacts": {
                "annotated_video": "preview.mp4", "review_frames": [
                    {"timestamp_seconds": 480, "image": "frame.jpg"},
                    {"timestamp_seconds": 480.2, "image": "frame.jpg"}]}}
        Image.new("RGB", (100, 100), "green").save(self.run / "frame.jpg")
        (self.run / "preview.mp4").write_bytes(b"synthetic preview placeholder")
        self.write_source()

    def write_source(self):
        (self.run / "tracking_summary.json").write_text(json.dumps(self.summary), encoding="utf-8")
        (self.run / "player_tracks.jsonl").write_text("".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8")

    def accept(self, scene=0):
        return decide(self.run, scene, "accepted", "Test reviewer", "Synthetic test only",
                      {key: True for key in CHECKS})

    def test_prepare_is_pending_and_never_enables_coaching(self):
        path = prepare(self.run, 2)
        review = json.loads(path.read_text())
        self.assertTrue(all(scene["review"]["decision"] == "pending" for scene in review["scenes"]))
        self.assertFalse(review["approved_intervals"])
        self.assertFalse(verify(self.run)["all_scenes_decided"])
        self.assertEqual(verify(self.run)["coaching_metrics"]["status"], "not_available")
        self.assertTrue((self.run / "tracking_review.html").exists())

    def test_acceptance_requires_explicit_checks_and_reviewer(self):
        prepare(self.run, 2)
        with self.assertRaisesRegex(ValueError, "reviewer"):
            decide(self.run, 0, "accepted", "", "", dict.fromkeys(CHECKS, True))
        with self.assertRaisesRegex(ValueError, "four verification"):
            decide(self.run, 0, "accepted", "Test reviewer", "", {"boxes": True})
        self.accept()
        result = verify(self.run)
        self.assertEqual(result["approval_scope"], "tracking_in_image_pixels")
        self.assertEqual(result["decisions"], {"accepted": 1, "pending": 1})
        self.assertEqual(result["approved_intervals"], [{"scene_id": 0, "start_frame": 1,
            "end_frame": 2, "start_seconds": 480.067, "end_seconds": 480.133, "track_ids": [0, 1]}])
        self.assertEqual(result["coaching_metrics"]["status"], "not_available")

    def test_intervals_never_join_scenes_or_unconfirmed_gaps(self):
        for i, row in enumerate(self.rows):
            row["scene_id"] = 0
            row["scene_cut"] = False
            for n, person in enumerate(row["persons"]):
                person["track_id"] = None if i in (0, 3) else n
        self.summary["scene_review"]["scenes_observed"] = 1
        self.write_source()
        prepare(self.run, 2)
        self.accept()
        self.assertEqual([(i["start_frame"], i["end_frame"]) for i in verify(self.run)["approved_intervals"]], [(1, 2), (4, 5)])

    def test_scene_boundaries_keep_approved_intervals_separate(self):
        prepare(self.run, 2)
        self.accept(0)
        self.accept(1)
        intervals = verify(self.run)["approved_intervals"]
        self.assertEqual([(i["scene_id"], i["start_frame"], i["end_frame"]) for i in intervals], [(0, 1, 2), (1, 4, 5)])
        self.assertTrue(verify(self.run)["all_scenes_decided"])

    def test_changed_source_invalidates_review_without_overwriting_it(self):
        path = prepare(self.run, 2)
        self.accept()
        saved = path.read_bytes()
        with (self.run / "player_tracks.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        for action in (lambda: verify(self.run), lambda: prepare(self.run), lambda: self.accept()):
            with self.assertRaisesRegex(ValueError, "stale"):
                action()
        self.assertEqual(path.read_bytes(), saved)

    def test_excluded_views_and_fragmented_ids_cannot_be_accepted(self):
        for row in self.rows[3:]:
            row["court_view_status"] = "reference_mismatch"
            row["persons"] = []
        self.rows[2]["persons"][0]["track_id"] = 99
        self.write_source()
        path = prepare(self.run, 2)
        review = json.loads(path.read_text())
        self.assertIn("Excluded camera view", review["scenes"][1]["blockers"])
        for scene in (0, 1):
            with self.assertRaisesRegex(ValueError, "no blockers"):
                self.accept(scene)
        decide(self.run, 1, "rejected", "Test reviewer", "Close-up", {})
        self.assertFalse(verify(self.run)["approved_intervals"])

    def test_reprepare_preserves_decisions_and_revocation_removes_intervals(self):
        prepare(self.run, 2)
        self.accept()
        path = prepare(self.run)
        self.assertEqual(json.loads(path.read_text())["scenes"][0]["review"]["decision"], "accepted")
        decide(self.run, 0, "pending", "", "Needs another look", {})
        self.assertFalse(verify(self.run)["approved_intervals"])
        history = json.loads(path.read_text())["scenes"][0]["review_history"]
        self.assertEqual([entry["decision"] for entry in history], ["accepted", "pending"])

    def test_forged_derived_intervals_and_flags_are_not_trusted(self):
        path = prepare(self.run, 2)
        review = json.loads(path.read_text())
        review["approved_intervals"] = [{"start_frame": 0, "end_frame": 100000}]
        path.write_text(json.dumps(review))
        self.assertFalse(verify(self.run)["approved_intervals"])
        review["scenes"][0]["review"] = {"decision": "accepted", "reviewer": "Test reviewer",
            "notes": "", "checks": dict.fromkeys(CHECKS, 1)}
        path.write_text(json.dumps(review))
        with self.assertRaisesRegex(ValueError, "cannot be accepted"):
            verify(self.run)

    def test_malformed_frame_or_boundary_data_is_rejected(self):
        self.rows[3]["scene_cut"] = False
        self.write_source()
        with self.assertRaisesRegex(ValueError, "boundaries"):
            prepare(self.run, 2)

    def test_html_escapes_notes_and_rejects_artifact_path_escape(self):
        prepare(self.run, 2)
        decide(self.run, 0, "rejected", "<reviewer>", "<script>alert(1)</script>", {})
        html = (self.run / "tracking_review.html").read_text()
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("&lt;reviewer&gt;", html)
        # A new source run with an escaping artifact cannot produce a packet.
        (self.run / REVIEW_FILE).unlink()
        self.summary["artifacts"]["annotated_video"] = "../outside.mp4"
        self.write_source()
        with self.assertRaisesRegex(ValueError, "stay inside"):
            prepare(self.run, 2)
        self.assertFalse((self.run / REVIEW_FILE).exists())


if __name__ == "__main__":
    unittest.main()
