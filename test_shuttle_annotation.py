from copy import deepcopy
import json
from pathlib import Path
import unittest

from badminton_proto import sha256_file
from shuttle_annotation import apply_corrections, prepare_editor
from shuttle_benchmark import evaluate
import test_shuttle_benchmark as benchmark_tests


class ShuttleAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = benchmark_tests.ShuttleBenchmarkTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.patch = {"schema_version":"0.1.0", "benchmark_sha256":sha256_file(self.fixture.benchmark),
                      "reviewer":"Synthetic automated reviewer", "reviewer_kind":"automated",
                      "changes":[{"case_id":"case-1", "visibility":"visible", "point_px":[1,1],
                                  "view":"wide_court", "notes":"Synthetic correction only"}]}
        self.correction = self.root/"correction.json"

    def apply(self, output=None):
        self.correction.write_text(json.dumps(self.patch), encoding="utf-8")
        return apply_corrections(self.fixture.benchmark, self.correction, output or self.root/"corrected")

    def test_editor_is_offline_source_only_and_escapes_embedded_data(self):
        self.fixture.data["cases"][0]["id"] = '</script><img src=x onerror="bad()">'
        self.fixture.data["cases"][0]["notes"] = '<script>bad()</script>&'
        self.fixture.save()
        target = prepare_editor(self.fixture.benchmark, self.root/"editor")
        html = target.read_text(encoding="utf-8")
        self.assertNotIn('<script>bad()', html)
        self.assertNotIn('<img src=x', html)
        self.assertIn('\\u003cscript\\u003e', html)
        self.assertNotIn('https://', html)
        self.assertTrue((target.parent/"editor.js").is_file())
        meta = json.loads((target.parent/"editor_source.json").read_text())
        self.assertEqual(meta["benchmark_sha256"], sha256_file(self.fixture.benchmark))
        for item in meta["images"]:
            self.assertEqual(item["sha256"], sha256_file(target.parent/item["image"]))
        with self.assertRaisesRegex(ValueError, "new editor"):
            prepare_editor(self.fixture.benchmark, target.parent)

    def test_correction_preserves_sources_unchanged_labels_and_review_history(self):
        original = self.fixture.benchmark.read_bytes()
        target = self.apply()
        data = json.loads(target.read_text())
        self.assertEqual(self.fixture.benchmark.read_bytes(), original)
        self.assertEqual(data["source_video_sha256"], self.fixture.data["source_video_sha256"])
        self.assertEqual(data["sources"], self.fixture.data["sources"])
        before, after = self.fixture.data["cases"][1], data["cases"][1]
        self.assertEqual(after["point_px"], [1,1])
        self.assertEqual(after["reviewer_kind"], "automated")
        self.assertEqual(after["label_history"][0]["previous_label"]["visibility"], before["visibility"])
        self.assertEqual(after["label_history"][0]["previous_label"]["reviewer"], before["reviewer"])
        self.assertEqual(data["cases"][0]["reviewer"], self.fixture.data["cases"][0]["reviewer"])
        self.assertEqual(data["annotation_revisions"][0]["correction_sha256"], sha256_file(self.correction))
        self.assertEqual(data["annotation_revisions"][0]["case_ids"], ["case-1"])
        self.assertTrue((target.parent/"editor/index.html").exists())
        result = evaluate(target, {"source":self.fixture.predictions})
        self.assertEqual(result["visible_points"], 2)
        self.assertEqual(result["verified_shuttle_positions"], 0)
        self.patch["benchmark_sha256"] = sha256_file(target)
        self.patch["changes"][0].update(visibility="uncertain", point_px=None)
        self.correction.write_text(json.dumps(self.patch))
        next_target = apply_corrections(target, self.correction, self.root/"revision-two")
        next_data = json.loads(next_target.read_text())
        self.assertEqual(len(next_data["annotation_revisions"]), 2)
        self.assertEqual(len(next_data["cases"][1]["label_history"]), 2)

    def test_stale_or_modified_source_packets_and_existing_outputs_are_rejected(self):
        target = self.apply()
        saved = target.read_bytes()
        with self.assertRaisesRegex(ValueError, "new corrected"):
            self.apply(target.parent)
        self.assertEqual(target.read_bytes(), saved)
        self.fixture.data["cases"][0]["notes"] = "changed after editor creation"
        self.fixture.save()
        with self.assertRaisesRegex(ValueError, "exact benchmark"):
            self.apply(self.root/"stale")
        self.assertFalse((self.root/"stale").exists())
        self.patch["benchmark_sha256"] = sha256_file(self.fixture.benchmark)
        (self.fixture.packet/"frame.png").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "image has changed"):
            self.apply(self.root/"changed-image")

    def test_patch_cannot_change_source_or_claim_location_for_unknown_frame(self):
        original = deepcopy(self.patch)
        invalid = []
        invalid.append({**original, "source_video_sha256":"b"*64})
        for change in ({"point_px":[100,1]}, {"visibility":"uncertain"}, {"case_id":"missing"},
                       {"frame_index":100}, {"view":"bad"}, {"notes":None}):
            item=deepcopy(original); item["changes"][0].update(change); invalid.append(item)
        duplicate=deepcopy(original); duplicate["changes"]*=2; invalid.append(duplicate)
        for number,item in enumerate(invalid):
            self.patch=item
            with self.assertRaises(ValueError): self.apply(self.root/f"invalid-{number}")
            self.assertFalse((self.root/f"invalid-{number}").exists())
        for reviewer,kind in (("", "automated"), ("Someone", ""), ("Someone", "unspecified")):
            self.patch={**original,"reviewer":reviewer,"reviewer_kind":kind}
            with self.assertRaisesRegex(ValueError, "explicit reviewer"): self.apply(self.root/"invalid-provenance")

    def test_escaping_images_and_contradictory_negative_labels_are_rejected(self):
        raw=(self.fixture.packet/"frame.png").read_bytes()
        (self.root/"outside.png").write_bytes(raw)
        self.fixture.data["cases"][0]["image"]="../outside.png"
        self.fixture.save()
        with self.assertRaisesRegex(ValueError, "outside"): prepare_editor(self.fixture.benchmark,self.root/"escape")
        self.fixture.data["cases"][0]["image"]="frame.png";self.fixture.save()
        self.patch["benchmark_sha256"]=sha256_file(self.fixture.benchmark)
        self.patch["changes"][0].update(case_id="case-2",point_px=[8,8])
        with self.assertRaisesRegex(ValueError,"contradicts"): self.apply()


if __name__ == "__main__": unittest.main()
