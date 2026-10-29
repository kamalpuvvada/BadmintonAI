"""Run with: python -m unittest discover -v"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from badminton_proto import (
    capture_warnings, project_points, solve_homography, calibrate,
)


class PrototypeTests(unittest.TestCase):
    def test_homography_maps_court_corners(self):
        pixels = np.asarray([[125, 90], [505, 105], [610, 910], [60, 880]], dtype=float)
        court = np.asarray([[0, 0], [6.1, 0], [6.1, 13.4], [0, 13.4]], dtype=float)
        transform = solve_homography(pixels, court)
        np.testing.assert_allclose(project_points(pixels, transform), court, atol=1e-8)
        self.assertEqual(project_points(np.asarray([[335, 500]]), transform).shape, (1, 2))

    def test_invalid_homography(self):
        bad = np.asarray([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=float)
        target = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        with self.assertRaises(ValueError):
            solve_homography(bad, target)

    def test_capture_warnings_are_explicit(self):
        metadata = {"width_px": 720, "height_px": 1280, "frame_rate_fps": 30}
        warnings = capture_warnings(metadata, True)
        self.assertEqual(len(warnings), 4)
        self.assertTrue(any("graphics" in item for item in warnings))

    def test_court_calibration_rejects_crossed_points(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "frame.jpg"
            corners = Path(directory) / "corners.json"
            Image.new("RGB", (600, 1000)).save(image)
            corners.write_text('{"far_left":[100,100],"far_right":[500,800],"near_right":[500,100],"near_left":[100,800]}')
            with self.assertRaises(ValueError):
                calibrate(image, corners, Path(directory) / "out", "court-01")

    def test_court_calibration_exports_preview_and_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "frame.jpg"
            corners = Path(directory) / "corners.json"
            Image.new("RGB", (600, 1000), "#287cb1").save(frame)
            corners.write_text(json.dumps({
                "far_left": [150, 200], "far_right": [450, 200],
                "near_right": [550, 850], "near_left": [50, 850],
            }))
            output = Path(directory) / "out"
            result = calibrate(frame, corners, output, "court-01")
            data = json.loads(result.read_text())
            self.assertTrue((output / "court_calibration_preview.jpg").is_file())
            self.assertLess(data["corner_fit_max_error_m"], 1e-8)
            self.assertEqual(data["court"]["length_m"], 13.4)


if __name__ == "__main__":
    unittest.main()
