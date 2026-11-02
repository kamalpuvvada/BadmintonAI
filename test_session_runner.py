import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from session_runner import (Cancelled, artifact_binding, cached_stage, execute, invalidate_report,
                            job_lock, run_job, validate_configuration, code_binding)
from session_results import observation_at, render_preview
from tracking_review import CHECKS, decide
from badminton_proto import sha256_file


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root / 'source.mp4'
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*'mp4v'), 10, (100, 80))
        for i in range(10):
            writer.write(np.full((80, 100, 3), (20*i, 80, 50), dtype=np.uint8))
        writer.release()
        self.weights = self.root / 'rf-detr-nano.pth'
        self.weights.write_bytes(b'synthetic model; never loaded')
        self.shuttle_weights = self.root / 'shuttle.pt'
        self.shuttle_weights.write_bytes(b'synthetic shuttle checkpoint; never loaded')
        self.output = self.root / 'job'
        self.config = dict(camera='test', sample_fps=5, expected_players=2, threads=1,
                           max_seconds=None, roi=None, detect_roi=None, processed_overlay=True)
        self.calls = []

    def fake_execute(self, command, directory, output, environment):
        script = Path(command[1]).name
        self.calls.append(script)
        if script == 'badminton_proto.py':
            (directory / 'session.json').write_text('{}')
            return
        if script == 'shuttle_tracknet.py':
            data = directory / 'data'
            data.mkdir()
            (data / 'shuttle_summary.json').write_text('{}')
            (data / 'shuttle_candidates.jsonl').write_text('')
            return
        summary = {'session_id': f'test-{sha256_file(self.video)[:12]}', 'sampling': {'fps': 5, 'start_seconds': 0, 'frames_processed': 5,
                   'max_seconds': None}, 'scene_review': {'scenes_observed': 1},
                   'video': {'width_px': 100, 'height_px': 80}, 'artifacts': {
                       'annotated_video': 'tracked_preview.mp4', 'review_frames': []},
                   'diagnostics': {'unique_track_ids_observed': 2}}
        rows = [{'frame_index': i, 'timestamp_seconds': i/5, 'scene_id': 0, 'scene_cut': False,
                 'persons': [{'track_id': n, 'bbox_xyxy_px': [5+40*n, 5, 25+40*n, 70],
                             'foot_px': [15+40*n, 70], 'confidence': .9} for n in range(2)]} for i in range(5)]
        (directory / 'tracking_summary.json').write_text(json.dumps(summary))
        (directory / 'player_tracks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (directory / 'tracked_preview.mp4').write_bytes(b'fixture preview')
        (directory / 'detector_review').mkdir()
        cv2.imwrite(str(directory / 'detector_review/review_contact_sheet.jpg'), np.zeros((80, 100, 3), np.uint8))

    def run_fake(self, resume=False):
        with patch('session_runner.execute', side_effect=self.fake_execute):
            return run_job(self.video, self.output, self.config, self.weights, self.shuttle_weights, resume)

    def test_end_to_end_pending_review_and_exact_native_preview(self):
        job = self.run_fake()
        self.assertEqual(job['status'], 'needs_review')
        self.assertEqual(job['stages']['shuttle']['status'], 'needs_review')
        result = json.loads((self.output / 'result.json').read_text())
        self.assertEqual(result['preview']['frames'], 10)
        self.assertEqual(result['preview']['decoded_preview_frames'], 10)
        self.assertEqual(result['preview']['source_rectangle_xywh_px'], [0, 64, 100, 80])
        self.assertFalse(result['players']['review']['approved_intervals'])
        self.assertTrue(all(v['value'] is None for v in result['metrics'].values()))
        self.assertEqual(self.calls, ['badminton_proto.py', 'player_tracking.py'])
        calls = list(self.calls)
        second = self.run_fake(True)
        self.assertEqual(self.calls, calls)
        self.assertEqual(second['stages']['render']['directory'], job['stages']['render']['directory'])
        self.assertEqual(second['status'], 'needs_review')

    def test_tampered_stage_regenerates_and_preserves_previous_attempt(self):
        first = self.run_fake()
        old = self.output / first['stages']['players']['directory']
        (old / 'player_tracks.jsonl').write_text('tampered')
        second = self.run_fake(True)
        self.assertNotEqual(first['stages']['players']['directory'], second['stages']['players']['directory'])
        self.assertEqual((old / 'player_tracks.jsonl').read_text(), 'tampered')
        self.assertEqual(self.calls.count('player_tracking.py'), 2)

    def test_results_ui_change_reuses_detections_and_refreshes_render(self):
        first = self.run_fake()
        changed = {**code_binding(), 'session_results.html': 'changed-UI-fixture'}
        before = list(self.calls)
        with patch('session_runner.code_binding',return_value=changed):
            second = self.run_fake(True)
        self.assertEqual(self.calls,before)
        self.assertEqual(first['stages']['players']['directory'],second['stages']['players']['directory'])
        self.assertNotEqual(first['stages']['render']['directory'],second['stages']['render']['directory'])

    def test_configuration_change_recomputes_and_source_change_is_rejected(self):
        self.run_fake()
        self.config['processed_overlay'] = False
        self.run_fake(True)
        self.assertEqual(self.calls.count('player_tracking.py'), 2)
        self.video.write_bytes(b'different video')
        with patch('session_runner.probe_video', return_value={'frame_rate_fps': 10}):
            with self.assertRaisesRegex(ValueError, 'different source'):
                self.run_fake(True)

    def test_failed_refresh_invalidates_public_result(self):
        self.run_fake()
        self.config['threads'] = 2
        with patch('session_runner.execute', side_effect=RuntimeError('<unsafe> fixture failure')):
            with self.assertRaises(RuntimeError):
                run_job(self.video, self.output, self.config, self.weights, self.shuttle_weights, True)
        self.assertEqual(json.loads((self.output / 'job.json').read_text())['status'], 'failed')
        self.assertEqual(json.loads((self.output / 'result.json').read_text())['status'], 'stale')
        html = (self.output / 'index.html').read_text()
        self.assertIn('&lt;unsafe&gt;', html)
        self.assertNotIn('<unsafe>', html)

    def test_worker_cancellation_terminates_process_and_releases_lock(self):
        self.output.mkdir()
        timer = threading.Timer(.4, lambda: (self.output / 'CANCEL').touch())
        timer.start()
        try:
            with self.assertRaises(Cancelled):
                execute([sys.executable, '-c', 'import time; time.sleep(30)'], self.output / 'stage', self.output, dict(os.environ))
        finally:
            timer.cancel()
            timer.join()
        with job_lock(self.output):
            with self.assertRaisesRegex(ValueError, 'already running'):
                with job_lock(self.output):
                    pass
        with job_lock(self.output):
            pass

    def test_review_acceptance_and_revocation_invalidate_shuttle_and_preview(self):
        first = self.run_fake()
        players = self.output / first['stages']['players']['directory']
        decide(players, 0, 'accepted', 'Automated test fixture', '', dict.fromkeys(CHECKS, True), reviewer_kind='automated')
        accepted = self.run_fake(True)
        self.assertEqual(accepted['status'], 'completed')
        self.assertEqual(self.calls.count('shuttle_tracknet.py'), 1)
        before = accepted['stages']['render']['directory']
        decide(players, 0, 'rejected', 'Automated test fixture', 'Revocation test', {}, reviewer_kind='automated')
        revoked = self.run_fake(True)
        self.assertEqual(revoked['stages']['shuttle']['status'], 'not_available')
        self.assertNotEqual(revoked['stages']['render']['directory'], before)
        result = json.loads((self.output / 'result.json').read_text())
        self.assertIsNone(result['artifacts']['shuttle_summary'])
        self.assertFalse(result['players']['review']['approved_intervals'])
        self.assertEqual(self.calls.count('shuttle_tracknet.py'), 1)

    def test_job_cancellation_records_state_and_resume_clears_request(self):
        def cancel(command, directory, output, env):
            (output / 'CANCEL').touch()
            raise Cancelled('fixture cancellation')
        with patch('session_runner.execute', side_effect=cancel):
            with self.assertRaises(Cancelled):
                run_job(self.video, self.output, self.config, self.weights, self.shuttle_weights)
        job = json.loads((self.output / 'job.json').read_text())
        self.assertEqual(job['status'], 'cancelled')
        self.assertEqual(job['stages']['inspect']['status'], 'cancelled')
        resumed = self.run_fake(True)
        self.assertEqual(resumed['status'], 'needs_review')
        self.assertFalse((self.output / 'CANCEL').exists())

    def test_sample_hold_does_not_bridge_missing_interval(self):
        rows = [{'timestamp_seconds': 0}, {'timestamp_seconds': .4}]
        self.assertIs(observation_at(rows, [0, .4], .1, 5), rows[0])
        self.assertIsNone(observation_at(rows, [0, .4], .2, 5))
        self.assertIsNone(observation_at(rows, [0, .4], -.1, 5))
        self.assertIs(observation_at(rows, [0, .4], .4, 5), rows[1])

    def test_native_render_uses_sampling_grid_not_rounded_millisecond_rows(self):
        video = self.root / 'thirty-fps.mp4'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 30, (100, 80))
        for _ in range(6):
            writer.write(np.zeros((80,100,3),np.uint8))
        writer.release()
        rows = [{'frame_index':i,'timestamp_seconds':round(i/15,3),'persons':[]} for i in range(3)]
        target = self.root / 'render-grid'; target.mkdir()
        result = render_preview(video,target,{'sampling':{'fps':15,'start_seconds':0,'max_seconds':None}},rows,None)
        self.assertEqual(result['frames'],6)
        self.assertEqual(result['frames_with_sampled_player_observation'],6)
        self.assertEqual(result['review_samples'][-1]['frame_index'],5)

    def test_review_change_during_render_prevents_publication(self):
        def changed(video, directory, summary, rows, shuttle, cancel):
            result = render_preview(video,directory,summary,rows,shuttle,cancel)
            players = directory.parent / 'players-001'
            decide(players,0,'rejected','Automated test fixture','Concurrent revocation',{},reviewer_kind='automated')
            return result
        with patch('session_runner.execute',side_effect=self.fake_execute), patch('session_results.render_preview',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'Review changed'):
                run_job(self.video,self.output,self.config,self.weights,self.shuttle_weights)
        self.assertEqual(json.loads((self.output/'job.json').read_text())['status'],'failed')
        self.assertFalse((self.output/'result.json').exists())

    def test_invalid_configuration_and_cached_path_escape(self):
        for key, value in [('sample_fps', float('nan')), ('expected_players', True), ('max_seconds', -1), ('threads', 0)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_configuration({**self.config, key: value})
        with self.assertRaisesRegex(ValueError, 'escapes'):
            cached_stage(self.output, {'status': 'completed', 'fingerprint': 'x', 'directory': '..'}, 'x')


if __name__ == '__main__':
    unittest.main()
