import json
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from academy_app import worker_arguments, resume_arguments, LocalResults, byte_range


class LauncherTests(unittest.TestCase):
    def test_worker_command_preserves_paths_and_limits_sampling_to_native_fps(self):
        with patch('academy_app.probe_video',return_value={'frame_rate_fps':10}):
            command=worker_arguments(Path('a video.mp4'),Path('out/new job'),4,True,False)
        self.assertEqual(command[command.index('--sample-fps')+1],'10')
        self.assertEqual(command[command.index('--video')+1],str(Path('a video.mp4').resolve()))
        self.assertIn('--processed-overlay',command)
        self.assertNotIn('--court-polygon',command)

    def test_demo_profile_is_bound_to_exact_source_and_does_not_apply_to_other_video(self):
        with patch('academy_app.probe_video',return_value={'frame_rate_fps':30}),patch('academy_app.sha256_file',return_value='0'*64):
            with self.assertRaisesRegex(ValueError,'only the supplied'):
                worker_arguments(Path('different.mp4'),Path('out/new'),4,False,True)
        with patch('academy_app.probe_video',return_value={'frame_rate_fps':30}),patch('academy_app.sha256_file',return_value=__import__('academy_app').PORTRAIT_DEMO_SHA256):
            command=worker_arguments(Path('reference.mp4'),Path('out/new'),4,False,True)
        self.assertIn('--court-polygon',command)
        self.assertIn('--secondary-detect-roi',command)
        self.assertIn('--processed-overlay',command)

    def test_resume_restores_original_configuration_and_checkpoint_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            job={'schema_version':'1.0.0','binding':{'source':{'path':'C:/video with spaces.mp4'},
                'models':{'person':{'path':'C:/models/rf-detr-nano.pth'},'shuttle':{'path':'C:/models/shuttle.pt'}},
                'configuration':{'camera':'court-2','sample_fps':15,'expected_players':4,'threads':4,'processed_overlay':True,
                                 'roi':None,'court_polygon':'0,0;1,0;1,1;0,1'}}}
            (root/'job.json').write_text(json.dumps(job))
            command=resume_arguments(root)
        self.assertIn('--resume',command)
        self.assertIn('--court-polygon',command)
        self.assertNotIn('--roi',command)
        self.assertEqual(command[command.index('--person-weights')+1],'C:/models/rf-detr-nano.pth')

    def test_read_only_results_handler_rejects_escapes_and_hidden_files(self):
        class Server:
            def __init__(self,address,handler):
                self.address=address;self.handler=handler;self.server_port=12345
            def serve_forever(self): pass
            def shutdown(self): pass
            def server_close(self): pass
        with tempfile.TemporaryDirectory() as temporary,patch('academy_app.ThreadingHTTPServer',Server):
            root=Path(temporary);results=LocalResults()
            self.assertEqual(results.url(root),'http://127.0.0.1:12345/')
            self.assertEqual(results.server.address[0],'127.0.0.1')
            handler=results.server.handler.__new__(results.server.handler)
            handler.path='/index.html';handler.send_error=Mock()
            for path in (root.parent/'outside.mp4',root/'.secret'):
                handler.translate_path=lambda value,p=path:str(p)
                self.assertIsNone(handler.send_head())
            self.assertEqual(handler.send_error.call_count,2)
            self.assertIsNone(handler.list_directory(str(root)))
            video=root/'preview.mp4';video.write_bytes(b'0123456789')
            handler.translate_path=lambda value:str(video)
            handler.headers={'Range':'bytes=2-5'}
            handler.send_response=Mock();handler.send_header=Mock();handler.end_headers=Mock()
            stream=handler.send_head();target=io.BytesIO()
            try: handler.copyfile(stream,target)
            finally: stream.close()
            self.assertEqual(target.getvalue(),b'2345')
            handler.send_response.assert_called_with(206)
            handler.send_header.assert_any_call('Content-Range','bytes 2-5/10')
            handler.headers={'Range':'bytes=99-'}
            self.assertIsNone(handler.send_head())
            handler.send_response.assert_called_with(416)
            results.close()

    def test_video_byte_ranges_include_suffixes_and_reject_malformed_or_empty_ranges(self):
        for header,expected in [('bytes=2-5',(2,5)),('bytes=8-99',(8,9)),('bytes=8-',(8,9)),('bytes=-3',(7,9)),('bytes=-99',(0,9))]:
            self.assertEqual(byte_range(header,10),expected)
        for header in ('bytes=-','bytes=-0','bytes=10-','bytes=5-2','bytes=1-2,4-5','invalid'):
            with self.subTest(header=header),self.assertRaises(ValueError): byte_range(header,10)


if __name__=='__main__': unittest.main()
