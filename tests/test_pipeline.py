"""Regression tests for size control, timekeeping, failure handling and report consistency."""
import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from src.core.video_processor import process_video
from src.core.encoding import ffmpeg_executable, CompactVideoWriter
from src.utils.helpers import Detection, BoxSmoother, resolve_class_ids, get_crowd_level
from src.utils.privacy import redact_sensitive
from src.services.storage import atomic_json, clear_session, retain_run
from src.services.report_generator import generate_pdf_report
from src.services.exports import csv_bytes

class DummyModel:
    names = {0:'head'}

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.source=self.root/'input.avi'
        writer=cv2.VideoWriter(str(self.source),cv2.VideoWriter_fourcc(*'MJPG'),30,(320,240))
        for i in range(60):
            frame=np.zeros((240,320,3),dtype=np.uint8)
            cv2.rectangle(frame,(i*3,30),(i*3+40,100),(255,160,50),-1)
            writer.write(frame)
        writer.release()
    def tearDown(self):
        self.temp.cleanup()
    def test_sampling_keeps_source_time_and_compresses(self):
        output=self.root/'annotated.mp4'
        with patch('src.core.video_processor.detect_heads_in_frame',return_value=[Detection(10,10,25,25,.9)]):
            result=process_video('unused',str(self.source),preloaded_model=DummyModel(),
                                 output_path=str(output),analysis_fps=5,max_width=160)
        self.assertEqual(result.processed_frames,10)
        self.assertEqual(result.timestamps,[i/5 for i in range(10)])
        self.assertEqual(result.source_fps,30)
        self.assertEqual(result.output_width,160)
        self.assertLess(result.output_bytes,self.source.stat().st_size*1.5+2*1024**2)
        cap=cv2.VideoCapture(str(output))
        self.assertTrue(cap.isOpened())
        self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS),5,places=2)
        self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT)/5,2,places=2)
        cap.release()
        self.assertFalse(list(self.root.glob('*.partial.mp4')))
    def test_failure_is_not_safe_and_partial_output_removed(self):
        output=self.root/'bad.mp4'
        with patch('src.core.video_processor.detect_heads_in_frame',side_effect=RuntimeError('failure')):
            with self.assertRaisesRegex(RuntimeError,'Detection failed'):
                process_video('unused',str(self.source),preloaded_model=DummyModel(),output_path=str(output))
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob('*.partial.mp4')))
    def test_untracked_smoothing_does_not_move_boxes(self):
        smoother=BoxSmoother()
        smoother.smooth([Detection(0,0,10,10,.9),Detection(100,0,110,10,.9)])
        result=smoother.smooth([Detection(100,0,110,10,.9),Detection(0,0,10,10,.9)])
        self.assertEqual([(d.x1,d.x2) for d in result],[(100,110),(0,10)])
    def test_unknown_class_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_class_ids({0:'chair'},('head','person'))
    def test_credentials_are_redacted_recursively(self):
        record={'file_name':'rtsp://user:password@camera:554/stream?token=abc','settings':{'password':'secret'}}
        clean=redact_sensitive(record)
        self.assertNotIn('user:',str(clean)); self.assertNotIn('abc',str(clean))
        self.assertEqual(clean['settings']['password'],'[redacted]')
    def test_atomic_run_storage_and_pruning(self):
        for key in ['a'*24,'b'*24]:
            atomic_json(self.root/key/'result.json',{'value':1})
        unrelated=self.root/'manual'; unrelated.mkdir()
        retain_run(self.root,'b'*24)
        self.assertFalse((self.root/('a'*24)).exists())
        self.assertTrue(unrelated.exists())
        with self.assertRaises(ValueError):
            clear_session(self.root,'../escape')
    def test_report_handles_user_filename_and_records_peak(self):
        output=self.root/'report.pdf'
        generate_pdf_report({'file_name':'<crowd & test>.mp4','max_head_count':12,'avg_head_count':4,
                             'final_head_count':1,'peak_crowd_level':'High Crowd','final_crowd_level':'Low Crowd',
                             'model_name':'yolo11m','settings':{}},str(output))
        self.assertTrue(output.read_bytes().startswith(b'%PDF'))
    def test_input_cannot_be_overwritten(self):
        with self.assertRaises(ValueError):
            process_video('unused',str(self.source),output_path=str(self.source))

    def test_encoder_rejects_oversize_and_removes_partial(self):
        path=self.root/'oversize.mp4'
        writer=CompactVideoWriter(str(path),320,240,5,100000,max_bytes=1)
        writer.write(np.zeros((240,320,3),dtype=np.uint8))
        with self.assertRaisesRegex(RuntimeError,'size budget'):
            writer.close()
        self.assertFalse(path.exists())
        self.assertFalse(list(self.root.glob('*.partial.mp4')))
    def test_csv_uses_original_source_timestamps_and_levels(self):
        record={'per_frame_counts':[1,3], 'timestamps':[0.0,0.2],
                'per_frame_levels':['Low Crowd','Moderate Crowd']}
        rows=list(csv.DictReader(io.StringIO(csv_bytes(record).decode())))
        self.assertEqual(rows[1]['source_time_seconds'],'0.2')
        self.assertEqual(rows[1]['crowd_level'],'Moderate Crowd')
        record['timestamps']=[]
        with self.assertRaises(ValueError):
            csv_bytes(record)
    def test_capacity_boundaries(self):
        self.assertEqual(get_crowd_level(60,100),'Low Crowd')
        self.assertEqual(get_crowd_level(61,100),'Moderate Crowd')
        self.assertEqual(get_crowd_level(85,100),'Moderate Crowd')
        self.assertEqual(get_crowd_level(86,100),'High Crowd')

if __name__=='__main__':
    unittest.main()
