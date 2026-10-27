import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import cv2
from src.core.scene import validate_polygon,validate_line,in_zone,capacity_events,FlowTracker
from src.utils.helpers import Detection, _parse_label_file
from src.core.video_processor import process_video,AnalysisCancelled
from src.core.detector import detect_heads_in_frame
from src.services.storage import atomic_json,session_inventory,check_storage_budget


class SceneTests(unittest.TestCase):
    def test_corrupted_annotations_are_identified(self):
        with tempfile.TemporaryDirectory() as folder:
            label=Path(folder)/'bad.txt'
            label.write_text('0 .5 .5 .1 .1\n0 .5 .5 .1 .1\n0 nan .5 .1 .1\n0.5 .5 .5 .1 .1\n')
            stats=_parse_label_file(str(label),1)
            self.assertEqual(stats['duplicate_rows'],1)
            self.assertLess(stats['valid_boxes'],4)
    def test_zone_counts_centres_and_includes_boundary(self):
        ds=[Detection(5,5,15,15,.9),Detection(75,75,85,85,.9),Detection(45,5,55,15,.9)]
        self.assertEqual(in_zone(ds,(100,100),[[0,0],[.5,0],[.5,1],[0,1]]),[ds[0],ds[2]])
    def test_invalid_geometry(self):
        for points in ([[0,0],[1,1],[0,1],[1,0]],[[0,0],[1,0]],[[0,0],[1,0],[float('nan'),1]]):
            with self.assertRaises(ValueError): validate_polygon(points)
        with self.assertRaises(ValueError): validate_line([[.5,.5],[.5,.5]])
    def test_sustained_event_not_one_frame_spike(self):
        counts=[5,11,12,13,10,50,0]; times=[0.,1.,2.,3.,4.,5.,6.]
        events=capacity_events(counts,times,10,2)
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]['start_seconds'],1)
        self.assertEqual(events[0]['end_seconds'],3)
        self.assertEqual(events[0]['peak_count'],13)
    def test_event_does_not_bridge_missing_time_or_extend_last_sample(self):
        self.assertEqual(capacity_events([20,20],[0.,10.],10,3),[])
        self.assertEqual(capacity_events([20],[0.],10,1),[])
        self.assertEqual(len(capacity_events([20]*4,[.2,1.2,2.2,3.1999999999999997],10,3)),1)
    def test_crossing_is_directional_and_confirmed(self):
        tracker=FlowTracker([[.5,.2],[.5,.8]])
        for i,x in enumerate([60,58,53,49,46,44,42]):
            tracker.update([Detection(x-2,48,x+2,52,.9)],(100,100),i*.2)
        self.assertEqual(tracker.forward,1)
        self.assertEqual(tracker.reverse,0)
        self.assertEqual(len(tracker.tracks),1)
    def test_crossing_outside_line_and_new_track_are_not_entries(self):
        tracker=FlowTracker([[.5,.2],[.5,.8]])
        for i,x in enumerate([60,58,53,49,46,44]):
            tracker.update([Detection(x-2,3,x+2,7,.9)],(100,100),i*.2)
        self.assertEqual(tracker.forward+tracker.reverse,0)
        tracker.update([Detection(38,48,42,52,.9)],(100,100),5.)
        self.assertEqual(tracker.forward+tracker.reverse,0)
    def test_mixed_head_person_model_uses_heads_only(self):
        class Model:
            names={0:'person',1:'head'}
            def predict(self,**kwargs):
                self.classes=kwargs['classes']; return []
        model=Model();detect_heads_in_frame(model,np.zeros((50,50,3),np.uint8))
        self.assertEqual(model.classes,[1])
    def test_storage_reserves_space_for_running_job(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            atomic_json(root/'sessions'/('a'*32)/('b'*24)/'status.json',{'state':'running'})
            with self.assertRaises(ValueError): check_storage_budget(root,1,limit=100)
    def test_inventory_reopens_valid_result(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); run=root/('a'*32)/('b'*24);run.mkdir(parents=True)
            out=run/'annotated.jpg';out.write_bytes(b'test')
            atomic_json(run/'result.json',{'annotated_path':str(out),'file_name':'test'})
            self.assertEqual(session_inventory(root)[0]['runs'][0]['file_name'],'test')


class PipelineSceneTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.source=self.root/'input.avi'
        writer=cv2.VideoWriter(str(self.source),cv2.VideoWriter_fourcc(*'MJPG'),10,(100,100))
        for i in range(40): writer.write(np.zeros((100,100,3),np.uint8))
        writer.release()
        self.model=type('Model',(),{'names':{0:'head'}})()
    def tearDown(self): self.temp.cleanup()
    def test_roi_and_events_integrated(self):
        ds=[Detection(5,5,15,15,.9),Detection(25,5,35,15,.9),Detection(75,75,85,85,.9)]
        with patch('src.core.video_processor.detect_heads_in_frame',return_value=ds):
            result=process_video('unused',str(self.source),preloaded_model=self.model,max_capacity=1,
                roi_polygon=[[0,0],[.5,0],[.5,1],[0,1]],event_seconds=2)
        self.assertEqual(result.max_head_count,2)
        self.assertEqual(len(result.events),1)
    def test_cancel_removes_partial_export(self):
        output=self.root/'result.mp4'
        with self.assertRaises(AnalysisCancelled):
            process_video('unused',str(self.source),preloaded_model=self.model,output_path=str(output),cancel_check=lambda:True)
        self.assertFalse(output.exists());self.assertFalse(list(self.root.glob('*.partial.mp4')))
    def test_duration_limit_before_inference(self):
        with self.assertRaisesRegex(ValueError,'limit'):
            process_video('unused',str(self.source),preloaded_model=self.model,max_duration_seconds=1)
    def test_variable_timestamps_use_decoder_time(self):
        class Capture:
            index=-1
            times=[1000,1080,1240,1500,1520,1900]
            def isOpened(self): return True
            def grab(self): self.index+=1; return self.index<len(self.times)
            def retrieve(self): return True,np.zeros((100,100,3),np.uint8)
            def release(self): pass
            def get(self,key):
                return {cv2.CAP_PROP_FPS:10,cv2.CAP_PROP_FRAME_COUNT:6,
                        cv2.CAP_PROP_FRAME_WIDTH:100,cv2.CAP_PROP_FRAME_HEIGHT:100,
                        cv2.CAP_PROP_POS_MSEC:self.times[min(max(self.index,0),5)]}.get(key,0)
        with patch('src.core.video_processor.cv2.VideoCapture',return_value=Capture()),patch('src.core.video_processor.detect_heads_in_frame',return_value=[]):
            result=process_video('unused',str(self.source),preloaded_model=self.model,analysis_fps=5)
        np.testing.assert_allclose(result.timestamps,[0,.24,.5,.9])
