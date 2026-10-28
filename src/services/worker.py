"""Background analysis process. No Streamlit calls or shared model predictor state."""
from pathlib import Path
from dataclasses import asdict
from datetime import datetime, timezone
import json
import sys
import time
import cv2
import torch
from filelock import FileLock
import config
from src.core.detector import load_model, run_detection
from src.core.video_processor import process_video, AnalysisCancelled
from src.services.storage import atomic_json, check_storage_budget
from src.utils.helpers import get_crowd_level


def run(run_dir, preloaded_model=None):
    run_dir=Path(run_dir)
    job=json.loads((run_dir/'job.json').read_text(encoding='utf-8'))
    settings=job['settings']; start=time.perf_counter(); last=[0.]
    torch.set_num_threads(2)
    def status(state, **values):
        atomic_json(run_dir/'status.json',dict(state=state,updated=time.time(),**values))
    def cancelled(): return (run_dir/'cancel').exists()
    def progress(index,total,frame,count,level,auto):
        now=time.perf_counter()
        if now-last[0]>.8:
            last[0]=now
            thumb=cv2.resize(frame,(640,max(2,int(frame.shape[0]*640/frame.shape[1]))))
            cv2.imwrite(str(run_dir/'preview.tmp.jpg'),thumb,[cv2.IMWRITE_JPEG_QUALITY,70])
            (run_dir/'preview.tmp.jpg').replace(run_dir/'preview.jpg')
            status('running',progress=min(index/total,.99) if total else 0.,
                   message=f'Frame {index:,} / {total:,} | {count} heads in area',elapsed_seconds=now-start)
    try:
        with FileLock(str(Path(config.OUTPUTS_ROOT)/'.analysis.lock'),timeout=0):
            status('running',progress=0.,message='Loading detector')
            check_storage_budget(config.OUTPUTS_ROOT,0)
            model=preloaded_model if preloaded_model is not None else load_model(settings['weights_path'])
            common=dict(weights_path=settings['weights_path'],source_path=job['source_path'],
                        imgsz=settings['imgsz'],preloaded_model=model,device='cpu',
                        max_capacity=settings['max_capacity'],roi_polygon=settings['roi_polygon'])
            if cancelled(): raise AnalysisCancelled('Analysis cancelled.')
            if job['media_type']=='video':
                output=run_dir/'annotated.mp4'
                result=process_video(**common,conf_threshold=settings['conf'],iou_threshold=.5,
                                     output_path=str(output),analysis_fps=settings['analysis_fps'],
                                     max_width=settings['max_width'],flow_line=settings.get('flow_line'),
                                     event_seconds=settings['event_seconds'],frame_callback=progress,cancel_check=cancelled,
                                     zones=settings.get('zones'))
                data=asdict(result)
                data['per_frame_levels']=[get_crowd_level(n,settings['max_capacity']) for n in result.per_frame_counts]
                cap=cv2.VideoCapture(str(output))
                try:
                    # Store a bounded number of evidence thumbnails, ordered by event severity.
                    for i,event in enumerate(sorted(data['events'],key=lambda e:e['peak_count'],reverse=True)[:12]):
                        cap.set(cv2.CAP_PROP_POS_MSEC,event['peak_seconds']*1000)
                        ok,frame=cap.read()
                        if ok:
                            image=run_dir/f'event_{i:02d}.jpg'
                            frame=cv2.resize(frame,(480,max(2,int(frame.shape[0]*480/frame.shape[1]))))
                            cv2.imwrite(str(image),frame,[cv2.IMWRITE_JPEG_QUALITY,75])
                            event['thumbnail']=str(image)
                finally: cap.release()
            else:
                output=run_dir/'annotated.jpg'
                annotated,count,level,_,_=run_detection(**common,conf=settings['conf'],iou=.5)
                if not cv2.imwrite(str(output),annotated,[cv2.IMWRITE_JPEG_QUALITY,88]):
                    raise IOError('Cannot write annotated image.')
                data=dict(max_head_count=count,avg_head_count=float(count),final_head_count=count,
                          final_crowd_level=level,peak_crowd_level=level,per_frame_counts=[count],
                          per_frame_levels=[level],timestamps=[0.],processed_frames=1,
                          input_bytes=Path(job['source_path']).stat().st_size,output_bytes=output.stat().st_size,
                          processing_time_seconds=time.perf_counter()-start,events=[],crossings=[],
                          forward_crossings=0,reverse_crossings=0,source_duration=0,source_fps=None,sample_fps=None)
            if cancelled():
                output.unlink(missing_ok=True)
                raise AnalysisCancelled('Analysis cancelled.')
            record=dict(data,signature=run_dir.name,id=run_dir.name,file_name=job['file_name'],
                        media_type=job['media_type'],annotated_path=str(output),source_path=job['source_path'],
                        settings=settings,model_name=str(model.model.yaml.get('yaml_file','YOLO')),
                        model_classes=', '.join(model.names.values()),crowd_level=data['peak_crowd_level'],
                        created_at=datetime.now(timezone.utc).isoformat())
            atomic_json(run_dir/'result.json',record)
            status('complete',progress=1.,message='Analysis complete')
    except AnalysisCancelled as exc:
        status('cancelled',error=str(exc))
    except Exception as exc:
        status('failed',error=str(exc))
    finally:
        (run_dir/'preview.tmp.jpg').unlink(missing_ok=True)

if __name__=='__main__': run(sys.argv[1])
