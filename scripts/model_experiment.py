"""Reproducible CPU fine-tuning pilot and count-error promotion gate.

Local dataset remains private. A new checkpoint is never installed automatically.
The existing model's training provenance is unknown: this is a local comparison,
not proof of generalization to unseen college cameras.
"""
from pathlib import Path
import argparse
import hashlib
import json
import random
import sys
import time
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from src.core.detector import detect_heads_in_frame
from src.services.storage import atomic_json


def audit(dataset):
    splits, seen, issues, excluded = {}, {}, [], []
    for split in ('train', 'val', 'test'):
        rows = []
        for image in sorted((dataset/'images'/split).glob('*')):
            if image.suffix.lower() not in ('.jpg', '.png', '.jpeg'):
                continue
            label = dataset/'labels'/split/(image.stem+'.txt')
            if not label.is_file():
                issues.append(f'{split}/{image.name}: missing label')
                continue
            boxes = []
            for line in label.read_text().splitlines():
                if not line.strip():
                    continue
                values = list(map(float, line.split()))
                if len(values) != 5 or values[0] != 0 or not all(np.isfinite(values)) or not all(0 <= v <= 1 for v in values[1:]) or min(values[3:]) <= 0:
                    issues.append(f'{split}/{image.name}: invalid label')
                    break
                boxes.append(values)
            unique = {tuple(box) for box in boxes}
            if len(boxes) > 1 and (len(boxes)-len(unique))/len(boxes) > .05:
                issues.append(f'{split}/{image.name}: repeated boxes indicate corrupt annotation conversion')
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in seen:
                excluded.append(f'{split}/{image.name}: duplicate image of {seen[digest]}')
                continue
            seen[digest] = f'{split}/{image.name}'
            rows.append(dict(path=str(image.resolve()), count=len(boxes), sha256=digest))
        splits[split] = rows
    if issues:
        raise ValueError('Dataset audit failed: '+ '; '.join(issues[:12]))
    if len(splits['val']) < 112:
        raise ValueError('Need at least 112 validation images for disjoint tuning/comparison subsets.')
    splits['excluded_duplicates']=excluded
    return splits


def evaluate(weights, rows, thresholds, imgsz=640, device='cpu'):
    model = YOLO(str(weights))
    samples = []
    start = time.perf_counter()
    for i, row in enumerate(rows):
        frame = cv2.imread(row['path'])
        if frame is None:
            raise ValueError('Unreadable evaluation image: '+row['path'])
        detections = detect_heads_in_frame(model, frame, conf=min(thresholds), iou=.5, imgsz=imgsz,
                                          device=device, class_ids=[0], native_head_class=True)
        label=Path(row['path']).parent.parent.parent/'labels'/Path(row['path']).parent.name/(Path(row['path']).stem+'.txt')
        truth=[]
        h,w=frame.shape[:2]
        for line in label.read_text().splitlines():
            if line.strip():
                _,x,y,bw,bh=map(float,line.split()); truth.append([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
        hits={}
        for threshold in thresholds:
            matched=set(); tp=0
            for d in sorted((d for d in detections if d.confidence>=threshold),key=lambda d:-d.confidence):
                best_i=None; best_iou=.5
                for j,(x1,y1,x2,y2) in enumerate(truth):
                    if j in matched: continue
                    intersection=max(0,min(x2,d.x2)-max(x1,d.x1))*max(0,min(y2,d.y2)-max(y1,d.y1))
                    union=(x2-x1)*(y2-y1)+(d.x2-d.x1)*(d.y2-d.y1)-intersection
                    iou=intersection/union if union else 0
                    if iou>=best_iou: best_iou=iou; best_i=j
                if best_i is not None: matched.add(best_i); tp+=1
            hits[str(threshold)]=tp
        samples.append(dict(image=Path(row['path']).name, truth=row['count'],
                            true_positives=hits,
                            predictions={str(t): sum(d.confidence >= t for d in detections) for t in thresholds}))
        if (i+1) % 8 == 0:
            print(f'Evaluated {i+1}/{len(rows)}: {Path(weights).name}', flush=True)
    metrics = {}
    for threshold in thresholds:
        errors = np.array([r['predictions'][str(threshold)] - r['truth'] for r in samples])
        tp=sum(r['true_positives'][str(threshold)] for r in samples)
        predicted=sum(r['predictions'][str(threshold)] for r in samples); truth=sum(r['truth'] for r in samples)
        metrics[str(threshold)] = dict(mae=float(abs(errors).mean()), rmse=float(np.sqrt((errors**2).mean())),
                                      bias=float(errors.mean()), images=len(samples),precision=tp/max(1,predicted),recall=tp/max(1,truth))
    return dict(metrics=metrics, samples=samples, seconds=time.perf_counter()-start, imgsz=imgsz)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--train-images', type=int, default=256)
    parser.add_argument('--name', default='campus_pilot_v1')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--batch',type=int,default=2)
    parser.add_argument('--freeze',type=int,default=22)
    parser.add_argument('--imgsz',type=int,default=640)
    parser.add_argument('--val-images',type=int,default=48)
    parser.add_argument('--test-images',type=int,default=64)
    parser.add_argument('--patience',type=int,default=8)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.device!='cpu' and not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for this run. Select a GPU runtime before continuing.')
    experiment = ROOT/'runs'/'experiments'/args.name
    experiment.mkdir(parents=True, exist_ok=False)
    baseline = ROOT/'models'/'head_detector.pt'
    data = audit(args.dataset)
    rng = random.Random(20261027)
    rng.shuffle(data['train']); rng.shuffle(data['val'])
    train = data['train'][:args.train_images] if args.train_images else data['train']
    tune = data['val'][:args.val_images] if args.val_images else data['val']
    rng.shuffle(data['test'])
    holdout = (data['test'][:args.test_images] if args.test_images else data['test']) if data['test'] else data['val'][len(tune):len(tune)+64]
    if not holdout: raise ValueError('A separate comparison split is required.')
    manifest = dict(seed=20261027, train=train, tune=tune, holdout=holdout,
                    available={k:len(v) for k,v in data.items()}, excluded_duplicates=data['excluded_duplicates'], baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
                    limitation='Existing checkpoint training exposure unknown; not an independently unseen test set.')
    atomic_json(experiment/'manifest.json', manifest)
    for name, rows in [('train',train), ('val',tune)]:
        (experiment/f'{name}.txt').write_text('\n'.join(row['path'] for row in rows),encoding='utf-8')
    dataset_yaml=experiment/'data.yaml'
    dataset_yaml.write_text(yaml.safe_dump(dict(train=str(experiment/'train.txt'),val=str(experiment/'val.txt'),names={0:'head'},nc=1)),encoding='utf-8')
    atomic_json(experiment/'status.json', dict(stage='training', epochs=args.epochs, train_images=len(train)))
    model = YOLO(str(baseline))
    model.train(data=str(dataset_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=2 if args.device!='cpu' else 0,
                device=args.device, freeze=args.freeze, optimizer='AdamW', lr0=.0001, lrf=.1,
                mosaic=0, mixup=0, close_mosaic=0, degrees=0, translate=.05, scale=.15,
                fliplr=.5, seed=20261027, deterministic=True, amp=args.device!='cpu',
                project=str(experiment), name='training', plots=False, save=True, cache=False,
                max_det=2000, patience=args.patience, save_period=5)
    candidate=experiment/'training'/'weights'/'best.pt'
    thresholds=[.15,.25,.35,.45]
    atomic_json(experiment/'status.json',dict(stage='threshold_tuning'))
    baseline_tune=evaluate(baseline,tune,thresholds,device=args.device)
    candidate_tune=evaluate(candidate,tune,thresholds,device=args.device)
    baseline_conf=float(min(baseline_tune['metrics'],key=lambda t:baseline_tune['metrics'][t]['mae']))
    candidate_conf=float(min(candidate_tune['metrics'],key=lambda t:candidate_tune['metrics'][t]['mae']))
    atomic_json(experiment/'status.json',dict(stage='heldout_comparison'))
    baseline_test=evaluate(baseline,holdout,[baseline_conf],device=args.device)
    candidate_test=evaluate(candidate,holdout,[candidate_conf],device=args.device)
    old=baseline_test['metrics'][str(baseline_conf)]; new=candidate_test['metrics'][str(candidate_conf)]
    eligible=new['mae'] < old['mae']*.98 and new['rmse'] <= old['rmse'] and new['precision']>=old['precision']-.02 and new['recall']>=old['recall']-.02
    comparison=dict(status='complete',eligible_for_promotion=eligible,baseline_confidence=baseline_conf,
                    candidate_confidence=candidate_conf,baseline=old,candidate=new,
                    baseline_tuning=baseline_tune,candidate_tuning=candidate_tune,
                    baseline_test=baseline_test,candidate_test=candidate_test,
                    candidate_path=str(candidate),train_images=len(train),epochs=args.epochs,
                    limitation=manifest['limitation'])
    atomic_json(experiment/'comparison.json', comparison)
    atomic_json(experiment/'status.json',dict(stage='complete',eligible_for_promotion=eligible))
    print(json.dumps({k:comparison[k] for k in ('eligible_for_promotion','baseline','candidate')},indent=2),flush=True)

if __name__=='__main__':
    main()
