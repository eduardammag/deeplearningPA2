"""Evaluation on a common frame clock."""
import json
from pathlib import Path
from metrics import evaluate_tracking,detection_map
from .tracking import run_baseline,run_temporal

def truth_records(gt):
    return {f:[dict(gt_id=x.gt_id,bbox=x.bbox) for x in items] for f,items in gt.items()}

def detection_records(det):
    return {f:[dict(bbox=x.bbox,score=x.score) for x in items] for f,items in det.items()}

def evaluate_baseline(detections_by_frame,gt_by_frame,matcher="greedy",iou_threshold=.3,max_missed=3):
    frames=sorted(set(detections_by_frame)|set(gt_by_frame))
    detections={f:detections_by_frame.get(f,[]) for f in frames}
    predictions=run_baseline(detections,matcher,iou_threshold,max_missed)
    truth=truth_records(gt_by_frame)
    return evaluate_tracking(predictions,truth),predictions,truth

def evaluate_model(detections,gt,model,image_size,max_missed=3):
    frames=sorted(set(detections)|set(gt))
    predictions=run_temporal({f:detections.get(f,[]) for f in frames},model,
                             image_size=image_size,max_missed=max_missed)
    return evaluate_tracking(predictions,truth_records(gt)),predictions

def save_metrics(metrics,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(metrics,indent=2),encoding="utf-8")
