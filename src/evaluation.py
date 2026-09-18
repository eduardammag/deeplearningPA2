"""Execucao de avaliacao e tabelas serializaveis."""
import json
from pathlib import Path
from metrics import evaluate_tracking
from .tracking import run_baseline


def align_ground_truth(gt_by_frame, frames):
    return [[{"gt_id": item.gt_id, "bbox": item.bbox, "track_id": item.gt_id} for item in gt_by_frame.get(frame, [])]
            for frame in frames]


def evaluate_baseline(detections_by_frame, gt_by_frame, matcher="greedy", iou_threshold=0.3, max_missed=3):
    frames = sorted(set(detections_by_frame) | set(gt_by_frame))
    predictions = run_baseline(detections_by_frame, matcher, iou_threshold, max_missed)
    truth = align_ground_truth(gt_by_frame, frames)
    return evaluate_tracking(predictions, truth), predictions, truth


def save_metrics(metrics, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
