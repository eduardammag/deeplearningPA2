"""Experimentos de ablação, horizonte e teste de estresse."""
import numpy as np
from .synthetic import corrupt_detections
from .tracking import run_baseline
from metrics import evaluate_tracking


def stress_detector(truth, intensities=((0.0, 0.0, 0.0), (0.2, 1.0, 0.2), (0.4, 3.0, 0.5)), seed=0):
    results = []
    gt_records = [[{"gt_id": item.gt_id, "bbox": item.bbox, "track_id": item.gt_id} for item in frame] for frame in truth]
    for drop, noise, false_positive in intensities:
        detections = corrupt_detections(truth, drop, noise, false_positive, seed=seed)
        by_frame = {index: frame for index, frame in enumerate(detections)}
        predictions = run_baseline(by_frame)
        results.append({"drop": drop, "noise": noise, "false_positive": false_positive,
                        **evaluate_tracking(predictions, gt_records)})
    return results


def ablation_cells(sequence, cells=("rnn", "lstm", "gru"), seeds=(0, 1, 2)):
    # The CLI writes metrics for each seed; training is intentionally delegated to train.py.
    return [{"cell": cell, "seed": seed, "sequence_length": len(sequence)} for cell in cells for seed in seeds]
