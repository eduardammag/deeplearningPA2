"""Baseline e tracker temporal, ambos sem tracker de terceiros."""
from .association import IoUTracker
from .geometry import center
from .types import Detection


def run_baseline(detections_by_frame, matcher="greedy", iou_threshold=0.3, max_missed=3):
    tracker = IoUTracker(iou_threshold, max_missed, matcher)
    predictions = []
    for frame in sorted(detections_by_frame):
        predictions.append([{"track_id": track.track_id, "bbox": track.bbox, "gt_id": track.gt_id}
                            for track in tracker.update(detections_by_frame[frame])])
    return predictions


def run_temporal(detections_by_frame, model, iou_threshold=0.1, max_missed=5):
    """Rollout simples: usa deslocamento recente como entrada e a RNN para prever caixa."""
    import torch
    from .temporal import box_to_features
    tracker = IoUTracker(iou_threshold, max_missed)
    predictions = []
    for frame in sorted(detections_by_frame):
        for track in tracker.tracks:
            features = torch.stack([box_to_features(box) for box in track.history[-8:]])[None]
            with torch.no_grad():
                predicted, _ = model(features)
            values = predicted[0, -1].tolist()
            cx, cy, width, height = values
            track.bbox = (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)
        predictions.append([{"track_id": track.track_id, "bbox": track.bbox, "gt_id": track.gt_id}
                            for track in tracker.update(detections_by_frame[frame])])
    return predictions
