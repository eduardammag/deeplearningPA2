"""Associacao de deteccoes escrita para este projeto."""
from .geometry import iou
from .types import Detection, Track


def greedy_matches(tracks, detections, threshold=0.3):
    candidates = sorted(((iou(track.bbox, detection.bbox), ti, di)
                         for ti, track in enumerate(tracks)
                         for di, detection in enumerate(detections)), reverse=True)
    matches, used_tracks, used_detections = [], set(), set()
    for score, ti, di in candidates:
        if score < threshold or ti in used_tracks or di in used_detections:
            continue
        matches.append((ti, di, score))
        used_tracks.add(ti)
        used_detections.add(di)
    return matches, [i for i in range(len(tracks)) if i not in used_tracks], [i for i in range(len(detections)) if i not in used_detections]


def hungarian_matches(tracks, detections, threshold=0.3):
    """Usa scipy apenas para a atribuicao; o tracker e a gestao de tracks sao proprios."""
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return greedy_matches(tracks, detections, threshold)
    import numpy as np
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))
    scores = np.array([[iou(track.bbox, det.bbox) for det in detections] for track in tracks])
    weights = np.where(scores >= threshold, min(len(tracks), len(detections)) + 1 + scores, 0)
    rows, cols = linear_sum_assignment(weights, maximize=True)
    matches = [(int(row), int(col), float(scores[row, col])) for row, col in zip(rows, cols)
               if scores[row, col] >= threshold]
    used_t, used_d = {m[0] for m in matches}, {m[1] for m in matches}
    return matches, [i for i in range(len(tracks)) if i not in used_t], [i for i in range(len(detections)) if i not in used_d]


class IoUTracker:
    def __init__(self, iou_threshold=0.3, max_missed=3, matcher="greedy"):
        self.iou_threshold, self.max_missed = iou_threshold, max_missed
        self.matcher = hungarian_matches if matcher == "hungarian" else greedy_matches
        self.tracks, self.next_id = [], 1

    def update(self, detections):
        matches, unmatched_tracks, unmatched_detections = self.matcher(self.tracks, detections, self.iou_threshold)
        for ti, di, _ in matches:
            track, detection = self.tracks[ti], detections[di]
            track.bbox, track.hits, track.missed, track.age, track.gt_id = detection.bbox, track.hits + 1, 0, track.age + 1, detection.gt_id
            track.history.append(track.bbox)
        for ti in unmatched_tracks:
            self.tracks[ti].missed += 1
            self.tracks[ti].age += 1
            self.tracks[ti].history.append(self.tracks[ti].bbox)
        self.tracks = [track for track in self.tracks if track.missed <= self.max_missed]
        for di in unmatched_detections:
            detection = detections[di]
            self.tracks.append(Track(self.next_id, detection.bbox, gt_id=detection.gt_id))
            self.next_id += 1
        return list(self.tracks)

    def reset(self):
        self.tracks, self.next_id = [], 1
