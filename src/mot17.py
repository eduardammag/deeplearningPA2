"""Leitura do layout oficial MOTChallenge/MOT17."""
from pathlib import Path
import csv
from .geometry import xywh_to_xyxy
from .types import Detection


def read_mot_file(path):
    rows = []
    with Path(path).open(newline="") as stream:
        for row in csv.reader(stream):
            if not row or row[0].startswith("#"):
                continue
            rows.append([float(value) for value in row])
    return rows


def load_ground_truth(sequence_root, visible_threshold=0.0):
    frames = {}
    for row in read_mot_file(Path(sequence_root) / "gt" / "gt.txt"):
        frame, identity, x, y, width, height, confidence, cls, visibility = row[:9]
        if int(cls) != 1 or confidence <= 0 or visibility < visible_threshold:
            continue
        frames.setdefault(int(frame), []).append(Detection(int(frame), xywh_to_xyxy((x, y, width, height)), confidence, int(identity)))
    return frames


def load_public_detections(sequence_root, detector="FRCNN", min_score=0.0):
    path = Path(sequence_root) / "det" / f"det.txt"
    # Public MOT17 files are commonly copied as det.txt; detector is kept in metadata.
    frames = {}
    for row in read_mot_file(path):
        frame, _, x, y, width, height, score = row[:7]
        if score < min_score:
            continue
        frames.setdefault(int(frame), []).append(Detection(int(frame), xywh_to_xyxy((x, y, width, height)), score))
    return frames


def sequence_frames(sequence_root):
    return sorted((Path(sequence_root) / "img1").glob("*.jpg"))
