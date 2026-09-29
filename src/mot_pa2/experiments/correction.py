"""Paired, post-hoc inspection of a gallery failure with a fixed lifespan change."""
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from mot_pa2.data.mot17 import load_ground_truth, load_public_detections, sequence_info, sequence_frames
from mot_pa2.evaluation.metrics import spatial_matches, overlap
from mot_pa2.evaluation.pipeline import truth_records
from mot_pa2.evaluation.serialization import save_metrics
from mot_pa2.tracking.trackers import run_temporal
from mot_pa2.visualization.annotations import annotate


def identity_trace(predictions, truth, identity, start, end):
    """Follow the matched identity and the original track separately through a gap."""
    original = None
    trace = []
    for frame in range(start, end + 1):
        predicted, targets = predictions.get(frame, []), truth.get(frame, [])
        matches = spatial_matches(predicted, targets)
        matched = next((predicted[p]["track_id"] for p, g in matches
                        if targets[g]["gt_id"] == identity), None)
        if frame == start:
            original = matched
        track = next((p for p in predicted if p["track_id"] == original), None)
        target = next((g for g in targets if g["gt_id"] == identity), None)
        trace.append(dict(frame=frame, matched_track_id=matched,
                          original_track_alive=track is not None,
                          missed=track["missed"] if track else None,
                          original_track_iou=overlap(track["bbox"], target["bbox"]) if track and target else None))
    return dict(original_track_id=original, trace=trace)


def correction_case(model, root, case, output):
    """Inspect the first gallery case without choosing a new final configuration."""
    root, output = Path(root), Path(output)
    info = sequence_info(root)
    truth = truth_records(load_ground_truth(root))
    detections = load_public_detections(root)
    start, event = case["previous_match_frame"], case["frame"]
    end = min(info["frames"], event + 2)
    identity = case["gt_id"]
    predictions, variants = {}, []
    for lifetime in (3, 16):
        predictions[lifetime] = run_temporal(detections, model, max_missed=lifetime,
                                            image_size=(info["width"], info["height"]))
        trace = identity_trace(predictions[lifetime], truth, identity, start, end)
        current = next(t for t in trace["trace"] if t["frame"] == event)
        retained = (trace["original_track_id"] is not None
                    and current["matched_track_id"] == trace["original_track_id"])
        variants.append(dict(max_missed=lifetime, same_id_at_event=retained, **trace))
    outcome = ("The original ID is recovered at the selected event after extending track lifetime."
               if variants[1]["same_id_at_event"] else
               "Extending lifetime does not recover the original ID at the selected event; inspect the paired IoU and missed-observation trace.")
    report = dict(sequence=root.name, gallery_figure=case["figure"], gt_id=identity,
                  event_frame=event, previous_match_frame=start,
                  protocol="Post-hoc illustration of fixed lifespan 3->16 on the first gallery case; not model selection or a new held-out benchmark",
                  diagnosis=case["diagnosis"], variants=variants, conclusion=outcome,
                  validation_result="correction.json", figure="correction_case.png")
    save_metrics(report, output / "correction_case.json")
    paths = sequence_frames(root)
    frames = [start, min(start + 1, event), event, end]
    target_boxes = [g["bbox"] for frame in frames for g in truth[frame] if g["gt_id"] == identity]
    x1 = max(0, min(b[0] for b in target_boxes) - 100)
    y1 = max(0, min(b[1] for b in target_boxes) - 100)
    x2 = min(info["width"], max(b[2] for b in target_boxes) + 100)
    y2 = min(info["height"], max(b[3] for b in target_boxes) + 100)
    fig, axes = plt.subplots(3, len(frames), figsize=(16, 9))
    for col, frame in enumerate(frames):
        with Image.open(paths[frame - 1]) as image:
            rgb = np.asarray(image.convert("RGB"))
        layers = [[g for g in truth[frame] if g["gt_id"] == identity]]
        for variant in variants:
            row = next(t for t in variant["trace"] if t["frame"] == frame)
            ids = {variant["original_track_id"], row["matched_track_id"]}
            layers.append([p for p in predictions[variant["max_missed"]][frame] if p["track_id"] in ids])
        for row, items in enumerate(layers):
            axes[row, col].imshow(annotate(rgb, items))
            axes[row, col].set_xlim(x1, x2)
            axes[row, col].set_ylim(y2, y1)
            axes[row, col].set_title(f"{['GT', 'Before: lifespan 3', 'After: lifespan 16'][row]} / frame {frame}")
            axes[row, col].axis("off")
    fig.suptitle(f"{root.name}: GT {identity}, same event {event}, fixed correction")
    fig.tight_layout()
    fig.savefig(output / "correction_case.png", dpi=100)
    plt.close(fig)
    return report
