"""Inference without ground truth or training, with persistent tracking state."""
from pathlib import Path

from mot_pa2.data.mot17 import load_public_detections
from mot_pa2.data.video import open_frames
from mot_pa2.models.temporal import load_checkpoint
from mot_pa2.tracking.trackers import TemporalTracker, records
from mot_pa2.tracking.association import IoUTracker
from mot_pa2.visualization.annotations import annotate


def infer_video(sequence_root, checkpoint="checkpoints/gru.pt",
                output="outputs/inference.mp4", source="auto", detector=None, fps=None):
    """Accept a MOT directory, generic image folder, or video file.

    Auto uses public detections when det/det.txt exists; otherwise torchvision.
    DPM/FRCNN/SDP is inferred from the directory suffix unless explicitly supplied.
    Generic image folders require fps. All frames must have the same dimensions.
    """
    import cv2
    root, output = Path(sequence_root), Path(output)
    if root.resolve() == output.resolve():
        raise ValueError("Output must not overwrite the input video")
    if source == "auto":
        source = "public" if (root / "det" / "det.txt").is_file() else "torchvision"
    if source not in ("public", "torchvision"):
        raise ValueError(f"Unknown detection source: {source}")
    if source == "public" and not (root / "seqinfo.ini").is_file():
        raise ValueError("Public detections require a MOT directory with seqinfo.ini")
    with open_frames(root, fps) as (info, images):
        if checkpoint:
            model, _ = load_checkpoint(checkpoint)
            tracker = TemporalTracker(model, (info["width"], info["height"]))
        else:
            tracker = IoUTracker()
        detections = load_public_detections(root, detector) if source == "public" else None
        if source == "torchvision":
            from mot_pa2.models.detectors import torchvision_person_detector
            detect = torchvision_person_detector()
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                                 info["fps"], (info["width"], info["height"]))
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"Cannot open video writer: {output}")
        identities, count = set(), 0
        try:
            for count, image in enumerate(images, 1):
                if image.shape[:2] != (info["height"], info["width"]):
                    raise ValueError(f"Inconsistent image dimensions at frame {count}")
                observed = detections[count] if detections is not None else detect(image, count)
                items = tracker.update(observed)[0] if checkpoint else records(tracker.update(observed))
                identities.update(item["track_id"] for item in items)
                writer.write(annotate(image, items)[:, :, ::-1])
        finally:
            writer.release()
    if not count:
        raise ValueError("Input contains no decodable frames")
    return dict(video=str(output), unique_objects=len(identities), frames=count)
