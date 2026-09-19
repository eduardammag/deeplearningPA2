"""Sequence inference, no ground truth required, persistent state across the whole video."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from functools import lru_cache
from .mot17 import sequence_info,sequence_frames,load_public_detections
from .temporal import load_checkpoint
from .tracking import TemporalTracker
from .association import IoUTracker
from .tracking import records

def color(identity):
    return tuple(int(v) for v in np.random.default_rng(int(identity)).integers(60,256,3))

@lru_cache(maxsize=8)
def label_font(size):
    from matplotlib.font_manager import findfont
    return ImageFont.truetype(findfont("DejaVu Sans"),size)

def annotate(image,items):
    image=Image.fromarray(image.copy()) if isinstance(image,np.ndarray) else image.copy()
    draw=ImageDraw.Draw(image)
    font=label_font(max(14,image.height//36))
    for item in items:
        identity=item.get("track_id",item.get("gt_id",0))
        draw.rectangle(item["bbox"],outline=color(identity),width=max(2,image.width//400))
        x,y=item["bbox"][:2]
        position=(max(0,x),max(0,y))
        draw.text(position,str(identity),font=font,fill=color(identity),stroke_width=2,stroke_fill="black")
    return np.asarray(image)

def infer_video(sequence_root,checkpoint="checkpoints/gru.pt",output="outputs/inference.mp4",source="public",detector="FRCNN"):
    import cv2
    root=Path(sequence_root)
    info=sequence_info(root)
    paths=sequence_frames(root)
    if len(paths)!=info["frames"]: raise FileNotFoundError(f"Expected {info['frames']} images in {root/'img1'}, found {len(paths)}")
    if checkpoint:
        model,_=load_checkpoint(checkpoint)
        tracker=TemporalTracker(model,(info["width"],info["height"]))
    else: tracker=IoUTracker()
    detections=load_public_detections(root,detector) if source=="public" else None
    if source=="torchvision":
        from .detectors import torchvision_person_detector
        detect=torchvision_person_detector()
    elif source!="public": raise ValueError(source)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*"mp4v"),info["fps"],(info["width"],info["height"]))
    if not writer.isOpened(): raise RuntimeError(f"Cannot open video writer: {output}")
    identities=set()
    try:
        for f,path in enumerate(paths,1):
            image=np.asarray(Image.open(path).convert("RGB"))
            observed=detections[f] if detections is not None else detect(image,f)
            items=tracker.update(observed)[0] if checkpoint else records(tracker.update(observed))
            identities.update(x["track_id"] for x in items)
            writer.write(annotate(image,items)[:,:,::-1])
    finally: writer.release()
    return dict(video=str(output),unique_objects=len(identities),frames=len(paths))
