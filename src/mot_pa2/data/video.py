"""Streaming RGB input for MOT sequences, image directories, and video files."""
from contextlib import contextmanager
from pathlib import Path
import math
import re

import numpy as np
from PIL import Image

from mot_pa2.data.mot17 import sequence_info


def image_paths(directory):
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    def order(path):
        return [int(part) if part.isdigit() else part.lower()
                for part in re.split(r"(\d+)", path.name)]
    return sorted((p for p in Path(directory).iterdir()
                   if p.is_file() and p.suffix.lower() in extensions), key=order)


@contextmanager
def open_frames(path, fps=None):
    """Yield metadata and lazy frames; generic image folders require explicit FPS."""
    import cv2
    path = Path(path)
    if path.is_file():
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError(f"Cannot open video: {path}")
            rate = fps if fps is not None else capture.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(rate) or rate <= 0:
                raise ValueError("Video has no valid FPS; supply fps explicitly")
            info = dict(width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), fps=rate)
            def frames():
                while True:
                    ok, bgr = capture.read()
                    if not ok:
                        break
                    yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            yield info, frames()
        finally:
            capture.release()
        return
    if not path.is_dir():
        raise FileNotFoundError(path)
    directory = path / "img1" if (path / "img1").is_dir() else path
    paths = image_paths(directory)
    if not paths:
        raise ValueError(f"No supported images in {directory}")
    if (path / "seqinfo.ini").exists():
        info = sequence_info(path)
        if len(paths) != info["frames"]:
            raise ValueError(f"Expected {info['frames']} frames, found {len(paths)}")
    else:
        if fps is None:
            raise ValueError("Image folders without seqinfo.ini require fps")
        with Image.open(paths[0]) as first:
            width, height = first.size
        info = dict(width=width, height=height, frames=len(paths), fps=fps)
    if fps is not None:
        info["fps"] = fps
    if not math.isfinite(info["fps"]) or info["fps"] <= 0:
        raise ValueError("fps must be a finite positive number")
    def frames():
        for image_path in paths:
            with Image.open(image_path) as image:
                yield np.asarray(image.convert("RGB"))
    yield info, frames()
