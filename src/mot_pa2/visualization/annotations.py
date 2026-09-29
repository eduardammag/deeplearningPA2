"""Deterministic identity colors and frame annotations."""
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont


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

