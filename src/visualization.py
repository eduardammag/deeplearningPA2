"""Visualizacao de IDs e galerias de falhas."""
from pathlib import Path
import matplotlib.pyplot as plt
from .types import Detection


def draw_frame(image, objects, output=None, title=""):
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.imshow(image)
    for item in objects:
        box = item["bbox"] if isinstance(item, dict) else item.bbox
        identity = item.get("track_id", item.get("gt_id", 0)) if isinstance(item, dict) else item.gt_id
        x1, y1, x2, y2 = box
        color = plt.cm.tab20(int(identity or 0) % 20)
        axis.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=color, linewidth=2))
        axis.text(x1, y1, str(identity), color="white", backgroundcolor=color)
    axis.set_title(title)
    axis.axis("off")
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, bbox_inches="tight")
        plt.close(figure)
    return figure
