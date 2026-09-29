"""Geometria de caixas e NMS, sem depender de torchvision.ops."""
import numpy as np


def iou(first, second) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def nms(boxes, scores, threshold=0.5):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    order = np.argsort(-np.asarray(scores), kind="stable")
    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    keep = []
    while len(order):
        current = int(order[0])
        keep.append(current)
        rest = order[1:]
        sizes = np.maximum(0, np.minimum(boxes[current, 2:], boxes[rest, 2:]) -
                           np.maximum(boxes[current, :2], boxes[rest, :2]))
        intersection = sizes[:, 0] * sizes[:, 1]
        union = areas[current] + areas[rest] - intersection
        overlap = np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)
        order = rest[overlap < threshold]
    return keep


def xywh_to_xyxy(box):
    x, y, width, height = box
    return (x, y, x + width, y + height)


def center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
