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
    order = sorted(range(len(boxes)), key=lambda index: scores[index], reverse=True)
    keep = []
    while order:
        current = order.pop(0)
        keep.append(current)
        order = [index for index in order if iou(boxes[current], boxes[index]) < threshold]
    return keep


def xywh_to_xyxy(box):
    x, y, width, height = box
    return (x, y, x + width, y + height)


def center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
