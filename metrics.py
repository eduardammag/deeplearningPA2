"""Métricas de tracking implementadas para o PA2.

As funções aceitam frames como iteráveis de objetos com `track_id`, `gt_id`
e uma caixa `xyxy`, mantendo a API pequena e independente do restante do repo.
"""
from collections import defaultdict
from typing import Iterable, Mapping, Sequence


def _frames(records: Iterable[Mapping]) -> list[list[Mapping]]:
    if isinstance(records, Mapping):
        return [list(records[key]) for key in sorted(records)]
    return [list(frame) for frame in records]


def _counts(predictions, ground_truth):
    pred_frames, gt_frames = _frames(predictions), _frames(ground_truth)
    pred_ids = sorted({item["track_id"] for frame in pred_frames for item in frame})
    gt_ids = sorted({item["gt_id"] for frame in gt_frames for item in frame})
    matrix = {(pred_id, gt_id): 0 for pred_id in pred_ids for gt_id in gt_ids}
    gt_total = sum(len(frame) for frame in gt_frames)
    pred_total = sum(len(frame) for frame in pred_frames)
    for pred_frame, gt_frame in zip(pred_frames, gt_frames):
        for pred in pred_frame:
            for gt in gt_frame:
                if pred.get("gt_id") == gt.get("gt_id"):
                    matrix[(pred["track_id"], gt["gt_id"])] += 1
    matches = _maximum_weight_matching(pred_ids, gt_ids, matrix)
    idtp = sum(matrix[pair] for pair in matches)
    return idtp, pred_total, gt_total, pred_frames, gt_frames


def _maximum_weight_matching(rows: Sequence, cols: Sequence, weights: dict) -> list[tuple]:
    """Maximum one-to-one assignment; dynamic programming is exact for MOT IDs."""
    if not rows or not cols:
        return []
    if len(cols) > 20:
        # A deterministic greedy fallback avoids exponential memory on large sets.
        available = set(cols)
        result = []
        for row in rows:
            candidates = sorted(((weights[(row, col)], col) for col in available), reverse=True)
            if candidates and candidates[0][0] > 0:
                result.append((row, candidates[0][1]))
                available.remove(candidates[0][1])
        return result
    best_value, best_pairs = 0, []
    def visit(index, available, value, pairs):
        nonlocal best_value, best_pairs
        if index == len(rows):
            if value > best_value:
                best_value, best_pairs = value, pairs.copy()
            return
        visit(index + 1, available, value, pairs)
        row = rows[index]
        for col in available:
            weight = weights[(row, col)]
            if weight:
                visit(index + 1, available - {col}, value + weight, pairs + [(row, col)])
    visit(0, set(cols), 0, [])
    return best_pairs


def idf1(predictions: Iterable, ground_truth: Iterable) -> float:
    idtp, pred_total, gt_total, _, _ = _counts(predictions, ground_truth)
    denominator = pred_total + gt_total
    return 2.0 * idtp / denominator if denominator else 1.0


def id_switches(predictions: Iterable, ground_truth: Iterable) -> int:
    previous = {}
    switches = 0
    for pred_frame, gt_frame in zip(_frames(predictions), _frames(ground_truth)):
        current = {item["gt_id"]: item["track_id"] for item in pred_frame if item.get("gt_id") is not None}
        for gt_id, track_id in current.items():
            if gt_id in previous and previous[gt_id] != track_id:
                switches += 1
        previous.update(current)
    return switches


def fragmentations(predictions: Iterable, ground_truth: Iterable) -> int:
    previous_visible = {}
    fragments = 0
    for pred_frame, gt_frame in zip(_frames(predictions), _frames(ground_truth)):
        visible = {item["gt_id"] for item in pred_frame if item.get("gt_id") is not None}
        gt_ids = {item["gt_id"] for item in gt_frame}
        for gt_id in gt_ids:
            if gt_id in previous_visible and not previous_visible[gt_id] and gt_id in visible:
                fragments += 1
        for gt_id in gt_ids:
            previous_visible[gt_id] = gt_id in visible
    return fragments


def unique_count_error(predictions: Iterable, ground_truth: Iterable) -> int:
    pred_ids = {item["track_id"] for frame in _frames(predictions) for item in frame}
    gt_ids = {item["gt_id"] for frame in _frames(ground_truth) for item in frame}
    return len(pred_ids) - len(gt_ids)


def evaluate_tracking(predictions, ground_truth) -> dict[str, float | int]:
    return {"IDF1": idf1(predictions, ground_truth), "IDSW": id_switches(predictions, ground_truth),
            "fragmentations": fragmentations(predictions, ground_truth),
            "unique_count_error": unique_count_error(predictions, ground_truth)}
