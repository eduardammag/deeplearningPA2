from metrics import evaluate_tracking, idf1, id_switches


def record(track_id, gt_id):
    return {"track_id": track_id, "gt_id": gt_id}


def test_perfect_identity_has_idf1_one():
    gt = [[record(1, 1), record(2, 2)] for _ in range(3)]
    assert idf1(gt, gt) == 1.0
    assert id_switches(gt, gt) == 0


def test_swap_counts_one_switch_per_identity():
    gt = [[record(1, 1), record(2, 2)] for _ in range(3)]
    pred = [gt[0], [record(2, 1), record(1, 2)], [record(2, 1), record(1, 2)]]
    assert id_switches(pred, gt) == 2
    assert evaluate_tracking(pred, gt)["IDF1"] == 2 / 3


def test_fragmentation_changes_idf1_differently_from_swap():
    gt = [[record(1, 1)] for _ in range(4)]
    pred = [[record(9, 1)], [], [record(10, 1)], [record(10, 1)]]
    result = evaluate_tracking(pred, gt)
    assert result["fragmentations"] == 1
    assert result["IDF1"] < 1.0
