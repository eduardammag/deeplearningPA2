"""Regression checks for the PA2 audit fixes."""
import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from mot_pa2.data.mot17 import load_public_detections
from mot_pa2.evaluation.metrics import detection_metrics
from mot_pa2.experiments.correction import identity_trace


def mot_directory(tmp_path, detector):
    root = tmp_path / f"MOT17-99-{detector}"
    (root / "img1").mkdir(parents=True)
    (root / "det").mkdir()
    (root / "seqinfo.ini").write_text(
        "[Sequence]\nimWidth=32\nimHeight=32\nseqLength=2\nframeRate=10\n")
    (root / "det/det.txt").write_text("1,-1,2,2,10,10,0.9\n2,-1,2,2,10,10,0.9\n")
    for frame in (1, 2):
        Image.new("RGB", (32, 32)).save(root / "img1" / f"{frame:06d}.jpg")
    return root


@pytest.mark.parametrize("detector", ["DPM", "FRCNN", "SDP"])
def test_public_detector_is_inferred_and_explicit_mismatch_is_rejected(tmp_path, detector):
    root = mot_directory(tmp_path, detector)
    assert len(load_public_detections(root)[1]) == 1
    wrong = "DPM" if detector != "DPM" else "FRCNN"
    with pytest.raises(ValueError, match="does not match"):
        load_public_detections(root, detector=wrong)


def test_frame_mean_and_pooled_ap_are_not_interchangeable():
    # Perfect detection on a one-object frame; no detections on a two-object frame.
    truth = {1: [dict(bbox=(0, 0, 10, 10))],
             2: [dict(bbox=(0, 0, 10, 10)), dict(bbox=(20, 0, 30, 10))]}
    scores = detection_metrics({1: [dict(bbox=(0, 0, 10, 10), score=1)], 2: []}, truth)
    assert scores["mAP_per_frame"] == {1: 1., 2: 0.}
    assert scores["mAP_frame_mean"] == .5
    assert scores["mAP_sequence"] == pytest.approx(34 / 101)
    assert scores["mAP"] == scores["mAP_sequence"]
    assert detection_metrics({1: []}, {1: []})["mAP_frame_mean"] == 0


def test_correction_trace_distinguishes_survival_from_replacement():
    box = (0, 0, 10, 10)
    truth = {f: [dict(gt_id=1, bbox=box)] for f in (1, 2, 3)}
    predictions = {1: [dict(track_id=7, bbox=box, missed=0)], 2: [],
                   3: [dict(track_id=9, bbox=box, missed=0)]}
    trace = identity_trace(predictions, truth, 1, 1, 3)
    assert trace["original_track_id"] == 7
    assert not trace["trace"][1]["original_track_alive"]
    assert trace["trace"][2]["matched_track_id"] == 9
    assert not trace["trace"][2]["original_track_alive"]


def test_evaluation_does_not_train_or_replace_checkpoint(tmp_path, monkeypatch):
    from mot_pa2.experiments import analysis, correction
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"existing weights")
    monkeypatch.setattr(analysis, "OUT", tmp_path)
    monkeypatch.setattr(analysis, "load_checkpoint", lambda path: (object(), dict(cell="gru", window=16, seed=0)))
    def forbidden(*args, **kwargs):
        pytest.fail("Evaluation must not train")
    monkeypatch.setattr(analysis, "ablation_cells", forbidden)
    monkeypatch.setattr(analysis, "train_model", forbidden)
    for name in ("synthetic_suite", "baseline_suite", "stress_detector", "memory_suite", "correction_suite"):
        monkeypatch.setattr(analysis, name, lambda *a, **kw: None)
    monkeypatch.setattr(analysis, "resolve_sequence", lambda *a: tmp_path)
    monkeypatch.setattr(analysis, "failure_gallery", lambda *a, **kw: [{}])
    monkeypatch.setattr(correction, "correction_case", lambda *a: None)
    analysis.main(checkpoint)
    assert checkpoint.read_bytes() == b"existing weights"
    assert json.loads((tmp_path / "evaluation.json").read_text())["sha256"] == hashlib.sha256(b"existing weights").hexdigest()


def test_manifest_covers_nested_code_and_artifacts_without_self_hash(tmp_path):
    from mot_pa2.evaluation.provenance import write_manifest
    (tmp_path / "src/mot_pa2/nested").mkdir(parents=True)
    (tmp_path / "src/mot_pa2/nested/module.py").write_text("x = 1\n")
    (tmp_path / "results").mkdir()
    (tmp_path / "results/correction_case.json").write_text("{}")
    (tmp_path / "AI_LOG.md").write_text("audit")
    write_manifest(["correction_case.json"], tmp_path)
    manifest = json.loads((tmp_path / "results/manifest.json").read_text())
    assert "src/mot_pa2/nested/module.py" in manifest["hashes"]
    assert "results/correction_case.json" in manifest["hashes"]
    assert "AI_LOG.md" in manifest["hashes"]
    assert "results/manifest.json" not in manifest["hashes"]
    assert all((tmp_path / path).exists() for path in manifest["hashes"])


def test_inference_auto_public_dpm_without_ground_truth(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from mot_pa2.pipelines.inference import infer_video
    root = mot_directory(tmp_path, "DPM")
    result = infer_video(root, checkpoint=None, output=tmp_path / "out.mp4")
    assert result["frames"] == 2
    assert result["unique_objects"] == 1
    capture = cv2.VideoCapture(result["video"])
    try:
        assert capture.isOpened()
        assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 2
    finally:
        capture.release()


def test_generic_images_and_video_use_streaming_detector(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    from mot_pa2.pipelines.inference import infer_video
    from mot_pa2.models import detectors
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (32, 32), (30, 0, 0)).save(images / "10.png")
    Image.new("RGB", (32, 32), (10, 0, 0)).save(images / "2.png")
    seen = []
    monkeypatch.setattr(detectors, "torchvision_person_detector",
                        lambda: lambda image, frame: seen.append((frame, int(image[0, 0, 0]))) or [])
    with pytest.raises(ValueError, match="require fps"):
        infer_video(images, checkpoint=None, output=tmp_path / "out.mp4")
    output = tmp_path / "out.mp4"
    result = infer_video(images, checkpoint=None, output=output, fps=10)
    assert result["frames"] == 2
    assert seen == [(1, 10), (2, 30)]
    seen.clear()
    result = infer_video(output, checkpoint=None, output=tmp_path / "video-out.mp4")
    assert result["frames"] == 2
    assert [f for f, _ in seen] == [1, 2]
    original = output.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        infer_video(output, checkpoint=None, output=output)
    assert output.read_bytes() == original
