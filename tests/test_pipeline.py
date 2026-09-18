from mot_pa2.association import IoUTracker
from mot_pa2.geometry import iou, nms
from mot_pa2.synthetic import SyntheticConfig, generate_video, corrupt_detections


def test_synthetic_has_real_occlusion():
    video, truth = generate_video(SyntheticConfig(frames=30, objects=3, seed=7, occlusion_duration=5))
    assert video.shape == (30, 128, 128, 3)
    assert any(len(frame) < 3 for frame in truth)


def test_detector_corruption_and_tracker():
    _, truth = generate_video(SyntheticConfig(frames=12, objects=2, seed=2, occlusion_duration=0))
    detections = corrupt_detections(truth, drop_probability=0.0, noise_std=0.0, false_positive_rate=0.0, seed=2)
    tracker = IoUTracker(iou_threshold=0.1, max_missed=2)
    tracks = [tracker.update(frame) for frame in detections]
    assert len({track.track_id for frame in tracks for track in frame}) == 2
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert nms([(0, 0, 10, 10), (1, 1, 11, 11)], [0.9, 0.8], 0.5) == [0]
