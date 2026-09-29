"""Compatibility exports; implementations live in mot_pa2.evaluation.metrics."""
from mot_pa2.evaluation.metrics import (
    detection_map,
    detection_metrics,
    evaluate_tracking,
    fragmentations,
    frame_map,
    id_switches,
    identity_events,
    idf1,
    overlap,
    spatial_matches,
    unique_count_error,
)

__all__ = [
    "detection_map", "detection_metrics", "evaluate_tracking", "fragmentations", "frame_map",
    "id_switches", "identity_events", "idf1", "overlap", "spatial_matches",
    "unique_count_error",
]
