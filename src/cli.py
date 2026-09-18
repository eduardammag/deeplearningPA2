"""Execucao dos experimentos com parametros definidos diretamente em Python."""
from pathlib import Path
import numpy as np
from .evaluation import evaluate_baseline, save_metrics
from .mot17 import load_ground_truth, load_public_detections
from .synthetic import SyntheticConfig, corrupt_detections, generate_video


def synthetic(frames=45, objects=8, speed=2.0, occlusion=8, drop=0.1,
              noise=1.0, false_positive=0.1, seed=0, output="outputs/synthetic"):
    config = dict(frames=frames, objects=objects, speed=speed, occlusion=occlusion,
                  drop=drop, noise=noise, false_positive=false_positive, seed=seed,
                  output=str(output), command="synthetic")
    video, truth = generate_video(SyntheticConfig(frames=frames, objects=objects,
                                                typical_speed=speed, occlusion_duration=occlusion, seed=seed))
    detections = corrupt_detections(truth, drop, noise, false_positive, seed=seed)
    by_frame = {index: frame for index, frame in enumerate(detections)}
    metrics, _, _ = evaluate_baseline(by_frame, {index: frame for index, frame in enumerate(truth)})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "video.npy", video)
    save_metrics({"metrics": metrics, "config": config}, output / "metrics.json")
    print(metrics)


def evaluate_mot(sequence, root="data/MOT17", detector="FRCNN", matcher="greedy",
                 output="outputs/mot_metrics.json"):
    root = Path(root) / "train" / sequence
    metrics, _, _ = evaluate_baseline(load_public_detections(root, detector), load_ground_truth(root), matcher=matcher)
    save_metrics(metrics, output)
    print(metrics)


def main():
    # Ajuste os parametros aqui para executar o experimento.
    synthetic(frames=45, objects=8, seed=0)

if __name__ == "__main__":
    main()
