"""Gerador controlado de videos 128x128 com oclusao por profundidade."""
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageDraw
from .types import Detection

@dataclass(frozen=True)
class SyntheticConfig:
    frames: int = 45
    objects: int = 8
    typical_speed: float = 2.0
    occlusion_duration: int = 8
    width: int = 128
    height: int = 128
    seed: int = 0


def generate_video(config: SyntheticConfig):
    rng = np.random.default_rng(config.seed)
    centers = rng.uniform([12, 12], [config.width - 12, config.height - 12], (config.objects, 2))
    velocity = rng.normal(0, config.typical_speed, (config.objects, 2))
    velocity[:, 0] += np.where(np.arange(config.objects) % 2, 0.8, -0.8)
    radii = rng.uniform(5, 12, (config.objects, 2))
    depth = rng.permutation(config.objects)
    colors = rng.integers(40, 230, (config.objects, 3), dtype=np.uint8)
    video, truth = [], []
    for frame_index in range(config.frames):
        image = Image.new("RGB", (config.width, config.height), (18, 18, 24))
        visible = []
        positions = centers + velocity * frame_index
        positions %= np.array([config.width, config.height])
        if config.objects >= 2 and config.occlusion_duration:
            start = max(0, config.frames // 2 - config.occlusion_duration // 2)
            if start <= frame_index < start + config.occlusion_duration:
                positions[1] = positions[0]
        # Objects with larger depth are drawn later and hide pixels of lower layers.
        for object_index in sorted(range(config.objects), key=lambda item: depth[item]):
            x, y = positions[object_index]
            rx, ry = radii[object_index]
            draw = ImageDraw.Draw(image)
            draw.ellipse((x - rx, y - ry, x + rx, y + ry), fill=tuple(colors[object_index]))
        pixels = np.asarray(image)
        for object_index in range(config.objects):
            x, y = positions[object_index]
            rx, ry = radii[object_index]
            box = (float(x - rx), float(y - ry), float(x + rx), float(y + ry))
            # A center pixel with the object's color is a robust visible/occluded test.
            center_color = pixels[int(y) % config.height, int(x) % config.width]
            is_visible = np.linalg.norm(center_color.astype(float) - colors[object_index]) < 2
            if is_visible:
                visible.append(Detection(frame_index, box, 1.0, object_index + 1))
        video.append(pixels)
        truth.append(visible)
    return np.asarray(video, dtype=np.uint8), truth


def corrupt_detections(truth, drop_probability=0.1, noise_std=1.0, false_positive_rate=0.1,
                       width=128, height=128, seed=0):
    """Aplica FN, ruído de caixa e falsos positivos de forma reproduzível."""
    rng = np.random.default_rng(seed)
    result = []
    for frame_index, frame in enumerate(truth):
        detections = []
        for item in frame:
            if rng.random() < drop_probability:
                continue
            noisy = np.asarray(item.bbox) + rng.normal(0, noise_std, 4)
            x1, y1, x2, y2 = noisy
            clipped = (float(np.clip(x1, 0, width - 1)), float(np.clip(y1, 0, height - 1)),
                       float(np.clip(x2, 1, width)), float(np.clip(y2, 1, height)))
            detections.append(Detection(frame_index, clipped, item.score, item.gt_id))
        false_count = rng.poisson(false_positive_rate * max(1, len(frame)))
        for _ in range(false_count):
            x, y = rng.uniform(0, width - 16), rng.uniform(0, height - 16)
            detections.append(Detection(frame_index, (x, y, x + rng.uniform(5, 18), y + rng.uniform(5, 18)), 0.2))
        result.append(detections)
    return result
