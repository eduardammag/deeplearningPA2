"""Fingerprints of the current source, inputs, checkpoints, and published artifacts."""
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import hashlib
import platform

from mot_pa2.evaluation.serialization import save_metrics


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(required, root=Path(".")):
    """Describe the current snapshot, not historical provenance of every training run."""
    root = Path(root)
    paths = set()
    for pattern in ("src/**/*.py", "tests/**/*.py", "checkpoints/**/*.pt",
                    "results/**/*.json", "results/**/*.png", "results/*.md",
                    "data_MOT17Labels/train/*-FRCNN/gt/gt.txt",
                    "data_MOT17Labels/train/*-FRCNN/det/det.txt",
                    "data_MOT17Labels/train/*-FRCNN/seqinfo.ini",
                    "data_MOT17Labels/train/MOT17-09-FRCNN/img1/*.jpg",
                    ".cache/torch/checkpoints/fasterrcnn_mobilenet_v3_large_320_fpn-*.pth"):
        paths.update(root.glob(pattern))
    paths.update(root / name for name in (
        "metrics.py", "train.py", "pyproject.toml", "README.md", "AI_LOG.md",
        "PA2_AUDIT.md", "inferencia.ipynb", "PA2.pdf", "outputs/inference.mp4",
        "outputs/torchvision_mobilenet_detections.json") if (root / name).exists())
    destination = root / "results/manifest.json"
    paths.discard(destination)
    dependencies = {}
    for name in ("torch", "torchvision", "numpy", "scipy", "matplotlib", "pillow", "opencv-python"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    hashes = {p.relative_to(root).as_posix(): sha256(p) for p in sorted(paths) if p.is_file()}
    save_metrics(dict(generated_at=datetime.now(timezone.utc).isoformat(),
                      scope="Current source/input/artifact snapshot; existing ablation weights were not retrained by evaluation",
                      python=platform.python_version(), dependencies=dependencies, device="cpu",
                      hashes=hashes, required_artifacts=required), destination)
