"""JSON serialization of experiment results."""
import json
from pathlib import Path


def save_metrics(metrics,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(metrics,indent=2),encoding="utf-8")
