"""Agregacao de resultados para tabelas da apresentacao."""
import json
from pathlib import Path
import numpy as np


def mean_std(records, key):
    values = np.asarray([record[key] for record in records], dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}


def save_ablation(records, path):
    grouped = {}
    for record in records:
        grouped.setdefault(record["cell"], []).append(record)
    table = {cell: {metric: mean_std(values, metric) for metric in ("IDF1", "IDSW") if metric in values[0]}
             for cell, values in grouped.items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"runs": records, "summary": table}, indent=2), encoding="utf-8")
    return table
