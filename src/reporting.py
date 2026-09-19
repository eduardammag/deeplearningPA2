"""Aggregate independent seeds by recurrent cell AND BPTT window."""
from pathlib import Path
import numpy as np
from .evaluation import save_metrics

def mean_std(records,key):
    values=np.asarray([r[key] for r in records],dtype=float)
    return dict(mean=float(values.mean()),std=float(values.std(ddof=1)) if len(values)>1 else 0.)

def save_ablation(records,path):
    grouped={}
    for r in records: grouped.setdefault(f"{r['cell']}_T{r['window']}",[]).append(r)
    table={name:{key:mean_std(runs,key) for key in ("IDF1","IDSW","fragmentations","unique_count_error")} for name,runs in grouped.items()}
    save_metrics(dict(runs=records,summary=table),path)
    return table
