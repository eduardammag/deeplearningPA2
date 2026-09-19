"""Spatial tracking metrics, global identity assignment and explicit events."""
from collections.abc import Mapping
import numpy as np
from scipy.optimize import linear_sum_assignment

def frame_map(records):
    return dict(records) if isinstance(records, Mapping) else dict(enumerate(records))

def overlap(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-inter
    return inter/union if union else 0.

def spatial_matches(pred, truth, threshold=.5):
    if not pred or not truth: return []
    scores=np.array([[overlap(p["bbox"],g["bbox"]) for g in truth] for p in pred])
    weights=np.where(scores>=threshold,min(len(pred),len(truth))+1+scores,0)
    rows,cols=linear_sum_assignment(weights,maximize=True)
    return [(int(p),int(g)) for p,g in zip(rows,cols) if scores[p,g]>=threshold]

def identity_events(predictions, ground_truth, threshold=.5):
    pred,truth=frame_map(predictions),frame_map(ground_truth)
    last_id,ever,interrupted={},set(),set()
    events=[]
    for frame in sorted(set(pred)|set(truth)):
        ps,gs=pred.get(frame,[]),truth.get(frame,[])
        matched={g:p for p,g in spatial_matches(ps,gs,threshold)}
        for gi,gt in enumerate(gs):
            identity=gt["gt_id"]
            if gi not in matched:
                if identity in ever: interrupted.add(identity)
                continue
            track=ps[matched[gi]]["track_id"]
            events.append(dict(frame=frame,gt_id=identity,track_id=track,
                switch=identity in last_id and last_id[identity]!=track,fragment=identity in interrupted))
            last_id[identity]=track
            ever.add(identity)
            interrupted.discard(identity)
    return events

def evaluate_tracking(predictions,ground_truth,threshold=.5):
    pred,truth=frame_map(predictions),frame_map(ground_truth)
    pids=sorted({p["track_id"] for ps in pred.values() for p in ps})
    gids=sorted({g["gt_id"] for gs in truth.values() for g in gs})
    pi,gi={x:i for i,x in enumerate(pids)},{x:i for i,x in enumerate(gids)}
    weights=np.zeros((len(pids),len(gids)),dtype=np.int64)
    for frame in sorted(set(pred)|set(truth)):
        ps,gs=pred.get(frame,[]),truth.get(frame,[])
        if len({p["track_id"] for p in ps})!=len(ps): raise ValueError("Duplicate predicted ID")
        if len({g["gt_id"] for g in gs})!=len(gs): raise ValueError("Duplicate true ID")
        for p in ps:
            for g in gs:
                if overlap(p["bbox"],g["bbox"])>=threshold:
                    weights[pi[p["track_id"]],gi[g["gt_id"]]]+=1
    rows,cols=linear_sum_assignment(weights,maximize=True)
    tp=int(weights[rows,cols].sum())
    npred,ngt=sum(map(len,pred.values())),sum(map(len,truth.values()))
    events=identity_events(pred,truth,threshold)
    return dict(IDF1=2*tp/(npred+ngt) if npred+ngt else 1.,IDTP=tp,IDFP=npred-tp,IDFN=ngt-tp,
        IDSW=sum(e["switch"] for e in events),fragmentations=sum(e["fragment"] for e in events),
        unique_count_error=len(pids)-len(gids),predicted_identities=len(pids),true_identities=len(gids))

def idf1(predictions,ground_truth):
    return evaluate_tracking(predictions,ground_truth)["IDF1"]

def id_switches(predictions,ground_truth):
    return sum(e["switch"] for e in identity_events(predictions,ground_truth))

def fragmentations(predictions,ground_truth):
    return sum(e["fragment"] for e in identity_events(predictions,ground_truth))

def unique_count_error(predictions,ground_truth):
    return evaluate_tracking(predictions,ground_truth)["unique_count_error"]

def detection_map(detections,ground_truth,thresholds=np.arange(.5,.96,.05)):
    """Single-class AP pooled across frames, 101 recall points and IoU .50:.95."""
    det,truth=frame_map(detections),frame_map(ground_truth)
    total=sum(map(len,truth.values()))
    ranked=sorted(((float(p.get("score",1)),f,p) for f,ps in det.items() for p in ps),
                  key=lambda x:x[0],reverse=True)
    aps=[]
    for threshold in thresholds:
        used,hits=set(),[]
        for _,f,p in ranked:
            candidates=[(overlap(p["bbox"],g["bbox"]),j) for j,g in enumerate(truth.get(f,[])) if (f,j) not in used]
            score,j=max(candidates,default=(0.,-1))
            hit=score>=threshold and j>=0
            hits.append(int(hit))
            if hit: used.add((f,j))
        tp=np.cumsum(hits)
        recall=tp/max(total,1)
        precision=tp/np.arange(1,len(tp)+1)
        aps.append(float(np.mean([precision[recall>=r].max(initial=0) for r in np.linspace(0,1,101)])) if total else 0.)
    return float(np.mean(aps))
