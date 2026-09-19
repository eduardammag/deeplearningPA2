"""Reproducible PA2 experiments. Constants are edited here, without CLI parsers."""
from pathlib import Path
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from metrics import evaluate_tracking,detection_map,identity_events,spatial_matches
from .evaluation import truth_records,detection_records,evaluate_baseline,evaluate_model,save_metrics
from .mot17 import load_ground_truth,load_public_detections,resolve_sequence,sequence_info,occlusion_durations
from .training import DATA_ROOT,TRAIN,VALIDATION,TEST,trajectory_chunks,train_model
from .temporal import load_checkpoint,gradient_horizon,box_to_features
from .synthetic import SyntheticConfig,generate_video,corrupt_detections
from .tracking import run_temporal
from .inference import annotate

OUT=Path("outputs")
EPOCHS=3

def finish(fig,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.tight_layout()
    fig.savefig(path,dpi=150)
    plt.close(fig)

def synthetic_suite():
    rows=[]
    for objects,speed,occlusion in [(5,.15,0),(8,1.,4),(12,2.,8),(15,4.,16)]:
        video,gt=generate_video(SyntheticConfig(objects=objects,typical_speed=speed,occlusion_duration=occlusion))
        det=corrupt_detections(gt,0,0,0)
        metrics,_,_=evaluate_baseline(dict(enumerate(det)),dict(enumerate(gt)))
        rows.append(dict(objects=objects,speed=speed,occlusion=occlusion,**metrics))
    if rows[0]["IDF1"]<.98: raise AssertionError("Easy baseline must be near perfect")
    save_metrics(rows,OUT/"synthetic/results.json")
    fig,ax=plt.subplots()
    ax.plot(range(len(rows)),[r["IDF1"] for r in rows],"o-")
    ax.set(xticks=range(len(rows)),xticklabels=[f"{r['objects']} / {r['speed']} / {r['occlusion']}" for r in rows],
           xlabel="Objects / speed / occlusion frames",ylabel="IDF1",ylim=(0,1.05))
    finish(fig,OUT/"synthetic/degradation.png")
    video,gt=generate_video(SyntheticConfig(objects=5,typical_speed=.15,occlusion_duration=8,image_noise=0))
    absent=[f for f,items in enumerate(gt) if 1 not in {x.gt_id for x in items}]
    save_metrics(dict(hidden_identity=1,absent_frames=absent),OUT/"synthetic/occlusion.json")
    frames=[min(absent)-1,min(absent),max(absent),max(absent)+1]
    fig,axes=plt.subplots(1,4,figsize=(12,3))
    for ax,f in zip(axes,frames):
        ax.imshow(video[f])
        for item in gt[f]:
            ax.text(item.bbox[0],item.bbox[1]-2,str(item.gt_id),color="white",fontsize=10)
        visible=1 in {x.gt_id for x in gt[f]}
        ax.set_title(f"frame {f}: ID 1 {'visible' if visible else 'fully hidden'}"); ax.axis("off")
    finish(fig,OUT/"synthetic/occlusion.png")

def baseline_suite(model):
    rows=[]
    for name in TEST:
        root=resolve_sequence(DATA_ROOT,name)
        info=sequence_info(root)
        gt,det=load_ground_truth(root),load_public_detections(root)
        baseline,_,truth=evaluate_baseline(det,gt)
        temporal,_=evaluate_model(det,gt,model,(info["width"],info["height"]))
        rows.append(dict(sequence=name,density=sum(map(len,gt.values()))/len(gt),
                         mAP=detection_map(detection_records(det),truth),baseline=baseline,temporal=temporal))
        print("Evaluated",name,flush=True)
        save_metrics(rows,OUT/"comparison.json")
    rows.sort(key=lambda x:x["density"])
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
    x=np.arange(len(rows))
    axes[0].plot(x,[r["mAP"] for r in rows],"o-",label="detector mAP .50:.95")
    for method in ("baseline","temporal"):
        axes[0].plot(x,[r[method]["IDF1"] for r in rows],"o-",label=method+" IDF1")
    axes[0].legend(); axes[0].set_ylabel("Score")
    axes[1].plot(x,[r["baseline"]["predicted_identities"]/r["baseline"]["true_identities"] for r in rows],"o-",label="pred IDs / GT IDs")
    axes[1].plot(x,[r["baseline"]["IDSW"]/r["baseline"]["true_identities"] for r in rows],"o-",label="switches / GT IDs")
    axes[1].set(xticks=x,xticklabels=[r["sequence"] for r in rows],xlabel="Sequences ordered by increasing density")
    axes[1].legend()
    finish(fig,OUT/"decoupling.png")
    return rows

def ablation_cells():
    train=trajectory_chunks([resolve_sequence(DATA_ROOT,s) for s in TRAIN])
    validation=trajectory_chunks([resolve_sequence(DATA_ROOT,s) for s in VALIDATION])
    root=resolve_sequence(DATA_ROOT,VALIDATION[0])
    gt,det=load_ground_truth(root),load_public_detections(root)
    info=sequence_info(root)
    results=[]
    for cell in ("rnn","lstm","gru"):
        for window in (4,8,16,32):
            for seed in (0,1,2):
                checkpoint=Path(f"checkpoints/ablation/{cell}_T{window}_s{seed}.pt")
                result_path=OUT/f"ablation/{cell}_T{window}_s{seed}.json"
                if result_path.exists():
                    results.append(json.loads(result_path.read_text()))
                    continue
                model,history=train_model(train,cell=cell,epochs=EPOCHS,window=window,seed=seed,
                    validation=validation,checkpoint=checkpoint,
                    metadata=dict(train=list(TRAIN),validation=list(VALIDATION),test=list(TEST)))
                metrics,_=evaluate_model(det,gt,model,(info["width"],info["height"]))
                result=dict(cell=cell,window=window,seed=seed,parameters=sum(p.numel() for p in model.parameters()),
                            history=history,**metrics)
                save_metrics(result,result_path)
                results.append(result)
                print("Ablation",cell,window,seed,metrics["IDF1"],flush=True)
    from .reporting import save_ablation
    save_ablation(results,OUT/"ablation.json")
    fig,ax=plt.subplots()
    for cell in ("rnn","lstm","gru"):
        groups=[[r["IDF1"] for r in results if r["cell"]==cell and r["window"]==w] for w in (4,8,16,32)]
        ax.errorbar([4,8,16,32],[np.mean(g) for g in groups],yerr=[np.std(g,ddof=1) for g in groups],marker="o",label=cell)
    ax.set(xlabel="Truncated BPTT window",ylabel="Validation IDF1: mean +/- sample SD")
    ax.legend()
    finish(fig,OUT/"ablation.png")
    return results

def stress_detector(model,root):
    info=sequence_info(root)
    gt,source=load_ground_truth(root),load_public_detections(root)
    results=[]
    for level,(drop,noise,fp) in enumerate(((.1,1.,.1),(.25,3.,.25),(.5,6.,.5)),1):
        corrupted=corrupt_detections(list(source.values()),drop,noise,fp,info["width"],info["height"],seed=0)
        det=dict(zip(source,corrupted))
        temporal,_=evaluate_model(det,gt,model,(info["width"],info["height"]))
        baseline,_,truth=evaluate_baseline(det,gt)
        results.append(dict(level=level,drop=drop,noise=noise,false_positive=fp,
            mAP=detection_map(detection_records(det),truth),temporal=temporal,baseline=baseline))
    save_metrics(results,OUT/"stress.json")
    fig,ax=plt.subplots()
    ax.plot([r["level"] for r in results],[r["mAP"] for r in results],"o-",label="mAP")
    for method in ("baseline","temporal"):
        ax.plot([r["level"] for r in results],[r[method]["IDF1"] for r in results],"o-",label=method+" IDF1")
    ax.set(xlabel="Detector corruption intensity",ylabel="Score"); ax.legend()
    finish(fig,OUT/"stress.png")

def memory_suite(model,root):
    data=trajectory_chunks([root],length=33,stride=100)
    curves={}
    for cell in ("rnn","lstm","gru"):
        candidate=Path(f"checkpoints/ablation/{cell}_T16_s0.pt")
        if candidate.exists():
            m,_=load_checkpoint(candidate)
            curves[cell]=np.mean([gradient_horizon(m,x[None]) for x in data[:16]],axis=0).tolist()
    curves["final"]=np.mean([gradient_horizon(model,x[None]) for x in data[:16]],axis=0).tolist()
    fig,ax=plt.subplots()
    for name,values in curves.items(): ax.semilogy(range(len(values)),np.maximum(values,1e-16),label=name)
    ax.set(xlabel="Lag k",ylabel="Norm dL(t)/dh(t-k)"); ax.legend()
    finish(fig,OUT/"memory/gradient.png")
    info=sequence_info(root)
    # Isolated real trajectories remove association distractors, measuring state survival.
    trials=[]
    for index,trajectory in enumerate(data[:24]):
        for gap in (1,2,3,4,8,16,24):
            from .temporal import features_to_box
            from .types import Detection
            det={f:[] if 4<=f<4+gap else [Detection(f,features_to_box(x.tolist(),(info["width"],info["height"])))]
                 for f,x in enumerate(trajectory)}
            pred=run_temporal(det,model,image_size=(info["width"],info["height"]))
            before=pred[3][0]["track_id"]
            end=4+gap
            target=dict(gt_id=1,bbox=features_to_box(trajectory[end].tolist(),(info["width"],info["height"])))
            matches=spatial_matches(pred[end],[target])
            survived=any(pred[end][p]["track_id"]==before for p,g in matches)
            trials.append(dict(trajectory=index,gap=gap,survived=survived))
    durations=occlusion_durations(root)
    horizons=[max([t["gap"] for t in trials if t["trajectory"]==i and t["survived"]],default=0) for i in range(min(24,len(data)))]
    save_metrics(dict(gradient=curves,trials=trials,empirical_horizons=horizons,dataset_occlusions=durations,
                      visibility_threshold=.2,max_missed=3),OUT/"memory/results.json")
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    gaps=sorted({t["gap"] for t in trials})
    axes[0].plot(gaps,[np.mean([t["survived"] for t in trials if t["gap"]==g]) for g in gaps],"o-")
    axes[0].set(xlabel="Injected occlusion (frames)",ylabel="Same-ID survival")
    axes[1].hist([horizons,durations],bins=[0,1,2,4,8,16,32,64,128,512],label=["empirical state horizon","dataset occlusions"])
    axes[1].set(xlabel="Frames",ylabel="Count"); axes[1].legend()
    finish(fig,OUT/"memory/empirical.png")

def correction_suite(model):
    # A predeclared lifespan correction; diagnose and test on validation, never tune on test.
    root=resolve_sequence(DATA_ROOT,VALIDATION[0])
    info=sequence_info(root)
    det,gt=load_public_detections(root),load_ground_truth(root)
    results=[]
    for lifespan in (3,16):
        metrics,_=evaluate_model(det,gt,model,(info["width"],info["height"]),max_missed=lifespan)
        results.append(dict(max_missed=lifespan,**metrics))
    save_metrics(dict(hypothesis="Long occlusion outlasts the 3-frame track lifetime; extend to 16.",
        sequence=VALIDATION[0],results=results,
        conclusion="Increasing lifetime changes IDF1 by "+str(results[1]["IDF1"]-results[0]["IDF1"])+
        "; a nonpositive change means longer survival alone does not solve association/prediction errors."),
        OUT/"correction.json")
    fig,axes=plt.subplots(1,2,figsize=(8,3))
    for ax,key in zip(axes,("IDF1","IDSW")):
        ax.bar(["before: 3","after: 16"],[r[key] for r in results])
        ax.set_ylabel(key)
    finish(fig,OUT/"correction.png")

def failure_gallery(model,root):
    from PIL import Image
    from .mot17 import sequence_frames
    info=sequence_info(root)
    gt,det=load_ground_truth(root),load_public_detections(root)
    pred,debug=run_temporal(det,model,image_size=(info["width"],info["height"]),return_debug=True)
    all_events=identity_events(pred,truth_records(gt))
    last_match={}
    events=[]
    for event in all_events:
        if event["switch"]:
            event=dict(event,previous_match_frame=last_match[event["gt_id"]])
            events.append(event)
        last_match[event["gt_id"]]=event["frame"]
    # Prefer errors whose 4..16-frame gaps can actually test the proposed lifespan correction.
    events.sort(key=lambda e: (not 3 < e["frame"]-e["previous_match_frame"]-1 <= 16,e["frame"]))
    chosen=[]
    for event in events:
        if all(abs(event["frame"]-x["frame"])>10 for x in chosen): chosen.append(event)
        if len(chosen)==3: break
    if len(chosen)<3: raise RuntimeError("Fewer than three separated ID-switch cases; inspect another held-out sequence")
    paths=sequence_frames(root)
    if len(paths)!=info["frames"]: raise FileNotFoundError("Failure gallery requires the original sequence images")
    analyses=[]
    examples=trajectory_chunks([root],length=33,stride=100)[:16]
    curve=np.mean([gradient_horizon(model,x[None]) for x in examples],axis=0)
    gradient_ratio=float(curve[0]/max(curve[8],1e-20))
    for index,event in enumerate(chosen,1):
        frames=[event["previous_match_frame"],event["frame"],min(info["frames"],event["frame"]+2)]
        fig,axes=plt.subplots(3,3,figsize=(14,8))
        for col,f in enumerate(frames):
            image=np.asarray(Image.open(paths[f-1]).convert("RGB"))
            layers=[truth_records({f:gt[f]})[f],pred[f],[dict(track_id=k,bbox=b) for k,b in debug[f].items()]]
            for row,items in enumerate(layers):
                axes[row,col].imshow(annotate(image,items))
                axes[row,col].axis("off")
                axes[row,col].set_title(f"{['GT','prediction','RNN forecast'][row]} / frame {f}")
        finish(fig,OUT/f"failures/case_{index}.png")
        identity=event["gt_id"]
        f=event["frame"]
        occlusion=0
        for previous in range(f-1,0,-1):
            objects=[x for x in gt[previous] if x.gt_id==identity]
            if objects and objects[0].visibility>=.2: break
            occlusion+=1
        target=next(x for x in gt[f] if x.gt_id==identity)
        from metrics import overlap
        max_iou=max((overlap(target.bbox,b) for b in debug[f].values()),default=0)
        gap=event["frame"]-event["previous_match_frame"]-1
        analyses.append(dict(**event,preceding_low_visibility_frames=occlusion,
            best_forecast_iou=max_iou,visible_fraction=target.visibility,
            bptt_window=16,gradient_decay_factor_at_8=gradient_ratio,
            frames_without_spatial_match=gap,
            diagnosis=("The gap without spatial match exceeds track lifetime; test lifespan 16. GT visibility distinguishes detector misses from full occlusion." if gap>3 else
                       "Identity switched despite short occlusion: inspect overlapping boxes and motion error; longer lifetime alone may not help."),
            figure=f"case_{index}.png"))
    save_metrics(analyses,OUT/"failures/diagnoses.json")

def main():
    torch.set_num_threads(2)
    synthetic_suite()
    ablation_cells()
    # Fixed primary configuration, selected in advance; seed zero is not cherry-picked.
    import shutil
    Path("checkpoints").mkdir(exist_ok=True)
    shutil.copyfile("checkpoints/ablation/gru_T16_s0.pt","checkpoints/gru.pt")
    model,_=load_checkpoint("checkpoints/gru.pt")
    baseline_suite(model)
    root=resolve_sequence(DATA_ROOT,TEST[0])
    stress_detector(model,root)
    memory_suite(model,root)
    correction_suite(model)
    failure_gallery(model,root)

if __name__=="__main__": main()
