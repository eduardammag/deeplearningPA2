"""Compare public FRCNN and frozen COCO torchvision on the same held-out sequence."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from .mot17 import sequence_frames,sequence_info,load_ground_truth,load_public_detections,resolve_sequence
from .training import DATA_ROOT,TEST
from .evaluation import evaluate_baseline,save_metrics,detection_records,truth_records
from metrics import detection_map
from .types import Detection

def main():
    import torch
    torch.set_num_threads(2)
    torch.hub.set_dir(str(Path(".cache/torch").resolve()))
    root=resolve_sequence(DATA_ROOT,TEST[0])
    paths=sequence_frames(root)
    if len(paths)!=sequence_info(root)["frames"]: raise FileNotFoundError("Download complete sequence images first")
    cache=Path("outputs/torchvision_mobilenet_detections.json")
    if cache.exists():
        saved=json.loads(cache.read_text())
        detected={int(f):[Detection(int(f),tuple(x["bbox"]),x["score"]) for x in items] for f,items in saved.items()}
    else:
        from .detectors import torchvision_person_detector
        detect=torchvision_person_detector()
        detected={}
        partial=Path("outputs/torchvision_mobilenet_partial.json")
        if partial.exists():
            saved=json.loads(partial.read_text())
            detected={int(f):[Detection(int(f),tuple(x["bbox"]),x["score"]) for x in items] for f,items in saved.items()}
        for f,path in enumerate(paths,1):
            if f in detected: continue
            detected[f]=detect(np.asarray(Image.open(path).convert("RGB")),f)
            if f%10==0:
                save_metrics(detection_records(detected),partial)
                print("torchvision",f,"/",len(paths),flush=True)
        save_metrics(detection_records(detected),cache)
    gt=load_ground_truth(root)
    results={}
    for name,det in (("public_FRCNN",load_public_detections(root)),("torchvision_COCO",detected)):
        result,_,truth=evaluate_baseline(det,gt)
        results[name]=dict(**result,mAP=detection_map(detection_records(det),truth))
    save_metrics(dict(sequence=TEST[0],detectors=results,torchvision_min_size=320,
                      torchvision_max_size=640,score_threshold=.5,
                      weights="FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1"), "outputs/detector_comparison.json")
    print(results,flush=True)

if __name__=="__main__": main()
