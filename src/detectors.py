"""Frozen torchvision detector, with project NMS replacing the internal NMS."""
from .types import Detection

def torchvision_person_detector(device="cpu",score_threshold=.5):
    import torch
    from pathlib import Path
    torch.hub.set_dir(str(Path(".cache/torch").resolve()))
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn,FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
    from torchvision.ops import boxes as box_ops
    from .geometry import nms
    # torchvision detection calls box_ops.nms internally (RPN and ROI heads).
    # Replace this primitive with the implementation authored in this repository.
    def own_nms(boxes,scores,iou_threshold):
        keep=nms(boxes.detach().cpu().tolist(),scores.detach().cpu().tolist(),iou_threshold)
        return torch.tensor(keep,dtype=torch.int64,device=boxes.device)
    box_ops.nms=own_nms
    model=fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT).to(device).eval()
    def detect(image,frame=0):
        tensor=torch.as_tensor(image.copy()).permute(2,0,1).float().div(255).to(device)
        with torch.inference_mode(): result=model([tensor])[0]
        return [Detection(frame,tuple(float(v) for v in box),float(score))
                for box,label,score in zip(result["boxes"],result["labels"],result["scores"])
                if int(label)==1 and float(score)>=score_threshold]
    return detect
