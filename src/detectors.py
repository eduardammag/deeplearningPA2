"""Fontes de deteccao: arquivos publicos MOT17 e detector COCO congelado."""
from .types import Detection


def torchvision_person_detector(device="cpu", score_threshold=0.5):
    """Retorna um callable; pesos podem ser baixados pelo torchvision na primeira chamada."""
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights).to(device).eval()
    person_label = 1

    def detect(image, frame=0):
        tensor = torch.as_tensor(image).permute(2, 0, 1).float().div(255).to(device)
        with torch.no_grad():
            result = model([tensor])[0]
        detections = []
        for box, label, score in zip(result["boxes"], result["labels"], result["scores"]):
            if int(label) == person_label and float(score) >= score_threshold:
                detections.append(Detection(frame, tuple(float(value) for value in box), float(score)))
        return detections
    return detect
