"""MOT17 data location and scene-disjoint experiment splits."""

TRAIN=("MOT17-02","MOT17-04")
VALIDATION=("MOT17-05",)
TEST=("MOT17-09","MOT17-10","MOT17-11","MOT17-13")
DATA_ROOT="data_MOT17Labels"

def validate_split(train=TRAIN,validation=VALIDATION,test=TEST):
    def scene(name):
        return name.rsplit("-",1)[0] if name.endswith(("-DPM","-FRCNN","-SDP")) else name
    groups=[{scene(name) for name in split} for split in (train,validation,test)]
    if any(groups[i]&groups[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError("A scene cannot occur in more than one split, even with different detectors")
    if not all(groups):
        raise ValueError("Train, validation and held-out test sequences are required")

