"""MOT17 readers. Frame clock and visibility are retained explicitly."""
from pathlib import Path
import csv
import configparser
from .geometry import xywh_to_xyxy,nms
from .types import Detection

def read_mot_file(path):
    with Path(path).open(newline="") as stream:
        return [[float(v) for v in row] for row in csv.reader(stream) if row and not row[0].startswith("#")]

def sequence_info(root):
    parser=configparser.ConfigParser()
    parser.read(Path(root)/"seqinfo.ini")
    s=parser["Sequence"]
    return dict(width=int(s["imWidth"]),height=int(s["imHeight"]),frames=int(s["seqLength"]),fps=int(s["frameRate"]))

def resolve_sequence(root,name,detector="FRCNN"):
    path=Path(root)/"train"/name
    if not path.exists(): path=Path(root)/"train"/f"{name}-{detector}"
    if not path.exists(): raise FileNotFoundError(path)
    return path

def load_ground_truth(sequence_root,visible_threshold=0.):
    info=sequence_info(sequence_root)
    frames={f:[] for f in range(1,info["frames"]+1)}
    for row in read_mot_file(Path(sequence_root)/"gt"/"gt.txt"):
        frame,identity,x,y,width,height,confidence,cls,visibility=row[:9]
        if int(cls)!=1 or confidence<=0 or visibility<visible_threshold: continue
        frames[int(frame)].append(Detection(int(frame),xywh_to_xyxy((x,y,width,height)),confidence,int(identity),visibility))
    return frames

def load_public_detections(sequence_root,detector="FRCNN",min_score=0.,nms_threshold=.7):
    root=Path(sequence_root)
    if root.name.endswith(("-DPM","-FRCNN","-SDP")) and not root.name.endswith("-"+detector):
        raise ValueError("Detector does not match the sequence directory")
    info=sequence_info(root)
    frames={f:[] for f in range(1,info["frames"]+1)}
    for row in read_mot_file(root/"det"/"det.txt"):
        frame,_,x,y,width,height,score=row[:7]
        if score>=min_score:
            frames[int(frame)].append(Detection(int(frame),xywh_to_xyxy((x,y,width,height)),score))
    for frame,items in frames.items():
        frames[frame]=[items[i] for i in nms([x.bbox for x in items],[x.score for x in items],nms_threshold)]
    return frames

def sequence_frames(sequence_root):
    return sorted((Path(sequence_root)/"img1").glob("*.jpg"))

def occlusion_durations(root,threshold=.2):
    identities={}
    for row in read_mot_file(Path(root)/"gt"/"gt.txt"):
        if row[6]>0 and int(row[7])==1:
            identities.setdefault(int(row[1]),{})[int(row[0])]=row[8]
    durations=[]
    for observations in identities.values():
        visible=[f for f,v in observations.items() if v>=threshold]
        for a,b in zip(visible,visible[1:]):
            if b>a+1: durations.append(b-a-1)
    return durations
