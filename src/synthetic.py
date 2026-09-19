"""Deterministic ellipses with a pixel-level depth buffer and controlled full occlusion."""
from dataclasses import dataclass
import numpy as np
from PIL import Image,ImageDraw
from .types import Detection

@dataclass(frozen=True)
class SyntheticConfig:
    frames:int=45
    objects:int=8
    typical_speed:float=2.
    occlusion_duration:int=8
    width:int=128
    height:int=128
    seed:int=0
    image_noise:float=2.
    contrast:float=1.

def generate_video(config):
    rng=np.random.default_rng(config.seed)
    cols=int(np.ceil(np.sqrt(config.objects)))
    centers=np.array([(18+(i%cols)*(config.width-36)/max(cols-1,1),
                       18+(i//cols)*(config.height-36)/max(cols-1,1)) for i in range(config.objects)])
    velocity=rng.normal(0,config.typical_speed,(config.objects,2))
    radii=rng.uniform(4,7,(config.objects,2))
    depth=list(rng.permutation(config.objects))
    if config.objects>=2:
        radii[0]=[4,4]
        radii[1]=[9,9]
        depth=[i for i in depth if i not in (0,1)]+[0,1]
    colors=np.clip(18+config.contrast*(rng.integers(60,230,(config.objects,3))-18),0,255).astype(np.uint8)
    video,truth=[],[]
    margin=12
    extent=np.array([config.width,config.height])-2*margin
    for f in range(config.frames):
        raw=centers-margin+velocity*f
        positions=margin+extent-np.abs(raw%(2*extent)-extent)
        start=max(1,config.frames//2-config.occlusion_duration//2)
        if config.objects>=2 and config.occlusion_duration:
            # Continuous approach, an exact dwell interval, then separation.
            transition=6
            if start-transition<=f<start:
                weight=(f-(start-transition))/transition
                positions[1]=(1-weight)*positions[1]+weight*positions[0]
            elif start<=f<start+config.occlusion_duration:
                positions[1]=positions[0]
            elif start+config.occlusion_duration<=f<start+config.occlusion_duration+transition:
                weight=1-(f-(start+config.occlusion_duration)+1)/transition
                positions[1]=(1-weight)*positions[1]+weight*positions[0]
        image=Image.new("RGB",(config.width,config.height),(18,18,24))
        labels=Image.new("I",image.size,0)
        boxes={}
        for identity in depth:
            x,y=positions[identity]
            rx,ry=radii[identity]
            box=(float(x-rx),float(y-ry),float(x+rx),float(y+ry))
            boxes[identity]=box
            ImageDraw.Draw(image).ellipse(box,fill=tuple(colors[identity]))
            ImageDraw.Draw(labels).ellipse(box,fill=int(identity)+1)
        ids=np.asarray(labels)
        truth.append([Detection(f,boxes[i],1.,i+1) for i in range(config.objects) if np.any(ids==i+1)])
        pixels=np.asarray(image).astype(float)+rng.normal(0,config.image_noise,(config.height,config.width,3))
        video.append(np.clip(pixels,0,255).astype(np.uint8))
    return np.stack(video),truth

def corrupt_detections(truth,drop_probability=.1,noise_std=1.,false_positive_rate=.1,
                       width=128,height=128,seed=0):
    rng=np.random.default_rng(seed)
    result=[]
    for f,items in enumerate(truth):
        detections=[]
        for item in items:
            if rng.random()<drop_probability: continue
            x1,y1,x2,y2=np.asarray(item.bbox)+rng.normal(0,noise_std,4)
            x1,x2=sorted(np.clip([x1,x2],0,width))
            y1,y2=sorted(np.clip([y1,y2],0,height))
            if x2<=x1 or y2<=y1: continue
            detections.append(Detection(f,(float(x1),float(y1),float(x2),float(y2)),item.score))
        for _ in range(rng.poisson(false_positive_rate*max(1,len(items)))):
            x,y=rng.uniform(0,width-18),rng.uniform(0,height-18)
            detections.append(Detection(f,(x,y,x+rng.uniform(5,18),y+rng.uniform(5,18)),.2))
        result.append(detections)
    return result
