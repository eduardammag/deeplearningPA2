"""Training on disjoint MOT17 sequences with truncated BPTT."""
from pathlib import Path
import random
import numpy as np
import torch
from .temporal import MotionRNN,box_to_features,train_step
from .mot17 import load_ground_truth,sequence_info,resolve_sequence

TRAIN=("MOT17-02","MOT17-04")
VALIDATION=("MOT17-05",)
TEST=("MOT17-09","MOT17-10","MOT17-11","MOT17-13")
DATA_ROOT="data_MOT17Labels"

def trajectory_chunks(roots,length=65,stride=32):
    chunks=[]
    for root in roots:
        info=sequence_info(root)
        identities={}
        for f,items in load_ground_truth(root).items():
            for item in items:
                identities.setdefault(item.gt_id,[]).append((f,box_to_features(item.bbox,(info["width"],info["height"]))))
        for observations in identities.values():
            runs,run=[],[]
            previous=None
            for f,box in observations:
                if previous is not None and f!=previous+1:
                    runs.append(run)
                    run=[]
                run.append(box)
                previous=f
            runs.append(run)
            for run in runs:
                for start in range(0,len(run)-length+1,stride):
                    chunks.append(torch.stack(run[start:start+length]))
    if not chunks: raise ValueError("No sufficiently long ground-truth trajectories")
    return torch.stack(chunks)

def parameter_budget(cell,target=14000):
    return min(range(8,160),key=lambda h:abs(sum(p.numel() for p in MotionRNN(cell,h).parameters())-target))

def train_model(sequences,cell="gru",epochs=5,hidden_size=None,checkpoint="checkpoints/gru.pt",
                window=16,seed=0,validation=None,metadata=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    hidden_size=hidden_size or parameter_budget(cell)
    model=MotionRNN(cell,hidden_size)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    data=sequences if isinstance(sequences,torch.Tensor) else torch.cat(sequences)
    history=[]
    best=float("inf")
    Path(checkpoint).parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(epochs):
        order=torch.randperm(len(data))
        losses=[train_step(model,optimizer,data[indices],window) for indices in order.split(64)]
        model.eval()
        with torch.no_grad():
            val=validation if validation is not None else data
            val_losses=[]
            for batch in val.split(128):
                prediction,_=model(batch[:,:-1])
                val_losses.append(float(torch.nn.functional.smooth_l1_loss(prediction,batch[:,1:],beta=.01)))
            val_loss=float(np.mean(val_losses))
        history.append(dict(epoch=epoch+1,train_loss=float(np.mean(losses)),validation_loss=val_loss))
        if val_loss<best:
            best=val_loss
            torch.save(dict(model=model.state_dict(),cell=cell,hidden_size=hidden_size,window=window,
                seed=seed,history=history.copy(),parameters=sum(p.numel() for p in model.parameters()),
                metadata=metadata or {}),checkpoint)
    from .temporal import load_checkpoint
    return load_checkpoint(checkpoint)[0],history

def main():
    train_roots=[resolve_sequence(DATA_ROOT,s) for s in TRAIN]
    validation_roots=[resolve_sequence(DATA_ROOT,s) for s in VALIDATION]
    data=trajectory_chunks(train_roots)
    validation=trajectory_chunks(validation_roots)
    _,history=train_model(data,validation=validation,metadata=dict(train=list(TRAIN),validation=list(VALIDATION),test=list(TEST)))
    print(history)

if __name__=="__main__": main()
