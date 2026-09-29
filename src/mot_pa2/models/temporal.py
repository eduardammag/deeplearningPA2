"""Persistent recurrent motion model on normalized boxes."""
import torch
from torch import nn

def box_to_features(box,image_size=(1,1)):
    x1,y1,x2,y2=box
    w,h=image_size
    return torch.tensor([(x1+x2)/(2*w),(y1+y2)/(2*h),(x2-x1)/w,(y2-y1)/h],dtype=torch.float32)

def features_to_box(values,image_size):
    cx,cy,w,h=values
    iw,ih=image_size
    w,h=max(float(w),1/iw),max(float(h),1/ih)
    return ((float(cx)-w/2)*iw,(float(cy)-h/2)*ih,(float(cx)+w/2)*iw,(float(cy)+h/2)*ih)

class MotionRNN(nn.Module):
    def __init__(self,cell="gru",hidden_size=64,input_size=4):
        super().__init__()
        self.cell_name,self.hidden_size=cell,hidden_size
        self.recurrent={"rnn":nn.RNNCell,"lstm":nn.LSTMCell,"gru":nn.GRUCell}[cell](input_size,hidden_size)
        self.head=nn.Linear(hidden_size,4)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def step(self,observation,state=None):
        if state is None:
            state=observation.new_zeros((len(observation),self.hidden_size))
            if self.cell_name=="lstm": state=(state,state.clone())
        state=self.recurrent(observation,state)
        hidden=state[0] if isinstance(state,tuple) else state
        return observation+self.head(hidden),state

    def forward(self,sequence,state=None):
        outputs=[]
        for observation in sequence.unbind(1):
            prediction,state=self.step(observation,state)
            outputs.append(prediction)
        return torch.stack(outputs,1),state

def detach_state(state):
    return tuple(s.detach() for s in state) if isinstance(state,tuple) else state.detach()

def train_step(model,optimizer,sequence,window=16,clip=1.):
    model.train()
    state,losses=None,[]
    for start in range(0,sequence.shape[1]-1,window):
        stop=min(start+window,sequence.shape[1]-1)
        optimizer.zero_grad()
        prediction,state=model(sequence[:,start:stop],state)
        loss=nn.functional.smooth_l1_loss(prediction,sequence[:,start+1:stop+1],beta=.01)
        loss.backward()
        if clip is not None: nn.utils.clip_grad_norm_(model.parameters(),clip)
        optimizer.step()
        state=detach_state(state)
        losses.append(float(loss.detach()))
    return sum(losses)/len(losses)

def gradient_horizon(model,sequence,max_lag=None):
    """Norm dL_final/dh_(t-k); LSTM h is measured separately from its cell state."""
    model.eval()
    states,state=[],None
    for observation in sequence[:,:-1].unbind(1):
        prediction,state=model.step(observation,state)
        states.append(state[0] if isinstance(state,tuple) else state)
    loss=nn.functional.smooth_l1_loss(prediction,sequence[:,-1],beta=.01)
    grads=torch.autograd.grad(loss,states,allow_unused=True)
    values=[float(g.norm(dim=-1).mean()) if g is not None else 0. for g in grads][::-1]
    return values[:max_lag] if max_lag else values

def load_checkpoint(path):
    checkpoint=torch.load(path,map_location="cpu",weights_only=True)
    model=MotionRNN(checkpoint["cell"],checkpoint["hidden_size"])
    model.load_state_dict(checkpoint["model"])
    return model.eval(),checkpoint
