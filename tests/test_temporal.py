"""Core temporal regressions, including persistent state and hidden-state gradients."""
import torch
from mot_pa2.temporal import MotionRNN,train_step,gradient_horizon
from mot_pa2.tracking import TemporalTracker
from mot_pa2.types import Detection

def test_all_recurrent_cells_have_box_output():
    sequence=torch.randn(2,8,4)
    for cell in ("rnn","lstm","gru"):
        prediction,state=MotionRNN(cell)(sequence)
        assert prediction.shape==sequence.shape
        assert state is not None

def test_incremental_state_matches_whole_sequence():
    torch.manual_seed(1)
    seq=torch.rand(1,8,4)
    for cell in ("rnn","lstm","gru"):
        model=MotionRNN(cell,8)
        torch.nn.init.normal_(model.head.weight)
        expected,_=model(seq)
        state=None
        results=[]
        for x in seq.unbind(1):
            y,state=model.step(x,state)
            results.append(y)
        assert torch.allclose(expected,torch.stack(results,1))

def test_hidden_gradient_and_training():
    model=MotionRNN("gru",8)
    sequence=torch.arange(33)[None,:,None].expand(1,33,4).float()/100
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    before=float(torch.nn.functional.mse_loss(model(sequence[:,:-1])[0],sequence[:,1:]).detach())
    for _ in range(10): train_step(model,optimizer,sequence,window=4)
    after=float(torch.nn.functional.mse_loss(model(sequence[:,:-1])[0],sequence[:,1:]).detach())
    assert after<before
    curve=gradient_horizon(model,sequence)
    assert len(curve)==32 and curve[0]>0
    assert all(x>=0 for x in curve)

def test_temporal_tracks_survive_missing_observations_and_expire():
    tracker=TemporalTracker(MotionRNN(hidden_size=8),max_missed=2)
    first=tracker.update([Detection(0,(10,10,20,20))])[0][0]["track_id"]
    assert tracker.update([])[0][0]["track_id"]==first
    assert tracker.update([])[0][0]["track_id"]==first
    assert tracker.update([])[0]==[]
    assert tracker.states=={}
    assert tracker.update([Detection(4,(10,10,20,20))])[0][0]["track_id"]!=first
