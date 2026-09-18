"""Modelos recorrentes autorais para previsão de movimento por track."""
from dataclasses import dataclass
import torch
from torch import nn


def box_to_features(box):
    x1, y1, x2, y2 = box
    return torch.tensor([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=torch.float32)


class MotionRNN(nn.Module):
    def __init__(self, cell="gru", hidden_size=64, input_size=4):
        super().__init__()
        cells = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
        if cell not in cells:
            raise ValueError(f"cell must be one of {tuple(cells)}")
        self.cell_name = cell
        self.recurrent = cells[cell](input_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 4))

    def forward(self, sequence, state=None):
        output, state = self.recurrent(sequence, state)
        return self.head(output), state


@dataclass
class RecurrentTrackState:
    track_id: int
    hidden: object = None
    missed: int = 0


def train_step(model, optimizer, sequence):
    model.train()
    optimizer.zero_grad()
    prediction, _ = model(sequence[:, :-1])
    loss = torch.nn.functional.smooth_l1_loss(prediction, sequence[:, 1:])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


def gradient_horizon(model, sequence, max_lag=None):
    """Calcula ||d L_t / d h_{t-k}|| por retencao de gradiente no rollout."""
    model.eval()
    sequence = sequence.detach().clone().requires_grad_(True)
    output, _ = model(sequence)
    loss = output[:, -1].square().mean()
    gradient = torch.autograd.grad(loss, sequence)[0].norm(dim=-1).mean(0)
    values = gradient.detach().cpu().tolist()
    return values[-max_lag:] if max_lag else values
