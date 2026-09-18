import torch
from mot_pa2.temporal import MotionRNN


def test_all_recurrent_cells_have_box_output():
    sequence = torch.randn(2, 8, 4)
    for cell in ("rnn", "lstm", "gru"):
        prediction, state = MotionRNN(cell)(sequence)
        assert prediction.shape == sequence.shape
        assert state is not None
