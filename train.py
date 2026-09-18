"""Treino do modelo de movimento em trajetorias MOT ou sinteticas."""
import argparse
from pathlib import Path
import torch
from mot_pa2.temporal import MotionRNN, train_step


def train_model(sequences, cell="gru", epochs=10, hidden_size=64, checkpoint="checkpoints/gru.pt"):
    model = MotionRNN(cell, hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    for _ in range(epochs):
        losses = [train_step(model, optimizer, sequence) for sequence in sequences]
        history.append(sum(losses) / max(1, len(losses)))
    Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cell": cell, "hidden_size": hidden_size, "history": history}, checkpoint)
    return model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/gru.pt")
    parser.add_argument("--cell", choices=("rnn", "lstm", "gru"), default="gru")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    sequence = torch.zeros(1, 32, 4)
    _, history = train_model([sequence], args.cell, args.epochs, checkpoint=args.checkpoint)
    print({"checkpoint": args.checkpoint, "final_loss": history[-1]})

if __name__ == "__main__":
    main()
