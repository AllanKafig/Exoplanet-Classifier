"""PyTorch nn.GRU reference, trained with the same gradient descent as
rnn_v2 (lr 0.5, 80 epochs, batch 64). The ONLY change vs rnn_v2 is the GRU
itself: torch.nn.GRU + nn.Linear. 
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from rnn_v2 import (downsample, standardize_rows, bce_loss, sgd_step,
                    evaluate, scores, LEARNING_RATE, DEVICE)


class TorchGRU(nn.Module):
    """Use PyTorch's libaray but same shape as our scratch model."""
    def __init__(self, hidden_size=32, input_size=1, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):           # x: (B, T, 1)
        _, h_n = self.gru(x)        # h_n: (1, B, H) -> last hidden state
        return self.out(h_n[-1])[:, 0]


def train_torch_ref(light_curves, labels, epochs, batch_size, learning_rate, seed=0):
    """Same training recipe as rnn_v2.train_rnn: manual BCE + plain SGD step."""
    x = torch.tensor(light_curves, dtype=torch.float32)[:, :, None]
    y = torch.tensor(np.asarray(labels), dtype=torch.float32)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    model = TorchGRU(seed=seed).to(DEVICE)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            loss = bce_loss(model(batch_x), batch_y)
            loss.backward()
            sgd_step(model.parameters(), learning_rate)   # same w = w - lr*grad
            total_loss += loss.item() * len(batch_x)
        print(f"Epoch {epoch + 1}: loss = {total_loss / len(x):.4f}")
    return model


if __name__ == "__main__":
    df = pd.read_csv("data/rnn_timeseries.csv")
    flux_columns = sorted([c for c in df.columns if "flux" in c], key=lambda c: int(c.split("_")[1]))
    labels = df["label"].to_numpy(dtype=np.float32)
    light_curves = standardize_rows(downsample(np.clip(df[flux_columns].to_numpy(np.float32), -1, 1), factor=8))

    x_train, x_test, y_train, y_test = train_test_split(
        light_curves, labels, test_size=0.2, random_state=0, stratify=labels)

    model = train_torch_ref(x_train, y_train, epochs=80, batch_size=64, learning_rate=LEARNING_RATE)

    prob = evaluate(model, x_test)
    print("\nPyTorch nn.GRU reference (same SGD) performance:")
    for k, v in scores(y_test, prob).items():
        print(f"  {k:10s} {v:.3f}")

    torch.save(model.state_dict(), "exoplanet_rnn_torch_ref.pt")
