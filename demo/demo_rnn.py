"""A file that demonstrates the use of the RNN class, on random data."""
import torch
import numpy as np
import pandas as pd

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from RNN.rnn_v2 import ExoplanetRNN, evaluate, scores
from GradientBooster.evaluate import get_scores

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

def load_weights(model, path):
    """Load the model weights from a file."""
    saved = torch.load(path)
    with torch.no_grad():
        for param, w in zip(model.parameters(), saved):
            param.copy_(w.to(param.device))
    return model

def demo_rnn(weights_path, test_X, test_y, threshold=0.3):
    model = ExoplanetRNN()
    load_weights(model, weights_path)

    test_prob = evaluate(model, test_X)
    test_pred = (test_prob >= threshold).astype(int)
    
    test_scores = get_scores(test_y, test_pred, None)
    print(f"\nTest set scores (out of {len(test_y)}):")
    for k, v in test_scores.items():
        print(f"  {k:10s} {v:.4f}")
    
    print("\nIndividual predictions:")
    for i, (pred, true) in enumerate(zip(test_pred, test_y)):
        print(f"  Sample {i}: Predicted={pred}, True={true}")

from demo import get_random_rows, load_data #LEAVE THIS HERE (else you'll get circular import errors)

def main():
    size = 20

    X, y, kep_id = load_data("rnn_timeseries.csv")
    test_X, test_y, test_kep_id = get_random_rows(X, y, kep_id, size, seed=42)
    demo_rnn("exoplanet_rnn_v2.pt", test_X, test_y, threshold=0.3)
    demo_rnn("exoplanet_rnn.pt", test_X, test_y, threshold=0.3)

if __name__ == "__main__":
   main()