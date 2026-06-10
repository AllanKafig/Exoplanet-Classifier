"""A file that demonstrates the use of the GB class, on random data."""
import numpy as np
import pandas as pd

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "GradientBooster")) 

from GradientBooster.gradient_booster import GradientBoosterClassifier
from GradientBooster.evaluate import get_scores

def demo_gb(model_path, test_X, test_y):
    model = GradientBoosterClassifier.load_model(model_path)
    test_pred = model.predict(test_X)

    test_scores = get_scores(test_y, test_pred, model.predict_proba(test_X))
    print(f"\nTest set scores (out of {len(test_y)}):")
    for k, v in test_scores.items():
        print(f"  {k:10s} {v:.4f}")

    print("\nIndividual predictions:")
    for i, (pred, true) in enumerate(zip(test_pred, test_y)):
        print(f"  Sample {i}: Predicted={pred}, True={true}")

from demo import get_random_rows, load_data #LEAVE THIS HERE (else you'll get circular import errors)

def main():
    size = 10

    X, y, kep_id = load_data("features_with_koi.csv")
    test_X, test_y, test_kep_id = get_random_rows(X, y, kep_id, size, seed=42)
    demo_gb("exoplanet_gb.pkl", test_X, test_y)

if __name__ == "__main__":
   main()