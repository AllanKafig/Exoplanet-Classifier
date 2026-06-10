import numpy as np
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

def get_random_rows(X, y, kep_id, size, seed=None):
    rng = np.random.default_rng(seed)
    indices = rng.choice(X.shape[0], size=size, replace=False)

    y_sample = y[indices] if y is not None else None

    return X[indices], y_sample, kep_id[indices]

def load_data(path):
    """Load the real exoplanet dataset. X is features and y is 0/1."""
    df = pd.read_csv(os.path.join(DATA_DIR, path))
    
    has_label = 'label' in df.columns

    if has_label:
        df = df.dropna(subset=["label"])

    df = df.dropna() 
    
    y = df["label"].values if has_label else None
    drop_cols = ['kep_id'] + (['label'] if has_label else [])
    X = df.drop(columns=drop_cols).values 

    kep_id = df["kep_id"].values

    return X, y, kep_id

def get_data_from_kep_id(X, y, kep_ids, kep_ids_to_use):
    """Return the X, y rows whose kep_id is in kep_ids_to_use, in that order."""
    pos_of = {k: i for i, k in enumerate(kep_ids)}     #kep_id -> row position
    positions, found_ids = [], []
    for k in kep_ids_to_use:
        if k in pos_of:
            positions.append(pos_of[k])
            found_ids.append(k)

    missing = len(kep_ids_to_use) - len(positions)
    if missing:
        print(f"Missing {missing} rows")

    return X[positions], y[positions], np.array(found_ids)

from demo_gb import demo_gb #LEAVE THIS HERE (else you'll get circular import errors)
from demo_rnn import demo_rnn #LEAVE THIS HERE (else you'll get circular import errors)

def run_on_known_data(size):
    """Run our models on known CONFIRMED/FALSE POSITIVE exoplanet data"""
    X_feat, y_feat, kep_feat = load_data("features_with_koi.csv")
    X_ts, y_ts, kep_ts = load_data("rnn_timeseries.csv")

    test_X_feat, test_y_feat, test_kep_id = get_random_rows(X_feat, y_feat, kep_feat, size, 42)
    test_X_ts, test_y_ts, found_ids = get_data_from_kep_id(X_ts, y_ts, kep_ts, test_kep_id)

    print("\n--------Demonstrating the Gradient Booster model--------")
    demo_gb("exoplanet_gb.pkl", test_X_feat, test_y_feat)

    print("\n--------Demonstrating the RNN (SGD) model--------")
    demo_rnn("exoplanet_rnn_v2.pt", test_X_ts, test_y_ts, threshold=0.3)

    print("\n--------Demonstrating the RNN (Adam) model--------")
    demo_rnn("exoplanet_rnn.pt", test_X_ts, test_y_ts, threshold=0.3)

def run_on_candidates(size):
    """Run our models on data that does not have a True label (unconfirmed exoplanet data)
    Shows what the model predicts for candidates that NASA has not yet confirmed as planets or false positives.
    """
    X_ts, y_ts, kep_ts = load_data("rnn_candidates_timeseries.csv")

    test_X_ts, test_y_ts, test_kep_id = get_random_rows(X_ts, y_ts, kep_ts, size, 42)
    
    print("\n--------Demonstrating the RNN (SGD) model on Candidates--------")
    demo_rnn("exoplanet_rnn_v2.pt", test_X_ts, test_y_ts, threshold=0.3)

    print("\n--------Demonstrating the RNN (Adam) model on Candidates--------")
    demo_rnn("exoplanet_rnn.pt", test_X_ts, test_y_ts, threshold=0.3)


def main():
    size = 30
    run_on_known_data(size)
    run_on_candidates(size)

if __name__ == "__main__":
   main()