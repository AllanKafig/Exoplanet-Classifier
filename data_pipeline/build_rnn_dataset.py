"""Build the RNN time-series dataset.

For each star, fold its full (all-quarter) Kepler light curve on the known KOI
period and bin it into a fixed-length, transit-aligned flux vector
(flux_0 … flux_{N_BINS-1}), then attach the label. Each output row is:

    kep_id, flux_0 … flux_{N_BINS-1}, label

This reuses build_dataset.py, and labels and star selection also come
from it, so this dataset covers the SAME stars as the GB dataset.
"""
import sys
import socket
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binned_statistic

# Put this folder (data_pipeline/) on the path so `from build_gb_dataset import ...`
# and its own `from light_curve import ...` resolve when run from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
sys.path.append(str(Path(__file__).resolve().parent))
from build_gb_dataset import (
    get_star_labels,
    pick_all_stars,
    download_lightcurve,
    _looks_like_network_error,
)

DATA_DIR = REPO_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "rnn_timeseries.csv"

SAMPLE_LIMIT = None      # set to None for the full ~6640-star build
SAVE_EVERY = 25          # checkpoint to disk every N completed stars
DOWNLOAD_TIMEOUT = 300 
MAX_CONSECUTIVE_NET_ERRORS = 40
N_BINS = 2000          # length of the folded flux vector
N_DIPS = 2             # how many transit dips to span in the folded window

socket.setdefaulttimeout(DOWNLOAD_TIMEOUT)


def get_fold_params():
    """Per-star period and first-transit time, needed to fold the light curve.

    Returns a DataFrame with one row per star (the first KOI per kepid):
    ["kepid", "koi_period", "koi_time0bk"].
    """
    koi = pd.read_csv(DATA_DIR / "koi_cumulative.csv", comment="#")
    return (koi[["kepid", "koi_period", "koi_time0bk"]]
            .drop_duplicates(subset="kepid", keep="first"))


def fold_and_bin(time, flux, period, t0, n_bins=N_BINS, n_dips=N_DIPS):
    """Collapse a variable-length light curve into a fixed-length, transit-aligned
    vector. FOLD on the period so every orbit's transit lands at the same phase,
    then BIN into n_bins slots taking the median flux per bin (median averages
    down noise). Empty bins are filled by interpolating from their neighbors.
    """
    phase = ((time - t0 + 0.5 * (n_dips * period)) % (n_dips * period)) / (n_dips * period) - 0.5
    binned, _, _ = binned_statistic(phase, flux, statistic="median", bins=n_bins, range=(-0.5, 0.5))

    idx = np.arange(n_bins)
    nan = np.isnan(binned)
    if np.any(nan):
        binned[nan] = np.interp(idx[nan], idx[~nan], binned[~nan])
    return binned


def get_lc_features(lc, kep_id, period, t0):
    """Normalize, fold, and bin a light curve into {kep_id, flux_0 … flux_N-1}."""
    time = np.asarray(lc.time.value)
    flux = np.asarray(lc.flux.value)

    flux = flux / np.nanmedian(flux) - 1.0   # normalize flux around 0
    flux = fold_and_bin(time, flux, period, t0)

    row = {"kep_id": kep_id}
    for i, f in enumerate(flux):
        row[f"flux_{i}"] = f
    return row


def save_rows(rows):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)


def main():
    labels = get_star_labels()
    stars = pick_all_stars(labels, limit=SAMPLE_LIMIT)
    fold_params = get_fold_params()
    print(f"processing {len(stars)} stars "
          f"(socket timeout {DOWNLOAD_TIMEOUT}s, checkpoint every {SAVE_EVERY})")

    # resume: skip stars already saved, retry the rest 
    rows = []
    done = set()
    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE)
        rows = existing.to_dict("records")
        done = set(existing["kep_id"].astype(int))
        print(f"resuming: {len(done)} stars already saved in {OUTPUT_FILE.name} — skipping them")

    consecutive_net_errors = 0
    for count, (_, star) in enumerate(stars.iterrows(), start=1):
        kepid = int(star["kepid"])
        label = int(star["label"])

        if kepid in done:
            continue

        try:
            match = fold_params[fold_params["kepid"] == kepid]
            if len(match) == 0:
                print(f"  [{count}/{len(stars)}] skip KIC {kepid}  (no KOI period/t0)")
                continue
            period = float(match["koi_period"].iloc[0])   # first KOI per star
            t0 = float(match["koi_time0bk"].iloc[0])

            # reads from the local cache when present (author="Kepler"), else MAST
            lc = download_lightcurve(kepid)
            if lc is None:
                print(f"  [{count}/{len(stars)}] skip KIC {kepid}  (no Kepler long-cadence data)")
                consecutive_net_errors = 0
                continue

            features = get_lc_features(lc, kep_id=kepid, period=period, t0=t0)
            features["label"] = label
            rows.append(features)
            done.add(kepid)
            consecutive_net_errors = 0
            print(f"  [{count}/{len(stars)}] OK   KIC {kepid}")

            if len(rows) % SAVE_EVERY == 0:
                save_rows(rows)
                print(f"  checkpoint: saved {len(rows)} rows")

        except Exception as error:
            print(f"  [{count}/{len(stars)}] skip KIC {kepid}  ({error})")
            if _looks_like_network_error(error):
                consecutive_net_errors += 1
                if consecutive_net_errors >= MAX_CONSECUTIVE_NET_ERRORS:
                    print(f"\n{consecutive_net_errors} network errors in a row — connection "
                          f"looks down. Saving progress and stopping; re-run to resume.")
                    break
            else:
                consecutive_net_errors = 0

    save_rows(rows)
    print(f"\nSaved {len(rows)} stars to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
