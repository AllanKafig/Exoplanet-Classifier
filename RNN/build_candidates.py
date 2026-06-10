"""Build the CANDIDATE time-series dataset (unlabeled) for scoring with the RNN.

Same fold/bin pipeline as build_rnn_dataset.py, but selects CANDIDATE KOIs
(unknown ground truth). Output columns match rnn_timeseries.csv minus `label`:
    kep_id, flux_0 ... flux_1999, <13 stellar features>
"""
import sys
import socket
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binned_statistic

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "src"))
from src.build_gb_dataset import download_lightcurve, _looks_like_network_error

DATA_DIR = REPO_ROOT / "data"
# light curves may be cached in the project dir OR lightkurve's home cache
# (the latter is where download_lightcurve actually saves them).
CACHE_DIRS = [REPO_ROOT / "kepler_data" / "mastDownload" / "Kepler",
              Path.home() / ".lightkurve" / "cache" / "mastDownload" / "Kepler"]
FLUX_FILE = DATA_DIR / "candidates_flux.csv"          # checkpoint (flux only)
OUTPUT_FILE = DATA_DIR / "rnn_candidates_timeseries.csv"  # final (flux + stellar)

N_BINS, N_DIPS = 2000, 2
SAVE_EVERY = 25
CACHED_ONLY = False   # False = also download the non-cached candidates from MAST
SAMPLE_LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None
socket.setdefaulttimeout(300)

STELLAR_FEATURES = [
    "koi_steff", "koi_slogg", "koi_smet", "koi_srad", "koi_smass",
    "koi_kepmag", "koi_gmag", "koi_rmag", "koi_imag", "koi_zmag",
    "koi_jmag", "koi_hmag", "koi_kmag",
]


def get_candidate_stars():
    koi = pd.read_csv(DATA_DIR / "koi_cumulative.csv", comment="#")
    cand = koi[koi["koi_disposition"] == "CANDIDATE"].drop_duplicates("kepid", keep="first")
    return cand[["kepid", "koi_period", "koi_time0bk"]].reset_index(drop=True)


def is_cached(kepid):
    pat = f"kplr{int(kepid):09d}_lc_*"
    return any(any(d.glob(pat)) for d in CACHE_DIRS if d.exists())


def fold_and_bin(time, flux, period, t0, n_bins=N_BINS, n_dips=N_DIPS):
    phase = ((time - t0 + 0.5 * (n_dips * period)) % (n_dips * period)) / (n_dips * period) - 0.5
    binned, _, _ = binned_statistic(phase, flux, statistic="median", bins=n_bins, range=(-0.5, 0.5))
    idx = np.arange(n_bins)
    nan = np.isnan(binned)
    if np.any(nan):
        binned[nan] = np.interp(idx[nan], idx[~nan], binned[~nan])
    return binned


def get_lc_features(lc, kep_id, period, t0):
    time = np.asarray(lc.time.value)
    flux = np.asarray(lc.flux.value)
    flux = flux / np.nanmedian(flux) - 1.0
    flux = fold_and_bin(time, flux, period, t0)
    row = {"kep_id": kep_id}
    for i, f in enumerate(flux):
        row[f"flux_{i}"] = f
    return row


def merge_stellar(flux_df):
    koi = pd.read_csv(DATA_DIR / "koi_cumulative.csv", comment="#").drop_duplicates("kepid")
    m = (flux_df.merge(koi[["kepid"] + STELLAR_FEATURES], left_on="kep_id", right_on="kepid", how="left")
                .drop(columns="kepid"))
    for c in STELLAR_FEATURES:
        m[c] = m[c].fillna(m[c].median())
    return m


def main():
    stars = get_candidate_stars()
    # Drop stars already in the training set (they have a confirmed/FP KOI too,
    # so the model saw them) -> scoring them would be leakage.
    trained = set(pd.read_csv(DATA_DIR / "rnn_timeseries.csv")["kep_id"].astype(int))
    stars = stars[~stars["kepid"].isin(trained)].reset_index(drop=True)
    if CACHED_ONLY:
        stars = stars[stars["kepid"].apply(is_cached)].reset_index(drop=True)
    if SAMPLE_LIMIT:
        stars = stars.head(SAMPLE_LIMIT)
    print(f"processing {len(stars)} candidate stars "
          f"(cached_only={CACHED_ONLY}, checkpoint every {SAVE_EVERY})")

    rows, done = [], set()
    if FLUX_FILE.exists():
        existing = pd.read_csv(FLUX_FILE)
        rows = existing.to_dict("records")
        done = set(existing["kep_id"].astype(int))
        print(f"resuming: {len(done)} candidates already built")

    net_errors = 0
    for count, (_, star) in enumerate(stars.iterrows(), start=1):
        kepid = int(star["kepid"])
        if kepid in done:
            continue
        try:
            period, t0 = float(star["koi_period"]), float(star["koi_time0bk"])
            lc = download_lightcurve(kepid)
            if lc is None:
                print(f"  [{count}/{len(stars)}] skip KIC {kepid}  (no data)")
                net_errors = 0
                continue
            rows.append(get_lc_features(lc, kepid, period, t0))
            done.add(kepid)
            net_errors = 0
            print(f"  [{count}/{len(stars)}] OK   KIC {kepid}")
            if len(rows) % SAVE_EVERY == 0:
                pd.DataFrame(rows).to_csv(FLUX_FILE, index=False)
                print(f"  checkpoint: {len(rows)} rows")
        except Exception as error:
            print(f"  [{count}/{len(stars)}] skip KIC {kepid}  ({error})")
            net_errors = net_errors + 1 if _looks_like_network_error(error) else 0
            if net_errors >= 40:
                print("\nnetwork looks down — saving and stopping; re-run to resume.")
                break

    pd.DataFrame(rows).to_csv(FLUX_FILE, index=False)
    merge_stellar(pd.DataFrame(rows)).to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(rows)} candidates to {OUTPUT_FILE.name} "
          f"(+ flux checkpoint {FLUX_FILE.name})")


if __name__ == "__main__":
    main()
