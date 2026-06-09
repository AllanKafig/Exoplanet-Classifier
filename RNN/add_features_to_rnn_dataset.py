"""Attach the 13 leakage-free stellar features to the folded RNN dataset.

For the RNN (a light-curve detector), the only leakage-safe scalar features are
STELLAR properties — they describe the star, independent of the transit. The
transit-fit columns (koi_depth, koi_prad, koi_period, koi_model_snr, ...) are EXCLUDED: 
they're derived from the very transit the RNN is meant to detect, so feeding them 
would be leakage.

Reads lightcurve_folded_timeseries.csv (kept as the flux-only baseline) and
writes lightcurve_folded_timeseries_with_stellar.csv. Missing stellar values are
filled with the column median so no stars are lost.
"""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
IN_CSV = DATA_DIR / "lightcurve_folded_timeseries.csv"
OUT_CSV = DATA_DIR / "rnn_timeseries.csv"

STELLAR_FEATURES = [
    "koi_steff", "koi_slogg", "koi_smet", "koi_srad", "koi_smass",   # core stellar params
    "koi_kepmag", "koi_gmag", "koi_rmag", "koi_imag", "koi_zmag",    # photometric magnitudes
    "koi_jmag", "koi_hmag", "koi_kmag",
]


def main():
    folded = pd.read_csv(IN_CSV)
    koi = (pd.read_csv(DATA_DIR / "koi_cumulative.csv", comment="#")
           .drop_duplicates("kepid"))   # one row per star (stellar props are per-star)

    merged = (folded.merge(koi[["kepid"] + STELLAR_FEATURES],
                           left_on="kep_id", right_on="kepid", how="left")
                    .drop(columns="kepid"))

    # fill missing stellar values with the column median (keeps all rows)
    filled = {}
    for c in STELLAR_FEATURES:
        n = int(merged[c].isna().sum())
        if n:
            merged[c] = merged[c].fillna(merged[c].median())
            filled[c] = n

    # keep label as the LAST column for a clean X / y split downstream
    cols = [c for c in merged.columns if c != "label"] + ["label"]
    merged = merged[cols]

    merged.to_csv(OUT_CSV, index=False)
    n_flux = sum(c.startswith("flux_") for c in merged.columns)
    print(f"wrote {OUT_CSV.name}: {len(merged)} rows, {merged.shape[1]} cols "
          f"(kep_id + {n_flux} flux + {len(STELLAR_FEATURES)} stellar + label)")
    print("median-filled missing values:", filled or "none")


if __name__ == "__main__":
    main()
