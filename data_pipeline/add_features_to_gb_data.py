"""Attach KOI scalar features (orbital/transit/stellar measurements) to the
existing light-curve feature table, joining by kep_id — NO re-download needed.

We only add *physical measurements* (period, depth, radius, stellar params, ...).
Other columns that encodes the vetting verdict itself (koi_disposition, koi_score, 
koi_fpflag_*) would cause leakag. Those would let the model read the
answer key and produce fake ~100% accuracy.
"""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# the light-curve feature table to enrich (has kep_id, <lc features>, label)
IN_CSV  = DATA_DIR / "features_all_quarters_per_star.csv"
OUT_CSV = DATA_DIR / "features_with_koi.csv"

# physical measurements only
SAFE_KOI_FEATURES = [
    "koi_period",      # orbital period (days)
    "koi_duration",    # transit duration (hours)
    "koi_depth",       # transit depth (ppm)
    "koi_prad",        # planet radius (Earth radii)
    "koi_ror",         # planet/star radius ratio
    "koi_dor",         # planet-star distance / star radius (a/R*)
    "koi_sma",         # semi-major axis (AU)  -> "distance"
    "koi_impact",      # impact parameter
    "koi_model_snr",   # transit signal-to-noise
    "koi_insol",       # insolation flux (Earth flux)
    "koi_teq",         # equilibrium temperature (K)
    "koi_steff",       # stellar effective temperature (K)
    "koi_slogg",       # stellar surface gravity (log10 cgs)
    "koi_srad",        # stellar radius (solar radii)
]

# these ENCODE the label: never use them as features
LEAKAGE_COLUMNS = [
    "koi_disposition",   # the verdict itself
    "koi_pdisposition",  # pipeline verdict
    "koi_score",         # ~probability it's a planet
    "koi_fpflag_nt",     # false-positive reason flags
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
]


def build_koi_scalar_table():
    """One row per star with the SAFE scalar features.

    A star can have multiple KOIs; we take the MEDIAN of each measurement
    across its KOIs as a robust per-star summary. Missing values are filled
    with the column median so no rows are lost downstream."""
    koi = pd.read_csv(DATA_DIR / "koi_cumulative.csv", comment="#")

    keep = ["kepid"] + [c for c in SAFE_KOI_FEATURES if c in koi.columns]
    per_star = koi[keep].groupby("kepid", as_index=False).median()

    # fill any remaining NaNs with the column median (neutral, keeps all rows)
    feat_cols = [c for c in per_star.columns if c != "kepid"]
    per_star[feat_cols] = per_star[feat_cols].fillna(per_star[feat_cols].median())
    return per_star


def main():
    base = pd.read_csv(IN_CSV)               # kep_id, <lc features>, label
    koi  = build_koi_scalar_table()          # kepid, <scalar features>

    merged = base.merge(koi, how="left", left_on="kep_id", right_on="kepid")
    merged = merged.drop(columns=["kepid"])  # drop the duplicate join key

    # fill stars that had no KOI scalar match (rare) with column medians
    added = [c for c in SAFE_KOI_FEATURES if c in merged.columns]
    merged[added] = merged[added].fillna(merged[added].median())

    # keep label as the LAST column for a clean X / y split downstream
    cols = [c for c in merged.columns if c != "label"] + ["label"]
    merged = merged[cols]

    # LEAKAGE ASSERTION: no verdict column may ever be present
    leaked = [c for c in merged.columns if c in LEAKAGE_COLUMNS]
    assert not leaked, f"LEAKAGE: refusing to write label-encoding columns {leaked}"

    merged.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV.name}: {len(merged)} rows, {merged.shape[1]} columns")
    print(f"  light-curve features + {len(added)} KOI scalar features + label")
    print(f"  added KOI features: {added}")
    print(f"  leakage columns excluded: {LEAKAGE_COLUMNS}")


if __name__ == "__main__":
    main()
