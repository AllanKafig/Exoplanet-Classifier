import sys
import pandas as pd
import numpy as np
import lightkurve as lk
from scipy.stats import binned_statistic

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))   #so that we can import from light_curve.py and build_dataset.py
from src.build_dataset import pick_balanced_stars

#resolve data paths relative to the repo root so the script works from any cwd                                
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

STARS_PER_GROUP = 100
OUTPUT_FILE = DATA_DIR / "1000koi_allquarters.csv"
SAVE_EVERY = 5

def get_star_labels():
    """
    This is similar to the get_star_labels() in build_dataset.py, except it does not include stars labeled as "CANDIDATE"
    
    Read the KOI table and assigns a ground truth label for each star.

    A star is labeled 1 (hosts an exoplanet) if any of its KOIs are CONFIRMED
    or CANDIDATE, and 0 (no exoplanet) if all are FALSE POSITIVE. Rows with
    unmapped dispositions are dropped before aggregation.
    
    Returns:
        DataFrame with columns ["kepid", "label"] — one row per star, where label is 
        1 or 0.
    """

    koi = pd.read_csv(DATA_DIR / "koi_cumulative.csv", comment="#")

    koi.drop(koi[koi["koi_disposition"] == "CANDIDATE"].index, inplace=True) #drop candidates so we only have confirmed planets vs false positives
 
    koi["label"] = koi["koi_disposition"].map({
        "CONFIRMED": 1,
        "FALSE POSITIVE": 0,
    })
    koi = koi.dropna(subset=["label"])
 
    # one row per star (if a star has any planet, label it 1)
    labels = koi.groupby("kepid")["label"].max().reset_index()
    labels["label"] = labels["label"].astype(int)

    return labels

def getStuff():
    """Get the period and t0 information for all the stars in the KOI table (we will need this to fold the light curves)
     Returns:
        DataFrame with columns ["kepid", "koi_period", "koi_time0bk"] — one row per star, where kepid is the Kepler ID, 
        koi_period is the known period of the planet (from the KOI table), and koi_time0bk is the known time of first transit (from the KOI table).
    """
    koi = pd.read_csv(DATA_DIR / "koi_cumulative.csv", comment="#")
    koi.drop(koi[koi["koi_disposition"] == "CANDIDATE"].index, inplace=True) #drop candidates so we only have confirmed planets vs false positives

    rows = []
    for _, row in koi.iterrows():
        rows.append({
            "kepid": row["kepid"],
            "koi_period": row["koi_period"],
            "koi_time0bk": row["koi_time0bk"],
        })

    return pd.DataFrame(rows)

def get_lc_features(lc, kep_id, period, t0):
    """Turn a light curve into features. We will fold and bin the light curve to get a fixed-length feature vector.
     Args:
        lc: a lightkurve LightCurve object
        kep_id: the Kepler ID of the star (for tracking/debugging purposes)
        period: the known period of the planet (from the KOI table)
        t0: the known time of first transit (from the KOI table)
    Returns:
        A dict with keys "kep_id", "flux_0", "flux_1", ..., "flux_{n_bins-1}", where the flux_i values are the median flux in each phase bin.
    """
    time = np.asarray(lc.time.value)
    flux = np.asarray(lc.flux.value)

    flux = flux / np.nanmedian(flux) - 1.0 #normalize the flux values to be between 0 and 1
    flux = fold_and_bin(time, flux, period, t0) #fold and bin the light curve to get a fixed-length feature vector

    row = {"kep_id": kep_id}
    for i, f in enumerate(flux):
        row[f"flux_{i}"] = f
    return row

def fold_and_bin(time, flux, period, t0, n_bins=2000, n_dips=2):
    """
    Convert a variable-length light curve into a fixed-length, transit-aligned vector.

    Two steps:
      FOLD — collapse every orbit onto one using phase = ((time - t0) % period) / period.
             Transits recur once per period, they all map to the same phase and
             STACK into the same bin.
      BIN  — split the phase axis into n_bins equal slots over a fixed range (-0.5, 0.5)
             and take the MEDIAN flux per slot. The median averages down noise. Empty bins 
             are filled by interpolating from neighbors.

    Args:
        time: 1D array of time values for the light curve
        flux: 1D array of flux values for the light curve (same length as time)
        period: the known period of the planet (from the KOI table)
        t0: the known time of first transit (from the KOI table)
        n_bins: how many bins to use when folding the light curve 
        n_dips: how many dips to include in the folded curve
    Returns:
        1D array of length n_bins, the folded and binned light curve.
    """
    #'fold' the light curve by the known period
    phase = ((time - t0 + 0.5*(n_dips*period)) % (n_dips*period)) / (n_dips*period) - 0.5

    #place the folded curve into bins
    #take the median flux value in each bin (this is done so that if you have a noisy value in one bin, it won't affect the overall value for that bin)
    binned, _, _ = binned_statistic(phase, flux, statistic="median", bins=n_bins, range=(-0.5, 0.5))

    #if any bins are empty (NaN), fill them in by interpolating between the nearest non-empty bins 
    idx = np.arange(n_bins)
    nan = np.isnan(binned)
    if np.any(nan):
        binned[nan] = np.interp(idx[nan], idx[~nan], binned[~nan])

    return binned

def main():
    labels = get_star_labels() #get the labels for all the stars in the KOI table
    stars = pick_balanced_stars(labels, STARS_PER_GROUP) #get a balanced sample of stars with and without exoplanets
    koi_info = getStuff() #get the period and t0 information for all the stars in the KOI table (we will need this to fold the light curves)

    rows = []
    for count, (_, star) in enumerate(stars.iterrows(), start=1):
        kepid = int(star["kepid"])
        label = int(star["label"])

        try:
            match = koi_info[koi_info["kepid"] == kepid]
            period = float(match["koi_period"].iloc[0]) #if a star has multiple KOIs, just take the first one (for simplicity)
            t0 = float(match["koi_time0bk"].iloc[0])

            # download all quarters of data for this star
            result = lk.search_lightcurve(f"KIC {kepid}", mission="Kepler", exptime="long")
            lc = result.download_all().stitch().remove_nans()
    
            # turn the light curve into features, then add the label
            features = get_lc_features(lc, kep_id=kepid, period=period, t0=t0)
            # Stick the label onto the features dict
            features["label"] = label 
            rows.append(features)
            print(f"  [{count}/{len(stars)}] OK   KIC {kepid}")

            # Save to file every few rows to save data in case the program terminates early
            if len(rows) % SAVE_EVERY == 0:
                save_rows(rows)
                print(f"checkpoint: saved {len(rows)} rows")
 
        except Exception as error:
            print(f"  [{count}/{len(stars)}] skip KIC {kepid}  ({error})")
 
    pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(rows)} stars to {OUTPUT_FILE}")

def save_rows(rows):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    main()

