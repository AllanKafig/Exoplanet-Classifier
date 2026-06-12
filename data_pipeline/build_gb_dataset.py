"""Builds the training set: labels Kepler stars from the KOI catalog,
downloads each star's light curve, and writes extracted features to data/features.csv.

Uses every available CONFIRMED and FALSE POSITIVE star (no per-class capping).
The resulting dataset is naturally imbalanced — FALSE POSITIVE is typically the
larger class. Downstream train/test splits should use stratify=y to preserve
that ratio, and metrics should favor precision/recall/F1/ROC-AUC over raw
accuracy."""
import re
import shutil
import socket
from pathlib import Path
import pandas as pd
import numpy as np
import lightkurve as lk

#resolve data paths relative to the repo root so the script works from any cwd
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

OUTPUT_FILE = DATA_DIR / "koi_cumulative.csv"
SAMPLE_LIMIT = None  # set to a small int (e.g. 50) for a quick sanity test; None = use all stars
SAVE_EVERY = 25      # checkpoint to disk every N completed stars
DOWNLOAD_TIMEOUT = 300  # seconds — caps any single network call so a hung connection can't stall forever
# If this many network errors happen back-to-back, the connection is almost
# certainly down — stop and save so the run can be resumed later instead of
# churning through the rest of the list marking everything "skip".
MAX_CONSECUTIVE_NET_ERRORS = 40

# Force a process-wide socket timeout so a stuck MAST connection can't hang the run.
socket.setdefaulttimeout(DOWNLOAD_TIMEOUT)

def extract_features(lc, kep_id=None):
    """Extract features directly from a lightkurve LightCurve object."""
    time = np.asarray(lc.time.value, dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)
    
    good = ~np.isnan(time) & ~np.isnan(flux)
    time = time[good]
    flux = flux[good]

    if len(flux) == 0:
        return None

    features = {
        "kep_id": kep_id,
        "num_points": len(flux),
        "time_span": time.max() - time.min(),
        "mean_flux": np.mean(flux),
        "median_flux": np.median(flux),
        "std_flux": np.std(flux),
        "max_flux": np.max(flux),
        "min_flux": np.min(flux),
        "range_flux": np.max(flux) - np.min(flux)
    }    

    return features

def get_star_labels():
    """
    Read the KOI table and assign a ground-truth label for each star.

    A star is labeled 1 (hosts an exoplanet) if any of its KOIs are CONFIRMED,
    and 0 (no exoplanet) if all are FALSE POSITIVE. CANDIDATE dispositions
    are excluded because they are unverified and would add label noise.
    
    Returns:
        DataFrame with columns ["kepid", "label"] — one row per star, where label is 
        1 or 0.
    """
    koi = pd.read_csv(DATA_DIR / "koi_cumulative.csv", comment="#")
 
    # CANDIDATE dispositions are intentionally omitted (unmapped → NaN → dropped)
    # because they are unverified and add label noise.
    koi["label"] = koi["koi_disposition"].map({
        "CONFIRMED": 1,
        "FALSE POSITIVE": 0,
    })
    koi = koi.dropna(subset=["label"])
 
    #one row per star (if a star has any planet, label it 1)
    labels = koi.groupby("kepid")["label"].max().reset_index()
    labels["label"] = labels["label"].astype(int)
    return labels
 
 
def pick_all_stars(labels, limit=None):
    """
    Use every CONFIRMED and FALSE POSITIVE star, preserving the natural class
    ratio (imbalanced — FP is typically the larger class).

    Args:
        labels: DataFrame with a "label" column (1 = host, 0 = non-host).
        limit: Optional cap on total rows for sanity testing. If set, the
            downsample is stratified so the natural class ratio is preserved.
            None = use all stars.

    Returns:
        DataFrame containing the selected rows, shuffled with a fixed seed
        for reproducibility.
    """
    if limit is None or limit >= len(labels):
        selected = labels.sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        # stratified subsample preserves the host/non-host ratio
        frac = limit / len(labels)
        selected = (labels.groupby("label", group_keys=False)
                          .apply(lambda g: g.sample(frac=frac, random_state=42))
                          .sample(frac=1, random_state=42)
                          .reset_index(drop=True))

    n_pos = int((selected["label"] == 1).sum())
    n_neg = int((selected["label"] == 0).sum())
    print(f"Selected {len(selected)} stars: {n_pos} confirmed (host), "
          f"{n_neg} false positive (non-host)")
    return selected
 
 
def save_rows(rows):
    """Write rows so far to disk — used for crash-safe checkpointing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)


def _looks_like_network_error(error):
    """True if the exception is a connectivity failure (DNS, refused, dropped,
    timeout) rather than a 'this star has no data' problem."""
    text = str(error)
    needles = ("Failed to resolve", "Max retries", "Connection aborted",
               "RemoteDisconnected", "NameResolutionError", "timed out",
               "Temporary failure in name resolution", "Connection reset")
    return any(n in text for n in needles)


def _corrupt_cache_dir(error):
    """If `error` is lightkurve failing to read a cached FITS product (a
    partial/corrupt file left by an interrupted download), return the cache
    directory to delete so it can be re-downloaded fresh; else None."""
    if "Error in reading Data product" not in str(error):
        return None
    match = re.search(r"(/\S+\.fits)", str(error))
    if not match:
        return None
    fits_path = Path(match.group(1))
    return fits_path.parent if fits_path.exists() else None


def download_lightcurve(kepid):
    """Download & stitch all long-cadence quarters for one star. If a cached
    product is corrupt (interrupted download), delete it and re-download once."""
    for attempt in range(2):
        try:
            # author="Kepler" restricts to the official Kepler pipeline products
            # (kplr*_llc.fits). Without it, the search also returns community
            # HLSP products (e.g. IRIS "stitched" files) that this astropy/
            # lightkurve can't parse, and one unreadable file fails the star.
            result = lk.search_lightcurve(f"KIC {kepid}", mission="Kepler",
                                          exptime="long", author="Kepler")
            if len(result) == 0:
                return None  # no official Kepler long-cadence data for this star
            return result.download_all().stitch().remove_nans()
        except Exception as error:
            bad_dir = _corrupt_cache_dir(error)
            if bad_dir is not None and attempt == 0:
                shutil.rmtree(bad_dir, ignore_errors=True)
                print(f"      removed corrupt cache {bad_dir.name} — re-downloading")
                continue
            raise


def main():
    labels = get_star_labels()
    stars = pick_all_stars(labels, limit=SAMPLE_LIMIT)
    print(f"processing {len(stars)} stars "
          f"(socket timeout {DOWNLOAD_TIMEOUT}s, checkpoint every {SAVE_EVERY})")

    # --- resume support -------------------------------------------------
    # Load stars already saved by a previous run so we (a) APPEND to the
    # existing CSV instead of overwriting it, and (b) skip work already done.
    # Stars that previously failed (e.g. the network outage) are NOT in the
    # CSV, so they are retried automatically. The star ORDER is deterministic
    # (random_state=42), so resuming picks up exactly where we left off.
    rows = []
    done = set()
    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE)
        rows = existing.to_dict("records")
        done = set(existing["kep_id"].astype(int))
        print(f"resuming: {len(done)} stars already saved in {OUTPUT_FILE.name} "
              f"— these will be skipped")

    consecutive_net_errors = 0
    for count, (_, star) in enumerate(stars.iterrows(), start=1):
        kepid = int(star["kepid"])
        label = int(star["label"])

        if kepid in done:
            continue  # already downloaded in a previous run

        try:
            # download ALL available long-cadence quarters and stitch them
            # into one continuous light curve (~4 years of data per star);
            # self-heals corrupt cache files left by interrupted downloads
            lc = download_lightcurve(kepid)
            if lc is None:
                print(f"  [{count}/{len(stars)}] skip KIC {kepid}  (no Kepler long-cadence data)")
                consecutive_net_errors = 0
                continue

            # extract features and attach the label
            features = extract_features(lc, kep_id=kepid)
            features["label"] = label
            rows.append(features)
            done.add(kepid)
            consecutive_net_errors = 0
            print(f"  [{count}/{len(stars)}] OK   KIC {kepid}")

            # crash-safe checkpoint
            if len(rows) % SAVE_EVERY == 0:
                save_rows(rows)
                print(f"  checkpoint: saved {len(rows)} rows")

        except Exception as error:
            print(f"  [{count}/{len(stars)}] skip KIC {kepid}  ({error})")
            if _looks_like_network_error(error):
                consecutive_net_errors += 1
                if consecutive_net_errors >= MAX_CONSECUTIVE_NET_ERRORS:
                    print(f"\n{consecutive_net_errors} network errors in a row — the "
                          f"connection looks down. Saving progress and stopping; "
                          f"re-run this script once you're back online to resume.")
                    break
            else:
                consecutive_net_errors = 0  # a 'no data' skip isn't a connectivity problem

    save_rows(rows)
    print(f"\nSaved {len(rows)} stars to {OUTPUT_FILE}")
 
 
if __name__ == "__main__":
    main()