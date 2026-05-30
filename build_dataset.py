"""
Reads the KOI table to label stars, downloads one quarter of Kepler 
light curve data per star, extracts features, and saves a dataset 
to CSV.
"""

import pandas as pd
import lightkurve as lk
from light_curve import extract_features

# Equal stars per class (planet / no-planet) so the model does not bias toward the majority.
# The KOI table has ~9,564 rows, but fewer unique stars after dedup () + class split.
# 3500 still takes some time to download but we can raise it if needed.
STARS_PER_GROUP = 3500   
OUTPUT_FILE = "features.csv"

def get_star_labels():
    """Read the KOI table and give each star a label."""
    koi = pd.read_csv("koi_cumulative.csv", comment="#")
 
    koi["label"] = koi["koi_disposition"].map({
        "CONFIRMED": 1,
        "CANDIDATE": 1,
        "FALSE POSITIVE": 0,
    })
    koi = koi.dropna(subset=["label"])
 
    # one row per star (if a star has any planet, label it 1)
    labels = koi.groupby("kepid")["label"].max().reset_index()
    labels["label"] = labels["label"].astype(int)
    return labels
 
 
def pick_balanced_stars(labels):
    """Choose an equal number of planet and no-planet stars."""
    planets = labels[labels["label"] == 1].sample(STARS_PER_GROUP, random_state=42)
    no_planets = labels[labels["label"] == 0].sample(STARS_PER_GROUP, random_state=42)
    return pd.concat([planets, no_planets])
 
 
def main():
    labels = get_star_labels()
    stars = pick_balanced_stars(labels)
    print(f"processing {len(stars)} stars")
 
    rows = []
    for count, (_, star) in enumerate(stars.iterrows(), start=1):
        kepid = int(star["kepid"])
        label = int(star["label"])
 
        try:
            # download one quarter of data for this star
            result = lk.search_lightcurve(f"KIC {kepid}", mission="Kepler", cadence="long")
            lc = result[0].download()
    
            # turn the light curve into features, then add the label
            features = extract_features(lc, kep_id=kepid)
            features["label"] = label # Ctick the label onto the features dict
            rows.append(features)
            print(f"  [{count}/{len(stars)}] OK   KIC {kepid}")
 
        except Exception as error:
            print(f"  [{count}/{len(stars)}] skip KIC {kepid}  ({error})")
 
    pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(rows)} stars to {OUTPUT_FILE}")
 
 
if __name__ == "__main__":
    main()