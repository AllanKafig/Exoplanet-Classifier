import pandas as pd
import lightkurve as lk
import numpy as np

path = "../koi_cumulative.csv"

def construct_dataset(num_objects: int, num_bins = 2000):
    df = pd.read_csv(path, comment = '#')
    confirmed = df[df["koi_disposition"].str.contains("CONFIRMED", na = False)].sample(n = num_objects, random_state = 0)
    false_positives = df[df["koi_disposition"].str.contains("FALSE POSITIVE", na = False)].sample(n = num_objects, random_state = 0)

    confirmed_set = [(row["kepoi_name"], get_folded_lightcurve(row, num_bins)) for i, row in confirmed.iterrows()]
    fp_set = [(row["kepoi_name"], get_folded_lightcurve(row, num_bins)) for i, row in false_positives.iterrows()]

    return confirmed_set, fp_set

def get_folded_lightcurve(row, num_bins: int = 2000):
    kepid = int(row["kepid"])
    period = float(row["koi_period"])
    epoch = float(row["koi_time0bk"])

    light_curve = lk.search_lightcurve(kepid, mission = "Kepler").download_all().stitch().remove_nans()

    folded = light_curve.fold(period = period, epoch_time = epoch, normalize_phase = True)
    phase = folded.phase.value
    flux = folded.flux.value

    edges = np.linspace(-0.5, 0.5, num_bins + 1)
    centers = (edges[:-1]+ edges[1:]) / 2
    binned_flux = np.full(num_bins, np.nan)
    bin_indices = np.digitize(phase, edges) - 1

    for i in range(num_bins):
        values = flux[bin_indices == i]

        if len(values) > 0:
            binned_flux[i] = np.nanmedian(values)

    valid = ~np.isnan(binned_flux)
    binned_flux = np.interp(centers, centers[valid], binned_flux[valid])

    return pd.DataFrame({"phase": centers, "flux": binned_flux})

if __name__ == "__main__":
    confirmed_set, fp_set = construct_dataset(
        num_objects=500,
        num_bins=2000,
    )

    print("Confirmed objects:", len(confirmed_set))
    print("False positives:", len(fp_set))

    print("Example confirmed KOI:", confirmed_set[0][0])
    print(confirmed_set[0][1].head())