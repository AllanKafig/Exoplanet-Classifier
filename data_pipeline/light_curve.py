"""Loads Kepler light curves from FITS files and extracts summary 
flux-statistic features used to train the classifier."""
import numpy as np

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

