import numpy as np
import pandas as pd
from astropy.io import fits

def load_kepler_light_curve(fits_path):
    """
    Load a Kelper light curve from a FITS file.
    """
    with fits.open(fits_path) as hdul:
        data = hdul[1].data
        
        flux_col = "PDCSAP_FLUX"
        err_col = "PDCSAP_FLUX_ERR" 
        quality_col = "SAP_QUALITY" 

        df = pd.DataFrame({
            "TIME": np.asarray(data["TIME"], dtype=float),
            "FLUX": np.asarray(data[flux_col], dtype=float),
            "FLUX_ERR": np.asarray(data[err_col], dtype=float),
            "QUALITY": np.asarray(data[quality_col], dtype=float),
        })
    
        kep_id = hdul[0].header.get("KEPLERID", None) 
    return df, kep_id

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

