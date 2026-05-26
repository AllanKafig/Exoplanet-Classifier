import numpy as np
import pandas as pd
import lightkurve
from astropy.io import fits

def load_kepler_light_curve(fits_path):
    with fits.open(fits_path) as hudl:
        data = hudl[1].data

        time = np.array(data["TIME"], dtype = float)
        flux = np.array(data["PDCSAP_FLUX"], dtype = float)

        df = pd.DataFrame({
            "TIME": np.asarray(data["TIME"], dtype = float),
            "PDCSAP_FLUX": np.asarray(data["PDCSAP_FLUX"], dtype = float),
            "PDCSAP_FLUX_ERR": np.asarray(data["PDCSAP_FLUX_ERR"], dtype = float),
            "SAP_QUALITY": np.asarray(data["SAP_QUALITY"], dtype = float),
        })

        kep_id = hudl[0].header.get("KEPLERID", None) 

    df = df[
        (df["SAP_QUALITY"] == 0) &
        (df["TIME"].notna()) & 
        (df["PDCSAP_FLUX"].notna()) &
        (df["PDCSAP_FLUX_ERR"]).notna()
    ]

    return df, kep_id

def extract_features(fits_path):
    df, kep_id = load_kepler_light_curve(fits_path)
    
    flux = df["PDCSAP_FLUX"].values
    time = df["TIME"].values
    flux_err = df["PDCSAP_FLUX_ERR"].values

    if len(flux) == 0:
        return None

    features = {
        "kep_id": kep_id,
        "num_points": len(df),
        "time_span": time.max() - time.min(),
        "flux_err": flux_err,

        "mean_flux": np.mean(flux),
        "median_flux": np.median(flux),
        "std_flux": np.std(flux),
        "max_flux": np.max(flux),
        "min_flux": np.min(flux),
        "range_flux": np.max(flux) - np.min(flux)
    }    

    return features

