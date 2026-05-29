import lightkurve as lk
from light_curve import load_kepler_light_curve, extract_features
  
# Download a known exoplanet's light curve (Kepler-10, the first rocky planet found)
search = lk.search_lightcurve("Kepler-10", mission="Kepler", cadence="long")
lc = search[0].download()

fits_path = "kepler10_sample.fits"
lc.to_fits(fits_path, overwrite=True)
print(f"Saved {fits_path}")   

df, kep_id = load_kepler_light_curve(fits_path)
print(f"Kepler ID: {kep_id}")
print(f"Loaded {len(df)} valid data points")
print(df.head())

features = extract_features(fits_path)
for key, value in features.items():
    if key != "flux_err":   # this is an array, skip printing
        print(f"  {key}: {value}")