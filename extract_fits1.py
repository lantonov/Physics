import pandas as pd
import numpy as np
from astropy.io import fits

def extract_cl_from_dell(fits_path):
    with fits.open(fits_path) as hdul:
        # Find binary table
        for hdu in hdul:
            if isinstance(hdu, fits.BinTableHDU) and {'ELL', 'D_ELL', 'ERRUP', 'ERRDOWN'}.issubset(hdu.columns.names):
                data = hdu.data
                ell = data['ELL']
                dell = data['D_ELL']
                err_up = data['ERRUP']
                err_down = data['ERRDOWN']

                # Avoid division by zero at ℓ = 0 or 1
                mask = ell > 1
                ell = ell[mask]
                dell = dell[mask]
                err_up = err_up[mask]
                err_down = err_down[mask]

                # Compute C_ell and symmetric error approximation
                cl = 2 * np.pi * dell / (ell * (ell + 1))
                sigma = (err_up + err_down) / 2
                sigma_cl = 2 * np.pi * sigma / (ell * (ell + 1))

                df = pd.DataFrame({
                    'ell': ell,
                    'Cl': cl,
                    'sigma': sigma_cl
                })
                return df
    raise ValueError("Could not find suitable binary table in FITS.")

# Example usage
fits_file = "COM_PowerSpect_CMB_R2.02.fits"  # Replace with your path
df = extract_cl_from_dell(fits_file)
df.to_csv("extracted_cl.csv", index=False)
print(df.head(20))
