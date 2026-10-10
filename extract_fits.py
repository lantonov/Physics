#!/usr/bin/env python3
"""
Extract low-ℓ TT power spectrum (ℓ, C_ℓ, σ_ℓ) from a Planck FITS file.

Dependencies:
    pip install astropy requests pandas numpy

Usage:
    # Local file:
    python extract_low_ell.py /path/to/COM_PowerSpect_CMB_R2.02.fits

    # Remote URL:
    python extract_low_ell.py https://irsa.ipac.caltech.edu/data/Planck/release_2/ancillary-data/COM_PowerSpect_CMB_R2.02.fits
"""

import sys
import io
import requests
import numpy as np
import pandas as pd

try:
    from astropy.io import fits
except ImportError:
    raise ImportError("Please install astropy: pip install astropy")

def load_fits(source):
    """
    Load a FITS file from a local path or URL and extract columns:
      - 'ELL'   : multipole ℓ
      - 'TT'    : C_ℓ^TT
      - 'TT_ERR': uncertainty σ_ℓ
    """
    # Open local or remote
    if source.startswith('http://') or source.startswith('https://'):
        resp = requests.get(source)
        resp.raise_for_status()
        hdul = fits.open(io.BytesIO(resp.content))
    else:
        hdul = fits.open(source)

    # Find the first binary table with the needed columns
    for hdu in hdul:
        if isinstance(hdu, fits.BinTableHDU):
            cols = hdu.columns.names
            if {'ELL','TT','TT_ERR'}.issubset(cols):
                data = hdu.data
                ell   = data['ELL']
                cl    = data['TT']
                sig   = data['TT_ERR']
                hdul.close()
                return pd.DataFrame({'ell': ell, 'Cl': cl, 'sigma': sig})

    hdul.close()
    raise ValueError("No suitable BINTABLEHDU with columns ELL, TT, TT_ERR found.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    source = sys.argv[1]
    df = load_fits(source)
    # Optionally, save to CSV
    df.to_csv('low_ell_spectrum.csv', index=False)
    print("Extracted low-ℓ spectrum:")
    print(df.head(30))
