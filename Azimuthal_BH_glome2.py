import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import j1

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------
csv_path = "C:\\Users\\lanto\\Documents\\SR1_M87_2017_095_hi_hops_netcal_StokesI.csv"
df = pd.read_csv(csv_path, comment="#", header=None)
df.columns = ["time", "T1", "T2", "u", "v", "amp", "phase_deg", "sigma"]

# Convert u,v from nanolambda → wavelengths (Gλ)
u = df["u"].values * 1e-9
v = df["v"].values * 1e-9

amp = df["amp"].values
phase = np.deg2rad(df["phase_deg"].values)
sigma = df["sigma"].values
sigma[sigma <= 0] = np.median(sigma[sigma > 0])

V = amp * np.exp(1j * phase)

# ------------------------------------------------------------
# 2. Baseline selection: keep only 0.5–3 Gλ
# ------------------------------------------------------------
rho = np.sqrt(u**2 + v**2)
mask = (rho > 0.5) & (rho < 3.0)

u = u[mask]
v = v[mask]
V = V[mask]
sigma = sigma[mask]
rho = rho[mask]

print(f"Selected {len(u)} visibilities in 0.5–3 Gλ annulus")

# ------------------------------------------------------------
# 3. Crescent model with FIXED R_out
# ------------------------------------------------------------
def crescent_visibility(params, u, v, Rout):
    """
    params = [I0, f, dalpha, dbeta]
    Rout   = fixed outer radius (rad)
    """
    I0, f, dalpha, dbeta = params
    Rin = f * Rout

    rho = np.sqrt(u**2 + v**2)
    phase_shift = np.exp(-2j * np.pi * (u * dalpha + v * dbeta))

    x_out = 2 * np.pi * Rout * rho
    x_in  = 2 * np.pi * Rin  * rho

    x_out = np.where(x_out == 0, 1e-20, x_out)
    x_in  = np.where(x_in  == 0, 1e-20, x_in)

    V_out = 2 * j1(x_out) / x_out
    V_in  = 2 * j1(x_in)  / x_in

    return I0 * (V_out - phase_shift * V_in)

# ------------------------------------------------------------
# 4. Residuals
# ------------------------------------------------------------
def residuals(params):
    V_model = crescent_visibility(params, u, v, Rout_fixed)
    res = np.concatenate([
        (V_model.real - V.real) / sigma,
        (V_model.imag - V.imag) / sigma
    ])
    return np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)

# ------------------------------------------------------------
# 5. Fix R_out to ~20 μas
# ------------------------------------------------------------
# 20 μas → radians
Rout_fixed = 20e-6 * np.pi / (180 * 3600)

# Initial guesses
I0_init = np.median(amp)
f_init = 0.5
dalpha_init = 0.0
dbeta_init = 0.0

p0 = np.array([I0_init, f_init, dalpha_init, dbeta_init])

bounds_lower = [0.0, 0.1, -2e-9, -2e-9]
bounds_upper = [I0_init*10, 0.99, 2e-9, 2e-9]

print("Running crescent fit with fixed R_out...")
res = least_squares(residuals, p0, bounds=(bounds_lower, bounds_upper),
                    verbose=2, max_nfev=200)

I0, f, dalpha, dbeta = res.x
Rin = f * Rout_fixed

print("\n=== Crescent Fit Results (Baseline-Filtered, Fixed R_out) ===")
print(f"I0      = {I0:.4f} Jy")
print(f"R_out   = {Rout_fixed:.3e} rad (fixed)")
print(f"R_in    = {Rin:.3e} rad (f = {f:.3f})")
print(f"offset  = ({dalpha:.3e}, {dbeta:.3e}) rad")
print(f"cost    = {res.cost:.3e}")
print(f"optimality = {res.optimality:.3e}")
