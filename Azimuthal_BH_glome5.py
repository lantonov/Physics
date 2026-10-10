import glob
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import j1
import matplotlib.pyplot as plt

# -----------------------------
# Model definitions
# -----------------------------
def crescent_visibility_free(params_cres, u, v):
    I0, Rout, f, dalpha, dbeta = params_cres
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

def crescent_plus_modes_visibility(params, u, v, psi):
    I0, Rout, f, dalpha, dbeta = params[0:5]
    c1_re, c1_im, c2_re, c2_im = params[5:9]

    V_cres = crescent_visibility_free([I0, Rout, f, dalpha, dbeta], u, v)

    c1 = c1_re + 1j * c1_im
    c2 = c2_re + 1j * c2_im

    V_modes = c1 * np.exp(1j * psi) + c2 * np.exp(2j * psi)
    return V_cres + V_modes

def residuals_cres_modes(params, u, v, V, sigma, psi):
    V_model = crescent_plus_modes_visibility(params, u, v, psi)
    res = np.concatenate([
        (V_model.real - V.real) / sigma,
        (V_model.imag - V.imag) / sigma
    ])
    return np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)

# -----------------------------
# Multi-epoch loop
# -----------------------------
epoch_files = sorted(glob.glob("E:\\Documents1\\SR1_M87_2017_*_StokesI.csv"))

results = []  # list of dicts per epoch

for fname in epoch_files:
    print(f"\n=== Processing {fname} ===")
    df = pd.read_csv(fname, comment="#", header=None)
    df.columns = ["time", "T1", "T2", "u", "v", "amp", "phase_deg", "sigma"]

    u_raw = df["u"].values
    v_raw = df["v"].values

    # convert to Gλ
    u = u_raw * 1e-9
    v = v_raw * 1e-9
    rho = np.sqrt(u**2 + v**2)

    amp = df["amp"].values
    phase = np.deg2rad(df["phase_deg"].values)
    sigma = df["sigma"].values
    sigma[sigma <= 0] = np.median(sigma[sigma > 0])

    V = amp * np.exp(1j * phase)

    # baseline selection: use data-driven band (e.g. 0.5–6.0 Gλ)
    mask = (rho > 0.5) & (rho < 6.0)
    u = u[mask]
    v = v[mask]
    V = V[mask]
    sigma = sigma[mask]
    rho = rho[mask]

    if len(u) == 0:
        print("No visibilities in selected annulus, skipping epoch.")
        continue

    psi = np.arctan2(v, u)

    # initial guesses
    Rout_init = 20e-6 * np.pi / (180 * 3600)  # 20 μas
    I0_init   = np.median(np.abs(V))
    f_init    = 0.9
    dalpha_init = 0.0
    dbeta_init  = 0.0
    c1_init_re = 0.0
    c1_init_im = 0.0
    c2_init_re = 0.0
    c2_init_im = 0.0

    p0 = np.array([
        I0_init, Rout_init, f_init, dalpha_init, dbeta_init,
        c1_init_re, c1_init_im, c2_init_re, c2_init_im
    ])

    Rout_min = 5e-6  * np.pi / (180 * 3600)
    Rout_max = 50e-6 * np.pi / (180 * 3600)

    bounds_lower = [
        0.0,       Rout_min, 0.1,  -2e-9, -2e-9,
       -1.0,      -1.0,     -1.0, -1.0
    ]
    bounds_upper = [
        I0_init*10, Rout_max, 0.99,  2e-9,  2e-9,
        1.0,        1.0,      1.0,   1.0
    ]

    res = least_squares(
        residuals_cres_modes, p0,
        bounds=(bounds_lower, bounds_upper),
        args=(u, v, V, sigma, psi),
        verbose=1, max_nfev=1000
    )

    params = res.x
    I0, Rout, f, dalpha, dbeta = params[0:5]
    c1 = params[5] + 1j * params[6]
    c2 = params[7] + 1j * params[8]
    Rin = f * Rout

    abs_c1 = np.abs(c1)
    abs_c2 = np.abs(c2)
    arg_c1 = np.angle(c1, deg=True)
    arg_c2 = np.angle(c2, deg=True)

    results.append({
        "epoch": fname,
        "I0": I0,
        "Rout": Rout,
        "Rin": Rin,
        "f": f,
        "dalpha": dalpha,
        "dbeta": dbeta,
        "c1": c1,
        "c2": c2,
        "abs_c1": abs_c1,
        "abs_c2": abs_c2,
        "arg_c1": arg_c1,
        "arg_c2": arg_c2,
        "cost": res.cost,
        "optimality": res.optimality
    })

# convert to DataFrame for convenience
res_df = pd.DataFrame(results)
print("\n=== Summary over epochs ===")
print(res_df[["epoch", "I0", "Rout", "abs_c1", "abs_c2", "arg_c1", "arg_c2", "cost"]])
