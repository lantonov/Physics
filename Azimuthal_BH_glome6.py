import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.special import j1

def crescent_visibility(params_cres, u, v):
    I0, Rout, f, dalpha, dbeta = params_cres
    Rin = f * Rout

    rho = np.sqrt(u**2 + v**2)
    phase_shift = np.exp(-2j * np.pi * (u*dalpha + v*dbeta))

    x_out = 2*np.pi*Rout*rho
    x_in  = 2*np.pi*Rin*rho

    x_out = np.where(x_out == 0, 1e-20, x_out)
    x_in  = np.where(x_in  == 0, 1e-20, x_in)

    V_out = 2*j1(x_out)/x_out
    V_in  = 2*j1(x_in)/x_in

    return I0*(V_out - phase_shift*V_in)


def crescent_plus_modes(params, u, v, psi):
    I0, Rout, f, dalpha, dbeta = params[:5]
    c1 = params[5] + 1j*params[6]
    c2 = params[7] + 1j*params[8]

    V_cres = crescent_visibility([I0, Rout, f, dalpha, dbeta], u, v)
    V_modes = c1*np.exp(1j*psi) + c2*np.exp(2j*psi)

    return V_cres + V_modes


def residuals(params, u, v, V, sigma, psi):
    V_model = crescent_plus_modes(params, u, v, psi)
    res = np.concatenate([
        (V_model.real - V.real)/sigma,
        (V_model.imag - V.imag)/sigma
    ])
    return np.nan_to_num(res)

def process_epoch(fname):
    print(f"\n=== Processing {fname} ===")

    df = pd.read_csv(fname, comment="#", header=None)
    df.columns = ["time","T1","T2","u","v","amp","phase_deg","sigma"]

    # convert to Gλ
    u = df["u"].values * 1e-9
    v = df["v"].values * 1e-9
    rho = np.sqrt(u**2 + v**2)

    amp = df["amp"].values
    phase = np.deg2rad(df["phase_deg"].values)
    sigma = df["sigma"].values
    sigma[sigma <= 0] = np.median(sigma[sigma > 0])
    V = amp*np.exp(1j*phase)

    # automatic baseline selection
    rho_min = np.quantile(rho, 0.20)
    rho_max = np.quantile(rho, 0.95)
    mask = (rho > rho_min) & (rho < rho_max)

    if mask.sum() < 50:
        print("Too few visibilities, skipping.")
        return None

    u = u[mask]
    v = v[mask]
    V = V[mask]
    sigma = sigma[mask]
    psi = np.arctan2(v, u)

    # initial guesses
    Rout_init = 20e-6*np.pi/(180*3600)
    p0 = np.array([
        np.median(np.abs(V)),  # I0
        Rout_init,             # Rout
        0.8,                   # f
        0.0, 0.0,              # offsets
        0.0, 0.0, 0.0, 0.0     # c1, c2
    ])

    Rout_min = 5e-6*np.pi/(180*3600)
    Rout_max = 50e-6*np.pi/(180*3600)

    bounds_lower = [0, Rout_min, 0.1, -2e-9, -2e-9, -1, -1, -1, -1]
    bounds_upper = [10, Rout_max, 0.99, 2e-9, 2e-9, 1, 1, 1, 1]

    res = least_squares(residuals, p0,
                        bounds=(bounds_lower, bounds_upper),
                        args=(u, v, V, sigma, psi),
                        max_nfev=800, verbose=0)

    params = res.x
    I0, Rout, f, dalpha, dbeta = params[:5]
    c1 = params[5] + 1j*params[6]
    c2 = params[7] + 1j*params[8]

    return {
        "epoch": fname,
        "I0": I0,
        "Rout": Rout,
        "Rin": f*Rout,
        "f": f,
        "dalpha": dalpha,
        "dbeta": dbeta,
        "c1": c1,
        "c2": c2,
        "abs_c1": np.abs(c1),
        "abs_c2": np.abs(c2),
        "arg_c1": np.angle(c1, deg=True),
        "arg_c2": np.angle(c2, deg=True),
        "cost": res.cost,
        "optimality": res.optimality,
        "Nvis": mask.sum()
    }

epoch_files = sorted(glob.glob("C:\\Users\\lanto\\Documents\\L2V1_M87_2018_111_b1_*_netcal_10s_StokesI.csv"))

results = []
for fname in epoch_files:
    out = process_epoch(fname)
    if out is not None:
        results.append(out)

res_df = pd.DataFrame(results)
print("\n=== Final DataFrame Columns ===")
print(res_df.columns.tolist())
print(res_df)

if len(res_df) > 0:
    epochs = np.arange(len(res_df))

    plt.figure(figsize=(10,6))
    plt.plot(epochs, res_df["abs_c1"], "o-", label="|c1|")
    plt.plot(epochs, res_df["abs_c2"], "o-", label="|c2|")
    plt.legend()
    plt.title("Mode amplitudes across epochs")
    plt.xlabel("Epoch index")
    plt.ylabel("Amplitude")
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(epochs, res_df["arg_c1"], "o-", label="arg(c1)")
    plt.plot(epochs, res_df["arg_c2"], "o-", label="arg(c2)")
    plt.legend()
    plt.title("Mode phases across epochs")
    plt.xlabel("Epoch index")
    plt.ylabel("Phase (deg)")
    plt.show()

def stability(x):
    return 1/(1 + np.std(x))

def dominance(c2, c1):
    r = np.mean(c2/(c1+1e-10))
    return min(r/3, 1)

if len(res_df) > 0:
    S_c1 = stability(res_df["abs_c1"])
    S_c2 = dominance(res_df["abs_c2"], res_df["abs_c1"])
    S_phi = stability(res_df["arg_c2"])
    S_cent = stability(np.sqrt(res_df["dalpha"]**2 + res_df["dbeta"]**2))

    S_glome = (S_c1 + S_c2 + S_phi + S_cent)/4

    print("\n=== Glome-likeness Score ===")
    print(f"S_glome = {S_glome:.3f}")
    print(f"  c1 stability   = {S_c1:.3f}")
    print(f"  c2 dominance   = {S_c2:.3f}")
    print(f"  c2 phase stab  = {S_phi:.3f}")
    print(f"  centroid stab  = {S_cent:.3f}")
