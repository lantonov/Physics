#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

# -------------------------
# Constants and Conversions
# -------------------------
G = 6.67430e-11               # Gravitational constant in m^3 kg^-1 s^-2
Msun = 1.989e30               # Solar mass in kg
kpc = 3.086e19                # 1 kiloparsec in meters

# -------------------------
# Observational Data (NGC 3198)
# -------------------------
# Approximate rotation curve data: radius in kpc and velocity in km/s.
r_obs_kpc = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
V_obs_kms = np.array([60, 85, 105, 120, 135, 145, 150, 155, 158, 160, 161, 162, 163, 163, 163])
V_err_kms = 5.0 * np.ones_like(V_obs_kms)  # measurement uncertainties (km/s)

# Convert data to SI units:
r_obs = r_obs_kpc * kpc   # in meters
V_obs = V_obs_kms * 1e3   # in m/s
V_err = V_err_kms * 1e3   # in m/s

# -------------------------
# Model Functions
# -------------------------
def M_disk(r, M_total, Rd):
    """
    Exponential disk cumulative mass:
      M(r) = M_total * [1 - (1 + r/Rd)*exp(-r/Rd)]
    r : radius (m)
    M_total : total luminous mass (kg)
    Rd : scale length (m)
    """
    return M_total * (1 - (1 + r/Rd) * np.exp(-r/Rd))

def V_model_linear(r, M_total, Rd, f0, f1):
    """
    Rotation velocity with an effective boost factor that varies linearly with radius.
    f_eff(r) = f0 + f1*(r/kpc)   [so that r/kpc is dimensionless]
    V(r) = sqrt( f_eff(r) * G * M_disk(r) / r )
    """
    f_eff = f0 + f1*(r/kpc)
    # To ensure we don't take sqrt of a negative number:
    # (if any f_eff becomes negative, return a very low value)
    if np.any(f_eff <= 0):
        return np.full_like(r, 1e-10)
    return np.sqrt(f_eff * G * M_disk(r, M_total, Rd) / r)

# -------------------------
# Log-Prior Function
# -------------------------
def log_prior(theta):
    """
    theta = (M_total, Rd, f0, f1, sigma_int)
      M_total: total luminous mass in kg
      Rd: disk scale length in m
      f0: boost intercept (dimensionless)
      f1: boost slope (dimensionless per kpc)
      sigma_int: additional scatter (m/s)
    Uniform bounds:
      M_total: 1e10 Msun to 1e12 Msun
      Rd: 0.5 kpc to 30 kpc
      f0: 1.0 to 10.0
      f1: -5.0 to 5.0
      sigma_int: 0 to 20 km/s
    Plus informative Gaussian priors:
      - For log10(M_total): mean = 41.3, sigma = 0.3
      - For Rd (in kpc): mean = 3.0, sigma = 0.5
      - For the effective boost at 5 kpc: f0 + 5*f1 ~ 3.45, sigma = 1.0
    """
    M_total, Rd, f0, f1, sigma_int = theta

    if not (1e10 * Msun < M_total < 1e12 * Msun):
        return -np.inf
    if not (0.5 * kpc < Rd < 30 * kpc):
        return -np.inf
    if not (1.0 < f0 < 10.0):
        return -np.inf
    if not (-5.0 < f1 < 5.0):
        return -np.inf
    if not (0 <= sigma_int < 20e3):
        return -np.inf

    # Informative priors:
    logM = np.log10(M_total)
    lnP_M = -0.5 * ((logM - 41.3) / 0.3) ** 2

    Rd_kpc = Rd / kpc
    lnP_Rd = -0.5 * ((Rd_kpc - 3.0) / 0.5) ** 2

    # For boost: expect f_eff at 5 kpc to be about 3.45.
    f_eff_5 = f0 + 5.0 * f1
    lnP_boost = -0.5 * ((f_eff_5 - 3.45) / 1.0) ** 2

    return lnP_M + lnP_Rd + lnP_boost

# -------------------------
# Log-Likelihood and Log-Posterior Functions
# -------------------------
def log_likelihood(theta, r, V_obs, V_err):
    M_total, Rd, f0, f1, sigma_int = theta
    model = V_model_linear(r, M_total, Rd, f0, f1)
    sigma_tot = np.sqrt(V_err**2 + sigma_int**2)
    return -0.5 * np.sum(((V_obs - model) / sigma_tot) ** 2 + np.log(2 * np.pi * sigma_tot ** 2))

def log_posterior(theta, r, V_obs, V_err):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, r, V_obs, V_err)

# -------------------------
# MCMC Setup
# -------------------------
ndim = 5
nwalkers = 70

# Parameter bounds for initialization:
M_total_min, M_total_max = 1e10 * Msun, 1e12 * Msun
Rd_min, Rd_max = 0.5 * kpc, 30 * kpc
f0_min, f0_max = 1.0, 10.0
f1_min, f1_max = -5.0, 5.0
sigma_int_min, sigma_int_max = 0, 20e3

# Initialize walkers uniformly over allowed ranges:
pos = np.empty((nwalkers, ndim))
for i in range(nwalkers):
    M_total_i = np.random.uniform(M_total_min, M_total_max)
    Rd_i = np.random.uniform(Rd_min, Rd_max)
    f0_i = np.random.uniform(f0_min, f0_max)
    f1_i = np.random.uniform(f1_min, f1_max)
    sigma_int_i = np.random.uniform(sigma_int_min, sigma_int_max)
    pos[i] = [M_total_i, Rd_i, f0_i, f1_i, sigma_int_i]

# -------------------------
# Run the MCMC
# -------------------------
nsteps = 6000
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(r_obs, V_obs, V_err))
print("Running MCMC for the linear boost model...")
sampler.run_mcmc(pos, nsteps, progress=True)

burnin = nsteps // 2
samples = sampler.get_chain(discard=burnin, flat=True)

# -------------------------
# Analyze the Chain
# -------------------------
def median_and_errors(param_samples):
    mcmc = np.percentile(param_samples, [16, 50, 84])
    median = mcmc[1]
    error_plus = mcmc[2] - median
    error_minus = median - mcmc[0]
    return median, error_plus, error_minus

M_total_med, M_total_err_plus, M_total_err_minus = median_and_errors(samples[:, 0])
Rd_med, Rd_err_plus, Rd_err_minus = median_and_errors(samples[:, 1])
f0_med, f0_err_plus, f0_err_minus = median_and_errors(samples[:, 2])
f1_med, f1_err_plus, f1_err_minus = median_and_errors(samples[:, 3])
sigma_int_med, sigma_int_err_plus, sigma_int_err_minus = median_and_errors(samples[:, 4])

print("MCMC Results (Linear Boost Model):")
print(f"  M_total = {M_total_med:.3e} +{M_total_err_plus:.3e} -{M_total_err_minus:.3e} kg")
print(f"            ({M_total_med/Msun:.3e} M_sun)")
print(f"  Rd      = {Rd_med/kpc:.3f} +{Rd_err_plus/kpc:.3f} -{Rd_err_minus/kpc:.3f} kpc")
print(f"  f0      = {f0_med:.3f} +{f0_err_plus:.3f} -{f0_err_minus:.3f}")
print(f"  f1      = {f1_med:.3f} +{f1_err_plus:.3f} -{f1_err_minus:.3f}")
print(f"  sigma_int = {sigma_int_med/1e3:.1f} +{sigma_int_err_plus/1e3:.1f} -{sigma_int_err_minus/1e3:.1f} km/s")

# -------------------------
# Corner Plot
# -------------------------
labels = ["M_total", "Rd", "f0", "f1", "sigma_int"]
fig = corner.corner(samples, labels=labels, show_titles=True)
fig.savefig("corner_plot_linear_boost.png")
plt.show()

# -------------------------
# Plot the Fitted Rotation Curve
# -------------------------
r_fit = np.linspace(0.5*kpc, 15*kpc, 300)
V_fit = V_model_linear(r_fit, M_total_med, Rd_med, f0_med, f1_med)
plt.figure(figsize=(10,6))
plt.errorbar(r_obs_kpc, V_obs/1e3, yerr=V_err/1e3, fmt='o', color='red', capsize=4, label='Observed Data')
plt.plot(r_fit/kpc, V_fit/1e3, lw=2, label='Median MCMC Fit (Linear Boost)')
plt.xlabel("Radius (kpc)", fontsize=14)
plt.ylabel("Rotation Velocity (km/s)", fontsize=14)
plt.title("NGC 3198 Rotation Curve with Linear Boost Model", fontsize=16)
plt.legend()
plt.grid(True)
plt.savefig("rotation_curve_fit_linear_boost.png")
plt.show()
