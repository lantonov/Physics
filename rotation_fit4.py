#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

# -------------------------
# Constants and Conversions
# -------------------------
G = 6.67430e-11              # Gravitational constant in m^3 kg^-1 s^-2
Msun = 1.989e30              # Solar mass in kg
kpc = 3.086e19               # 1 kiloparsec in meters

# -------------------------
# Observational Data (NGC 3198)
# -------------------------
# Approximate rotation curve: radius in kpc, velocity in km/s
r_obs_kpc = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
V_obs_kms = np.array([60, 85, 105, 120, 135, 145, 150, 155, 158, 160, 161, 162, 163, 163, 163])
V_err_kms = 5.0 * np.ones_like(V_obs_kms)  # measurement uncertainty (km/s)

# Convert the observational data to SI units (r in meters, V in m/s)
r_obs = r_obs_kpc * kpc         # in meters
V_obs = V_obs_kms * 1e3         # in m/s
V_err = V_err_kms * 1e3         # in m/s

# -------------------------
# Model Functions
# -------------------------
def M_disk(r, M_total, Rd):
    """
    Cumulative luminous mass of an exponential disk.
    M(r) = M_total * [1 - (1 + r/Rd) * exp(-r/Rd)]
    Parameters:
      r: radius (m)
      M_total: total luminous mass (kg)
      Rd: disk scale length (m)
    """
    return M_total * (1 - (1 + r / Rd) * np.exp(-r / Rd))

def V_model(r, M_total, Rd, f):
    """
    Model rotation velocity from luminous mass with an effective boost.
    V(r) = sqrt( f * G * M_disk(r) / r )
    """
    return np.sqrt(f * G * M_disk(r, M_total, Rd) / r)

# -------------------------
# Log-Prior with Informative Priors
# -------------------------
def log_prior(theta):
    """
    Log-prior for parameters:
      theta = (M_total, Rd, f, sigma_int)
      M_total: total luminous mass in kg
      Rd: disk scale length in m
      f: effective boost factor (dimensionless)
      sigma_int: intrinsic scatter (in m/s)
      
    We impose:
      - Uniform bounds on M_total: [1e10 Msun, 1e12 Msun]
      - Uniform bounds on Rd: [0.5 kpc, 30 kpc]
      - Uniform bounds on f: [1.0, 10.0]
      - Uniform bounds on sigma_int: [0, 20 km/s]
      
    Plus Gaussian priors:
      - ln_prior_M_total assumes log10(M_total) ~ N(41.3, 0.3^2)
      - ln_prior_Rd assumes (Rd in kpc) ~ N(3.0, 0.5^2)
      - ln_prior_f assumes f ~ N(3.45, 1.0^2)
    """
    M_total, Rd, f, sigma_int = theta

    # Check allowed ranges:
    if not (1e10 * Msun < M_total < 1e12 * Msun):
        return -np.inf
    if not (0.5 * kpc < Rd < 30 * kpc):
        return -np.inf
    if not (1.0 < f < 10.0):
        return -np.inf
    if not (0 <= sigma_int < 20e3):
        return -np.inf

    # Gaussian priors (we add log-likelihood contributions):
    # For M_total, work in log10 space:
    logM = np.log10(M_total)
    mu_logM = 41.3
    sigma_logM = 0.3
    lnP_M = -0.5 * ((logM - mu_logM) / sigma_logM)**2
    
    # For Rd in kpc:
    Rd_kpc = Rd / kpc
    mu_Rd = 3.0
    sigma_Rd = 0.5
    lnP_Rd = -0.5 * ((Rd_kpc - mu_Rd) / sigma_Rd)**2
    
    # For f:
    mu_f = 3.45
    sigma_f = 1.0
    lnP_f = -0.5 * ((f - mu_f) / sigma_f)**2
    
    return lnP_M + lnP_Rd + lnP_f

# -------------------------
# Log-Likelihood and Log-Posterior Functions
# -------------------------
def log_likelihood(theta, r, V_obs, V_err):
    M_total, Rd, f, sigma_int = theta
    model = V_model(r, M_total, Rd, f)
    sigma_tot = np.sqrt(V_err**2 + sigma_int**2)
    return -0.5 * np.sum(((V_obs - model) / sigma_tot)**2 + np.log(2 * np.pi * sigma_tot**2))

def log_posterior(theta, r, V_obs, V_err):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, r, V_obs, V_err)

# -------------------------
# MCMC Setup
# -------------------------
# Parameter bounds for initialization (same as in log_prior):
M_total_min, M_total_max = 1e10 * Msun, 1e12 * Msun  # in kg
Rd_min, Rd_max = 0.5 * kpc, 30 * kpc                 # in m
f_min, f_max = 1.0, 10.0                             # dimensionless
sigma_int_min, sigma_int_max = 0, 20e3               # in m/s

ndim = 4  # parameters: M_total, Rd, f, sigma_int
nwalkers = 50

# Initialize walkers uniformly across the allowed ranges:
pos = np.empty((nwalkers, ndim))
for i in range(nwalkers):
    M_total_i = np.random.uniform(M_total_min, M_total_max)
    Rd_i = np.random.uniform(Rd_min, Rd_max)
    f_i = np.random.uniform(f_min, f_max)
    sigma_int_i = np.random.uniform(sigma_int_min, sigma_int_max)
    pos[i] = [M_total_i, Rd_i, f_i, sigma_int_i]

# -------------------------
# Run the MCMC
# -------------------------
nsteps = 5000
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(r_obs, V_obs, V_err))
print("Running MCMC...")
sampler.run_mcmc(pos, nsteps, progress=True)

# Discard burn-in (first half) and flatten chain.
burnin = nsteps // 2
samples = sampler.get_chain(discard=burnin, flat=True)

# -------------------------
# Analyze the Results
# -------------------------
def median_and_errors(param_samples):
    mcmc = np.percentile(param_samples, [16, 50, 84])
    median = mcmc[1]
    error_plus = mcmc[2] - median
    error_minus = median - mcmc[0]
    return median, error_plus, error_minus

M_total_median, M_total_err_plus, M_total_err_minus = median_and_errors(samples[:, 0])
Rd_median, Rd_err_plus, Rd_err_minus = median_and_errors(samples[:, 1])
f_median, f_err_plus, f_err_minus = median_and_errors(samples[:, 2])
sigma_int_median, sigma_int_err_plus, sigma_int_err_minus = median_and_errors(samples[:, 3])

print("MCMC Results:")
print(f"  M_total = {M_total_median:.3e} +{M_total_err_plus:.3e} -{M_total_err_minus:.3e} kg")
print(f"            ({M_total_median/Msun:.3e} M_sun)")
print(f"  Rd      = {Rd_median/kpc:.3f} +{Rd_err_plus/kpc:.3f} -{Rd_err_minus/kpc:.3f} kpc")
print(f"  f       = {f_median:.3f} +{f_err_plus:.3f} -{f_err_minus:.3f}")
print(f"  sigma_int = {sigma_int_median/1e3:.1f} +{sigma_int_err_plus/1e3:.1f} -{sigma_int_err_minus/1e3:.1f} km/s")

# -------------------------
# Generate Corner Plot of the Posteriors
# -------------------------
labels = ["M_total", "Rd", "f", "sigma_int"]
figure = corner.corner(samples, labels=labels, show_titles=True)
figure.savefig("corner_plot_improved.png")
plt.show()

# -------------------------
# Plot the Rotation Curve with the Median MCMC Parameters
# -------------------------
r_fit = np.linspace(0.5 * kpc, 15 * kpc, 300)
V_fit = V_model(r_fit, M_total_median, Rd_median, f_median)

plt.figure(figsize=(10, 6))
plt.errorbar(r_obs_kpc, V_obs/1e3, yerr=V_err/1e3, fmt='o', color='red', capsize=4, label='Observed Data')
plt.plot(r_fit/kpc, V_fit/1e3, lw=2, label='Median MCMC Fit')
plt.xlabel("Radius (kpc)", fontsize=14)
plt.ylabel("Rotation Velocity (km/s)", fontsize=14)
plt.title("NGC 3198 Rotation Curve (Improved MCMC Bayesian Fit)", fontsize=16)
plt.legend()
plt.grid(True)
plt.savefig("rotation_curve_fit_improved.png")
plt.show()
