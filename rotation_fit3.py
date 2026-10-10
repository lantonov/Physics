#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

# -------------------------
# Constants and Conversions
# -------------------------
G = 6.67430e-11              # Gravitational constant, in m^3 kg^-1 s^-2
Msun = 1.989e30              # Solar mass in kg
kpc = 3.086e19               # 1 kiloparsec in meters

# -------------------------
# Observational Data (NGC 3198)
# -------------------------
# Approximate rotation curve data: radius in kpc, velocity in km/s
r_obs_kpc = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
V_obs_kms = np.array([60, 85, 105, 120, 135, 145, 150, 155, 158, 160, 161, 162, 163, 163, 163])
V_err_kms = 5.0 * np.ones_like(V_obs_kms)  # measurement uncertainties in km/s

# Convert observational data to SI units (r in m, V in m/s)
r_obs = r_obs_kpc * kpc         # radius in meters
V_obs = V_obs_kms * 1e3         # velocity in m/s
V_err = V_err_kms * 1e3         # uncertainty in m/s

# -------------------------
# Model Functions
# -------------------------
def M_disk(r, M_total, Rd):
    """
    Cumulative mass at radius r for an exponential disk with total mass M_total and scale length Rd.
    M(r) = M_total * [1 - (1 + r/Rd) exp(-r/Rd)]
    """
    return M_total * (1 - (1 + r / Rd) * np.exp(-r / Rd))

def V_model(r, M_total, Rd, f):
    """
    Rotational velocity at radius r from luminous matter with an effective boost factor f.
    V(r) = sqrt( f * G * M_disk(r) / r )
    """
    return np.sqrt(f * G * M_disk(r, M_total, Rd) / r)

# -------------------------
# Log-Probability Functions
# -------------------------
def log_prior(theta):
    """
    Log-prior function.
    Parameters theta = (M_total, Rd, f, sigma_int)
    M_total: total luminous mass in kg
    Rd: disk scale length in m
    f: effective boost factor (dimensionless)
    sigma_int: intrinsic scatter (in m/s)
    """
    M_total, Rd, f, sigma_int = theta
    # Set uniform priors within physically plausible ranges:
    if not (1e10 * Msun < M_total < 1e12 * Msun):
        return -np.inf
    if not (0.5 * kpc < Rd < 30 * kpc):
        return -np.inf
    if not (1.0 < f < 10.0):
        return -np.inf
    if not (0 <= sigma_int < 20e3):  # sigma_int between 0 and 20 km/s
        return -np.inf
    return 0.0

def log_likelihood(theta, r, V_obs, V_err):
    """
    Gaussian log-likelihood.
    sigma_tot includes the measurement uncertainty and any additional scatter sigma_int.
    """
    M_total, Rd, f, sigma_int = theta
    model = V_model(r, M_total, Rd, f)
    sigma_tot = np.sqrt(V_err**2 + sigma_int**2)
    return -0.5 * np.sum(((V_obs - model) / sigma_tot)**2 + np.log(2 * np.pi * sigma_tot**2))

def log_posterior(theta, r, V_obs, V_err):
    """
    Log-posterior = log(prior) + log(likelihood)
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, r, V_obs, V_err)

# -------------------------
# MCMC Setup
# -------------------------
# Parameter bounds from the log_prior:
M_total_min, M_total_max = 1e10 * Msun, 1e12 * Msun   # in kg
Rd_min, Rd_max = 0.5 * kpc, 30 * kpc                   # in m
f_min, f_max = 1.0, 10.0                               # dimensionless
sigma_int_min, sigma_int_max = 0, 20e3                 # in m/s

ndim = 4       # number of free parameters: M_total, Rd, f, sigma_int
nwalkers = 50  # number of walkers

# Initialize walkers uniformly across allowed ranges
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

# Discard burn-in (first half) and flatten the chain.
burnin = nsteps // 2
samples = sampler.get_chain(discard=burnin, flat=True)

# -------------------------
# Analyze the Results
# -------------------------
# For each parameter compute the median and the 16th and 84th percentiles.
def median_and_errors(param_samples):
    mcmc = np.percentile(param_samples, [16, 50, 84])
    median = mcmc[1]
    error_minus = median - mcmc[0]
    error_plus = mcmc[2] - median
    return median, error_plus, error_minus

M_total_median, M_total_err_plus, M_total_err_minus = median_and_errors(samples[:,0])
Rd_median, Rd_err_plus, Rd_err_minus = median_and_errors(samples[:,1])
f_median, f_err_plus, f_err_minus = median_and_errors(samples[:,2])
sigma_int_median, sigma_int_err_plus, sigma_int_err_minus = median_and_errors(samples[:,3])

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
figure.savefig("corner_plot.png")
plt.show()

# -------------------------
# Plot the Rotation Curve with the Median MCMC Parameters
# -------------------------
r_fit = np.linspace(0.5 * kpc, 15 * kpc, 300)
V_fit = V_model(r_fit, M_total_median, Rd_median, f_median)

plt.figure(figsize=(10,6))
plt.errorbar(r_obs_kpc, V_obs/1e3, yerr=V_err/1e3, fmt='o', color='red',
             capsize=4, label='Observed Data')
plt.plot(r_fit/kpc, V_fit/1e3, lw=2, label='Median MCMC Fit')
plt.xlabel("Radius (kpc)", fontsize=14)
plt.ylabel("Rotation Velocity (km/s)", fontsize=14)
plt.title("NGC 3198 Rotation Curve (MCMC Bayesian Fit)", fontsize=16)
plt.legend()
plt.grid(True)
plt.savefig("rotation_curve_fit.png")
plt.show()
