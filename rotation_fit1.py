import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Constants in SI units
G = 6.67430e-11           # Gravitational constant, m^3 kg^-1 s^-2
Msun = 1.989e30           # Solar mass in kg
kpc = 3.086e19            # 1 kiloparsec in meters

# --- Model Functions ---

# Exponential disk cumulative mass:
# M(r) = 2π Σ0 Rd^2 [1 - (1 + r/Rd)*exp(-r/Rd)]
# Here, Sigma0 is the central surface density (kg/m^2), and Rd is the scale length (m).
def cumulative_mass(r, Sigma0, Rd):
    return 2 * np.pi * Sigma0 * Rd**2 * (1 - (1 + r / Rd) * np.exp(-r / Rd))

# The rotation curve model (in m/s) taking into account an effective boost factor f.
# In your framework the effective rotation velocity is:
#   V(r) = sqrt( f * G * M(r) / r )
def V_model(r, Sigma0, Rd, f):
    M_r = cumulative_mass(r, Sigma0, Rd)
    return np.sqrt(f * G * M_r / r)

# --- Observational Data ---
# For demonstration we use approximate, published data for NGC 3198.
# (For a rigorous analysis you would retrieve the data from, e.g., Begeman 1989 or other sources.)
# r_obs in kpc and V_obs in km/s:
r_obs_kpc = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
V_obs_kms = np.array([60, 85, 105, 120, 135, 145, 150, 155, 158, 160, 161, 162, 163, 163, 163])
# Assume an uncertainty of 5 km/s in each measurement
V_err_kms = 5.0 * np.ones_like(V_obs_kms)

# Convert observed data into SI units (r in meters and V in m/s)
r_obs = r_obs_kpc * kpc
V_obs = V_obs_kms * 1e3
V_err = V_err_kms * 1e3

# --- Fitting the Model ---
# Our model has three free parameters:
#   Sigma0 (central surface density, kg/m^2),
#   Rd (disk scale length, m),
#   f (effective boost factor)
#
# For an initial guess:
#   Assume a luminous mass ~ 1e11 Msun and Rd ~ 3 kpc.
#   Then Sigma0 = M_total / (2π Rd^2).
M_total_guess = 1e11 * Msun
Rd_guess = 3.0 * kpc
Sigma0_guess = M_total_guess / (2 * np.pi * Rd_guess**2)
f_guess = 3.45  # as suggested by the 71% negative binding energy effect
p0 = [Sigma0_guess, Rd_guess, f_guess]

# Use curve_fit to fit the model. The uncertainties (V_err) are passed so that we obtain proper parameter errors.
popt, pcov = curve_fit(lambda r, Sigma0, Rd, f: V_model(r, Sigma0, Rd, f),
                        r_obs, V_obs, p0=p0, sigma=V_err, absolute_sigma=True)
Sigma0_fit, Rd_fit, f_fit = popt
Sigma0_err, Rd_err, f_err = np.sqrt(np.diag(pcov))

print("Fitted parameters:")
print(f"  Sigma0 = {Sigma0_fit:.3e} ± {Sigma0_err:.3e} kg/m²")
print(f"  Rd     = {Rd_fit/kpc:.3f} ± {Rd_err/kpc:.3f} kpc")
print(f"  f      = {f_fit:.3f} ± {f_err:.3f}")

# Evaluate the model at a dense set of radii for plotting:
r_fit = np.linspace(0.5*kpc, 15*kpc, 300)
V_fit = V_model(r_fit, Sigma0_fit, Rd_fit, f_fit)

# --- Statistical Assessment ---
# Compute the chi-square of the fit:
chi_sq = np.sum(((V_model(r_obs, Sigma0_fit, Rd_fit, f_fit) - V_obs) / V_err)**2)
dof = len(V_obs) - len(popt)
reduced_chi_sq = chi_sq / dof
print(f"Reduced chi-square: {reduced_chi_sq:.2f}")

# --- Plotting the Results ---
plt.figure(figsize=(10,6))
plt.errorbar(r_obs_kpc, V_obs_kms, yerr=V_err_kms, fmt='o', color='red', label='Observed Data (NGC 3198)', capsize=4)
plt.plot(r_fit/kpc, V_fit/1e3, label='Best Fit Model', lw=2)
plt.xlabel("Radius (kpc)", fontsize=14)
plt.ylabel("Rotation Velocity (km/s)", fontsize=14)
plt.title("Data-Driven Rotation Curve Model for NGC 3198", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
