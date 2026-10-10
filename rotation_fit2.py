import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Constants in SI units
G = 6.67430e-11           # Gravitational constant, m^3 kg^-1 s^-2
Msun = 1.989e30           # Solar mass in kg
kpc = 3.086e19            # 1 kiloparsec in meters

# --- Reparameterized Model Function ---
# V_model2(r, M_total, Rd, f) returns the rotation velocity (m/s) at radius r given:
#   M_total   : total luminous mass (kg)
#   Rd        : disk scale length (m)
#   f         : effective boost factor (dimensionless)
def V_model2(r, M_total, Rd, f):
    # Cumulative mass for an exponential disk
    M_r = M_total * (1 - (1 + r / Rd) * np.exp(-r / Rd))
    return np.sqrt(f * G * M_r / r)

# --- Observational Data ---
# Using approximate rotation-curve data for NGC 3198 (in kpc and km/s)
r_obs_kpc = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
V_obs_kms = np.array([60, 85, 105, 120, 135, 145, 150, 155, 158, 160, 161, 162, 163, 163, 163])
# Assume an uncertainty of about 5 km/s per data point
V_err_kms = 5.0 * np.ones_like(V_obs_kms)

# Convert observed data to SI units
r_obs = r_obs_kpc * kpc     # radius in meters
V_obs = V_obs_kms * 1e3     # velocity in m/s
V_err = V_err_kms * 1e3     # uncertainty in m/s

# --- Initial Parameter Guesses ---
# For a typical spiral like NGC 3198:
M_total_guess = 1e11 * Msun   # ~10^11 solar masses in kg
Rd_guess = 3.0 * kpc          # ~3 kpc
f_guess = 3.45                # from the 71% effective boost idea
p0 = [M_total_guess, Rd_guess, f_guess]

# --- Physically Motivated Parameter Bounds ---
# M_total: between 1e10 and 1e12 solar masses (in kg)
lower_bounds = [1e10 * Msun, 0.5 * kpc, 1.0]
upper_bounds = [1e12 * Msun, 30 * kpc, 10.0]
bounds = (lower_bounds, upper_bounds)

# --- Fitting the Model ---
# We perform the fit using curve_fit with our model function and the bounds.
popt2, pcov2 = curve_fit(V_model2, r_obs, V_obs, p0=p0, sigma=V_err, absolute_sigma=True, bounds=bounds)
M_total_fit, Rd_fit, f_fit = popt2
M_total_err, Rd_err, f_err = np.sqrt(np.diag(pcov2))

print("Fitted parameters (reparameterized):")
print(f"  M_total = {M_total_fit:.3e} ± {M_total_err:.3e} kg ({M_total_fit/Msun:.3e} M☉)")
print(f"  Rd      = {Rd_fit/kpc:.3f} ± {Rd_err/kpc:.3f} kpc")
print(f"  f       = {f_fit:.3f} ± {f_err:.3f}")

# --- Statistical Assessment ---
# Compute chi-square of the fit
chi_sq = np.sum(((V_model2(r_obs, M_total_fit, Rd_fit, f_fit) - V_obs) / V_err) ** 2)
dof = len(V_obs) - len(popt2)
reduced_chi_sq = chi_sq / dof
print(f"Reduced chi-square: {reduced_chi_sq:.2f}")

# --- Plot the Data and Best-Fit Model ---
r_fit = np.linspace(0.5 * kpc, 15 * kpc, 300)
V_fit = V_model2(r_fit, M_total_fit, Rd_fit, f_fit)

plt.figure(figsize=(10, 6))
plt.errorbar(r_obs_kpc, V_obs_kms, yerr=V_err_kms, fmt='o', color='red',
             capsize=4, label='Observed Data (NGC 3198)')
plt.plot(r_fit/kpc, V_fit/1e3, lw=2, label='Best Fit Model')
plt.xlabel("Radius (kpc)", fontsize=14)
plt.ylabel("Rotation Velocity (km/s)", fontsize=14)
plt.title("Data-Driven Rotation Curve Model for NGC 3198", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
