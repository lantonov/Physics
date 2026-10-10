import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Constants
G0 = 4.300e-6  # Gravitational constant in kpc (km/s)^2 / M_sun

# Parameters for NGC 3198
M0 = 3e10  # Total disk mass in M_sun
Rd = 3.5  # Scale length in kpc

# Observational data for NGC 3198 (radius in kpc, velocity in km/s)
r_obs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
v_obs = np.array([110, 130, 150, 160, 165, 170, 175, 180, 185, 190, 195, 200])
v_err = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5])  # Constant error

# Exponential disk mass function
def enclosed_mass(r, M0, Rd):
    """Mass enclosed within radius r for an exponential disk."""
    return M0 * (1 - np.exp(-r / Rd))

# Gravitational potential for varying G
def potential(r, M0, Rd, G_func, *params):
    """Compute the gravitational potential at radius r."""
    M_enc = enclosed_mass(r, M0, Rd)
    Phi = -G_func(M_enc / r, *params) * M_enc / r
    return Phi


# Orbital velocity calculation
def velocity(r, M0, Rd, G_func, *params):
    """Compute the orbital velocity."""
    dr = 1e-4  # Small increment for numerical differentiation
    Phi_r = potential(r, M0, Rd, G_func, *params)
    Phi_r_plus = potential(r + dr, M0, Rd, G_func, *params)
    dPhi_dr = (Phi_r_plus - Phi_r) / dr
    # Ensure non-negative argument for sqrt
    return np.sqrt(np.maximum(0, r * dPhi_dr))


# Functional forms of G(Phi)

def G_linear(Phi, alpha):
    return G0 * (1 + alpha * Phi)

def G_exponential(Phi, beta):
    return G0 * np.exp(beta * Phi)

def G_power_law(Phi, gamma, n):
    return G0 * (1 + gamma * np.abs(Phi)**n)

# Wrapper function for curve fitting
def fit_velocity_model(r, alpha, beta, gamma, n):
    """Velocity model wrapper for curve fitting."""
    G_func = lambda Phi, alpha, beta, gamma, n: G_power_law(Phi, gamma, n) # Using power-law as an example
    return np.array([velocity(ri, M0, Rd, G_func, alpha, beta, gamma, n) for ri in r])


# Fit the model to the data
initial_params = [0.01, 0.01, 0.01, 1.5] # Adjusted initial guesses for alpha, beta, gamma, n
bounds = ([0, 0, 0, 1], [np.inf, np.inf, np.inf, np.inf])  # Adjust bounds as needed
popt, pcov = curve_fit(fit_velocity_model, r_obs, v_obs, sigma=v_err, p0=initial_params, bounds=bounds)


# Extract best-fit parameters
alpha_best, beta_best, gamma_best, n_best = popt

# Compute the best-fit rotation curve
r_fit = np.linspace(1, 12, 100)
v_fit = fit_velocity_model(r_fit, alpha_best, beta_best, gamma_best, n_best)

# Plot the results
plt.errorbar(r_obs, v_obs, yerr=v_err, fmt='o', label="Observed Data", capsize=3)
plt.plot(r_fit, v_fit, label="Best-Fit Model", color="red")
plt.xlabel("Radius (kpc)")
plt.ylabel("Velocity (km/s)")
plt.title("Best-Fit Rotation Curve for NGC 3198")
plt.legend()
plt.grid()
plt.show()

