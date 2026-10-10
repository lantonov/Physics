import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw
from scipy.linalg import eig

# ------------------------------------------------------------------
# Improved inversion from r* to r using the Lambert W function.
# For Schwarzschild with M = 1:
#   r* = r + 2 ln(r/2 - 1)
# Solve for r using:
#   r = 2 (1 + W(exp(r*/2 - 1)))
# ------------------------------------------------------------------
def r_of_rstar(rstar, M=1.0):
    argument = np.exp(rstar/(2*M) - 1)
    x = np.real(lambertw(argument))
    return 2 * M * (1 + x)

# ------------------------------------------------------------------
# Regge–Wheeler potential for gravitational perturbations.
# For Schwarzschild and l = 2:
#   V(r) = (1 - 2M/r)[l(l+1)/r^2 - 6M/r^3]
# With M = 1, l = 2.
# ------------------------------------------------------------------
def regge_wheeler_potential(r, M=1.0, l=2):
    return (1 - 2*M/r) * (l*(l+1) / r**2 - 6*M / r**3)

# ------------------------------------------------------------------
# Set up a revised grid in tortoise coordinate r*.
# We now choose r* in [-30, 30] to concentrate on the key region.
# Use a high-resolution grid.
# ------------------------------------------------------------------
M = 1.0  # Geometrized units
rstar_min = -30.0
rstar_max = 30.0
N_grid = 3000  # Increase resolution
rstar = np.linspace(rstar_min, rstar_max, N_grid)
# Compute r for each r* using the Lambert W inversion.
r_vals = np.array([r_of_rstar(rs, M) for rs in rstar])

# ------------------------------------------------------------------
# Compute the Regge–Wheeler potential on the grid.
# (No extra angular term is added here as it is already included.)
# ------------------------------------------------------------------
l = 2  # For the quadrupolar (dominant) mode.
V_rw = regge_wheeler_potential(r_vals, M, l)
V_eff = V_rw  # This is our effective potential.

# ------------------------------------------------------------------
# Add a complex absorbing potential near the boundaries.
# We choose an absorber that activates for |r*| >= L_abs.
# ------------------------------------------------------------------
def absorbing_potential(rstar, L_abs=25.0, rstar_max=30.0, W_max=0.005):
    # Use a quadratic absorber.
    if np.abs(rstar) < L_abs:
        return 0.0
    else:
        return 1j * W_max * ((np.abs(rstar) - L_abs) / (rstar_max - L_abs))**2

W_absorption = np.array([absorbing_potential(rs, L_abs=25.0, rstar_max=rstar_max, W_max=0.005) for rs in rstar])
V_eff_total = V_eff + W_absorption

# Plot V_eff_total for diagnostics.
plt.figure(figsize=(10, 6))
plt.plot(rstar, np.real(V_eff_total), label="Re(V_eff_total)")
plt.plot(rstar, np.imag(V_eff_total), label="Im(V_eff_total)", linestyle="--")
plt.xlabel("Tortoise coordinate r*")
plt.ylabel("Effective Potential")
plt.title("Effective Potential on [-30, 30] with Absorber (|r*|>=25)")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------------
# Construct the finite-difference matrix for the second derivative.
# Use periodic boundary conditions.
# ------------------------------------------------------------------
dx = rstar[1] - rstar[0]
D2 = np.zeros((N_grid, N_grid), dtype=complex)
for i in range(N_grid):
    D2[i, (i - 1) % N_grid] = 1.0
    D2[i, i] = -2.0
    D2[i, (i + 1) % N_grid] = 1.0
D2 = D2 / dx**2

# ------------------------------------------------------------------
# Build the full operator A = -D2 + diag(V_eff_total)
# Solve the eigenvalue problem: A R = ω² R.
# ------------------------------------------------------------------
A = -D2 + np.diag(V_eff_total)
eigvals, eigvecs = eig(A)
# eigvals are our approximate ω².
# Take square roots to obtain ω. Choose branch with nonnegative real part.
omega_all = []
for lam in eigvals:
    w = np.sqrt(lam)
    if w.real < 0:
        w = -w
    omega_all.append(w)
omega_all = np.array(omega_all)

# ------------------------------------------------------------------
# Sort eigenvalues according to the absolute value of their imaginary parts.
indices = np.argsort(np.abs(np.imag(omega_all)))
omega_sorted = omega_all[indices]

print("Computed eigenvalue frequencies (ω) for the first 10 modes:")
for i in range(10):
    print(f"Mode {i}: ω = {omega_sorted[i]}")

# ------------------------------------------------------------------
# Search for a mode in the expected range.
# For Schwarzschild (M=1, l=2) published value is roughly ω ≃ 0.373 - 0.089 i.
fundamental_omega = None
for w in omega_sorted:
    if 0.2 < w.real < 0.6 and w.imag < 0:
        fundamental_omega = w
        break

if fundamental_omega is not None:
    print("\nFundamental (dominant) QNM frequency found:")
    print(f"ω = {fundamental_omega}")
    print("Expected (published) value: ω ≃ 0.373 - 0.089 i")
else:
    fundamental_omega = omega_sorted[0]
    print("\nNo mode found in the expected range; using the smallest eigenvalue:")
    print(f"ω = {fundamental_omega}")

# ------------------------------------------------------------------
# Plot the eigenfunction corresponding to the chosen mode.
# ------------------------------------------------------------------
mode_indices = np.where(np.isclose(omega_all, fundamental_omega, rtol=1e-3))[0]
if mode_indices.size > 0:
    i_mode = mode_indices[0]
else:
    i_mode = 0
R_fund = eigvecs[:, i_mode]
plt.figure(figsize=(10, 6))
plt.plot(rstar, np.real(R_fund), label="Real Part")
plt.plot(rstar, np.imag(R_fund), label="Imag Part", linestyle="--")
plt.xlabel("Tortoise coordinate r*")
plt.ylabel("Eigenfunction R(r*)")
plt.title("Eigenfunction for Fundamental QNM")
plt.legend()
plt.grid(True)
plt.show()
