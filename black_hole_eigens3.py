import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw
from scipy.linalg import eig

# --- Improved r_of_rstar using the Lambert W function ---
def r_of_rstar(rstar, M=1.0):
    # Using the analytic inversion for the Schwarzschild tortoise coordinate:
    #   r* = r + 2M ln(r/(2M) - 1)
    # Let x = r/(2M) - 1, then r* = 2M (x + ln x + 1)
    # Solve: x + ln x = r*/(2M) - 1  --> x = W(exp(r*/(2M) - 1))
    # Then, r = 2M (1 + x)
    argument = np.exp(rstar/(2*M) - 1)
    x = np.real(lambertw(argument))
    return 2*M * (1 + x)

# --- Regge–Wheeler Potential function ---
def regge_wheeler_potential(r, M=1.0, l=2):
    # For gravitational perturbations (Schwarzschild) the Regge–Wheeler potential is:
    #   V(r) = (1 - 2M/r)[l(l+1)/r^2 - 6M/r^3]
    return (1 - 2*M/r) * (l*(l+1) / r**2 - 6*M / r**3)

# --- Set up an extended grid in the tortoise coordinate r* ---
# Because the horizon is at r* -> -∞, we choose a large negative bound.
M = 1.0  # work in geometrized units (G = c = 1)
rstar_min = -200.0
rstar_max = 200.0
N_grid = 4000  # use a high-resolution grid
rstar = np.linspace(rstar_min, rstar_max, N_grid)
# Compute radial coordinate r for each rstar using the Lambert W inversion.
r_vals = np.array([r_of_rstar(rs, M) for rs in rstar])

# --- Compute the Regge–Wheeler potential and add the angular term ---
l = 2  # quadrupole mode
V_rw = regge_wheeler_potential(r_vals, M, l)
# Angular-term contribution; in geometric units, an approximate angular separation term is:
V_ang = l * (l + 1) / (2*M)**2   # For M = 1, l = 2, this gives 2*3/4 = 1.5
# Total effective potential:
V_eff = V_rw + V_ang

# --- Add a complex absorbing potential near the boundaries ---
def absorbing_potential(rstar, L_abs=150.0, rstar_max=200.0, W_max=0.001):
    # For |rstar| > L_abs, increase the absorber quadratically.
    if np.abs(rstar) < L_abs:
        return 0.0
    else:
        return 1j * W_max * ((np.abs(rstar) - L_abs) / (rstar_max - L_abs))**2

W_absorption = np.array([absorbing_potential(rs, L_abs=150.0, rstar_max=rstar_max, W_max=0.001) for rs in rstar])
V_eff_total = V_eff + W_absorption

# --- Plot the effective potential (for diagnostics) ---
plt.figure(figsize=(10, 6))
plt.plot(rstar, np.real(V_eff_total), label="Real(V_eff)")
plt.plot(rstar, np.imag(V_eff_total), label="Imag(V_eff)", linestyle="--")
plt.xlabel("Tortoise coordinate r*")
plt.ylabel("Effective potential V_eff_total")
plt.title("Effective Potential with Absorbing Boundary")
plt.legend()
plt.grid(True)
plt.show()

# --- Construct the finite-difference matrix for d^2/dr*^2 ---
dx = rstar[1] - rstar[0]
D2 = np.zeros((N_grid, N_grid), dtype=complex)
for i in range(1, N_grid-1):
    D2[i, i-1] = 1.0
    D2[i, i] = -2.0
    D2[i, i+1] = 1.0
D2 = D2 / dx**2
# Enforce Dirichlet boundary conditions at the edges:
D2[0, :] = 0.0; D2[0, 0] = 1.0/dx**2
D2[-1, :] = 0.0; D2[-1, -1] = 1.0/dx**2

# --- Build the full operator A = -D^2 + diag(V_eff_total) ---
A = -D2 + np.diag(V_eff_total)

# --- Solve the eigenvalue problem A R = ω² R ---
eigvals, eigvecs = eig(A)
eigvals = np.array(eigvals)  # omega^2 values (may be complex)
# Take the square roots to obtain ω. Choose the branch with nonnegative real part.
omega_all = []
for lam in eigvals:
    w = np.sqrt(lam)
    if w.real < 0:
        w = -w
    omega_all.append(w)
omega_all = np.array(omega_all)

# --- Sort eigenvalues by the absolute value of their imaginary parts (damping rates) ---
indices = np.argsort(np.abs(np.imag(omega_all)))
omega_sorted = omega_all[indices]

print("Computed eigenvalue frequencies (ω) for the first 10 modes:")
for i in range(10):
    print(f"Mode {i}: ω = {omega_sorted[i]}")

# --- Identify the fundamental (dominant) QNM by searching for a mode with small damping ---
# In published data for Schwarzschild l=2, M=1:
#   ω ≃ 0.373 - 0.089 i.
found_mode = None
for w in omega_sorted:
    if 0.3 < w.real < 0.5 and w.imag < 0:
        found_mode = w
        break

if found_mode is not None:
    fundamental_omega = found_mode
    print("\nFundamental (dominant) QNM frequency found:")
    print(f"ω = {fundamental_omega}")
    print(f"Published value (approx): ω = 0.373 - 0.089 i")
else:
    # If not found, simply choose the mode with the smallest real part.
    fundamental_omega = omega_sorted[0]
    print("\nNo mode found in the expected range; using the smallest eigenvalue:")
    print(f"ω = {fundamental_omega}")

# --- Plot the eigenfunction corresponding to the fundamental QNM ---
mode_index = np.where(np.isclose(omega_all, fundamental_omega))[0]
if mode_index.size > 0:
    mode_index = mode_index[0]
else:
    mode_index = 0
R_fund = eigvecs[:, mode_index]
plt.figure(figsize=(10,6))
plt.plot(rstar, np.real(R_fund), label="Real part")
plt.plot(rstar, np.imag(R_fund), label="Imaginary part", linestyle="--")
plt.xlabel("Tortoise coordinate r*")
plt.ylabel("Eigenfunction R(r*)")
plt.title("Fundamental QNM Eigenfunction")
plt.legend()
plt.grid(True)
plt.show()
