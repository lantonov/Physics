import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw
from scipy.linalg import eig

# ------------------------------------------------------------------
# Improved inversion of the tortoise coordinate using Lambert W.
# For Schwarzschild (M=1) we have:
#    r* = r + 2M ln(r/(2M) - 1)
# Let x = (r/(2M))-1, so r* = 2M (x + ln x + 1).
# Then, r = 2M (1 + W(exp(r*/(2M)-1))).
# ------------------------------------------------------------------
def r_of_rstar(rstar, M=1.0):
    argument = np.exp(rstar/(2*M) - 1)
    x = np.real(lambertw(argument))
    return 2*M * (1 + x)

# ------------------------------------------------------------------
# Regge–Wheeler potential for Schwarzschild gravitational perturbations.
#   V(r) = (1 - 2M/r)[l(l+1)/r^2 - 6M/r^3]
# ------------------------------------------------------------------
def regge_wheeler_potential(r, M=1.0, l=2):
    return (1 - 2*M/r) * (l*(l+1)/r**2 - 6*M/r**3)

# ------------------------------------------------------------------
# Set up a revised grid in the tortoise coordinate, now using a smaller domain.
# We choose r* in [rstar_min, rstar_max] = [-50, 50] with high resolution.
# ------------------------------------------------------------------
M = 1.0  # Use geometrized units (G = c = 1, M = 1)
rstar_min = -50.0
rstar_max = 50.0
N_grid = 2000
rstar = np.linspace(rstar_min, rstar_max, N_grid)
r_vals = np.array([r_of_rstar(rs, M) for rs in rstar])

# ------------------------------------------------------------------
# Compute Regge–Wheeler potential and add angular contribution.
# For l = 2, the angular term (from angular separation) is approximated as:
#    V_ang = l(l+1) / (2M)^2. For M=1 and l=2, this gives 6/4 = 1.5.
# ------------------------------------------------------------------
l = 2  # dominant quadrupole mode
V_rw = regge_wheeler_potential(r_vals, M, l)
V_ang = l*(l+1) / (2*M)**2  # = 1.5 for M=1, l=2
V_eff = V_rw + V_ang

# ------------------------------------------------------------------
# Add a stronger complex absorbing potential near the boundaries.
# We'll activate the absorber for |r*| > 40 (in a domain of [-50, 50]).
# ------------------------------------------------------------------
def absorbing_potential(rstar, L_abs=40.0, rstar_max=50.0, W_max=0.005):
    # Quadratic absorption for |rstar| beyond L_abs.
    if np.abs(rstar) <= L_abs:
        return 0.0
    else:
        return 1j * W_max * ((np.abs(rstar) - L_abs) / (rstar_max - L_abs))**2

W_absorption = np.array([absorbing_potential(rs, L_abs=40.0, rstar_max=rstar_max, W_max=0.005) for rs in rstar])
V_eff_total = V_eff + W_absorption

# Plot the effective potential for diagnostics.
plt.figure(figsize=(10, 6))
plt.plot(rstar, np.real(V_eff_total), label="Re(V_eff_total)")
plt.plot(rstar, np.imag(V_eff_total), label="Im(V_eff_total)", linestyle="--")
plt.xlabel("Tortoise coordinate r*")
plt.ylabel("Effective Potential")
plt.title("Effective Potential with Absorbing Term (|r*| > 40)")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------------
# Construct the finite-difference Laplacian using a central difference scheme.
# ------------------------------------------------------------------
dx = rstar[1] - rstar[0]
D2 = np.zeros((N_grid, N_grid), dtype=complex)
for i in range(1, N_grid - 1):
    D2[i, i-1] = 1.0
    D2[i, i]   = -2.0
    D2[i, i+1] = 1.0
D2 = D2 / dx**2

# Enforce Dirichlet boundary conditions at the endpoints.
D2[0, :] = 0.0; D2[0, 0] = 1.0/dx**2
D2[-1, :] = 0.0; D2[-1, -1] = 1.0/dx**2

# ------------------------------------------------------------------
# Build the full operator A = -D2 + diag(V_eff_total)
# Compute the eigenvalue problem: A R = ω² R.
# ------------------------------------------------------------------
A = -D2 + np.diag(V_eff_total)
eigvals, eigvecs = eig(A)
eigvals = np.array(eigvals)

# Take square roots to get ω from ω².
omega_all = []
for lam in eigvals:
    # Compute the principal square root.
    w = np.sqrt(lam)
    if w.real < 0:
        w = -w
    omega_all.append(w)
omega_all = np.array(omega_all)

# ------------------------------------------------------------------
# Sort eigenvalues according to the absolute value of the imaginary part.
indices = np.argsort(np.abs(np.imag(omega_all)))
omega_sorted = omega_all[indices]

print("Computed eigenvalue frequencies (ω) for the first 10 modes:")
for i in range(10):
    print(f"Mode {i}: ω = {omega_sorted[i]}")

# ------------------------------------------------------------------
# Identify the fundamental QNM in the expected range.
# Published: ω ≈ 0.373 - 0.089 i for l=2, M=1.
# We search for an eigenvalue with 0.2 < Re(ω) < 0.6 and a small negative imaginary part.
fundamental_omega = None
for w in omega_sorted:
    if 0.2 < w.real < 0.6 and w.imag < 0:
        fundamental_omega = w
        break

if fundamental_omega is not None:
    print("\nFundamental (dominant) QNM frequency found:")
    print(f"ω = {fundamental_omega}")
    print(f"Expected (published) value: ω ≈ 0.373 - 0.089 i")
else:
    fundamental_omega = omega_sorted[0]
    print("\nNo mode found in the expected range; using the smallest eigenvalue:")
    print(f"ω = {fundamental_omega}")

# ------------------------------------------------------------------
# Plot the eigenfunction corresponding to the fundamental QNM.
# ------------------------------------------------------------------
mode_index = np.where(np.isclose(omega_all, fundamental_omega))[0]
if mode_index.size > 0:
    mode_index = mode_index[0]
else:
    mode_index = 0
R_fund = eigvecs[:, mode_index]

plt.figure(figsize=(10, 6))
plt.plot(rstar, np.real(R_fund), label="Real Part")
plt.plot(rstar, np.imag(R_fund), label="Imag Part", linestyle="--")
plt.xlabel("Tortoise coordinate r*")
plt.ylabel("Eigenfunction R(r*)")
plt.title("Fundamental QNM Eigenfunction")
plt.legend()
plt.grid(True)
plt.show()
