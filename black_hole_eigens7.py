import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw
from scipy.linalg import eig

# =============================================================================
# 1. Inversion of the tortoise coordinate using Lambert W.
#    For Schwarzschild (M=1 in geometric units):
#      r* = r + 2 ln(r/2 - 1)
#    Invert: r = 2 (1 + W(exp(r*/2 - 1)) )
# =============================================================================
def r_of_rstar(rstar, M=1.0):
    argument = np.exp(rstar/(2*M) - 1)
    x = np.real(lambertw(argument))
    return 2 * M * (1 + x)

# =============================================================================
# 2. Regge–Wheeler potential for gravitational perturbations
#    For Schwarzschild, l = 2:
#      V(r) = (1 - 2M/r)[l(l+1)/r^2 - 6M/r^3]
#    We use M = 1.
# =============================================================================
def regge_wheeler_potential(r, M=1.0, l=2):
    return (1 - 2*M/r) * (l*(l+1)/r**2 - 6*M/r**3)

# =============================================================================
# 3. Set up the tortoise coordinate grid.
#    We choose a domain r* in [rstar_min, rstar_max] where the potential is active.
#    Here we use r* in [-30, 30] with high resolution.
# =============================================================================
M = 1.0  # Geometric units (G = c = 1, M = 1)
rstar_min = -30.0
rstar_max = 30.0
N_grid = 3000
rstar = np.linspace(rstar_min, rstar_max, N_grid)
# Compute r for each rstar
r_vals = np.array([r_of_rstar(rs, M) for rs in rstar])

# =============================================================================
# 4. Compute the Regge–Wheeler potential on the grid.
#    For l = 2 this is our effective potential.
# =============================================================================
l = 2
V_rw = regge_wheeler_potential(r_vals, M, l)
V_eff = V_rw  # No extra angular term needed here.

# =============================================================================
# 5. Add a complex absorbing potential near the boundaries.
#    We choose an absorber that turns on for |r*| >= L_abs.
# =============================================================================
def absorbing_potential(rstar, L_abs=25.0, rstar_max=30.0, W_max=0.005):
    if np.abs(rstar) < L_abs:
        return 0.0
    else:
        return 1j * W_max * ((np.abs(rstar) - L_abs) / (rstar_max - L_abs))**2

W_abs = np.array([absorbing_potential(rs, L_abs=25.0, rstar_max=rstar_max, W_max=0.005) for rs in rstar])
V_total = V_eff + W_abs

# Plot V_total for diagnostics.
plt.figure(figsize=(10, 6))
plt.plot(rstar, np.real(V_total), label="Re(V_total)")
plt.plot(rstar, np.imag(V_total), label="Im(V_total)", linestyle="--")
plt.xlabel("Tortoise coordinate r*")
plt.ylabel("Effective Potential")
plt.title("Effective Potential with Absorber")
plt.legend()
plt.grid(True)
plt.show()

# =============================================================================
# 6. Build the finite-difference Laplacian.
#    We use Dirichlet boundary conditions: R = 0 at r* = rstar_min and rstar_max.
#    Thus, we only construct the operator on interior points.
# =============================================================================
dx = rstar[1] - rstar[0]
N_int = N_grid - 2  # interior points
D2 = np.zeros((N_int, N_int), dtype=complex)
for i in range(N_int):
    if i > 0:
        D2[i, i-1] = 1.0
    D2[i, i] = -2.0
    if i < N_int - 1:
        D2[i, i+1] = 1.0
D2 = D2 / dx**2

# Extract the effective potential for the interior nodes.
V_diag = V_total[1:-1]

# =============================================================================
# 7. Introduce the scaling.
#    We assume a new time variable: t_new = S * τ.
#    Then d²/dτ² transforms into d²/dt_new² = (1/S²)d²/dτ².
#    Our operator becomes:
#       A_scaled = -D2/S^2 + diag(V_diag)
#
#    The (rescaled) eigenvalue equation becomes:
#       A_scaled R = λ R,   with ω_unscaled = √λ.
#    We then define the (physical) eigenfrequency as:
#       ω_phys = ω_unscaled / S.
#
#    We now set S as a tunable parameter. It should also scale with M.
#    If published GR data gives ω ~ 0.373 - 0.089i for M=1, then we
#    choose S so that ω_unscaled/S ~ 0.373. For example, if an earlier model
#    suggested S ~ 268, you might need to adjust it.
#    Here we will sweep over S values.
# =============================================================================
def compute_omega_phys(S):
    A_scaled = -D2 / (S**2) + np.diag(V_diag)
    eigvals, _ = eig(A_scaled)
    omega_unscaled = []
    for lam in eigvals:
        w = np.sqrt(lam)
        if w.real < 0:
            w = -w
        omega_unscaled.append(w)
    omega_unscaled = np.array(omega_unscaled)
    # The physical frequency is then
    return omega_unscaled / S

# =============================================================================
# 8. Sweep over several S values and print the fundamental (lowest nonzero)
#    eigenfrequency. We look for one with a real part near 0.373.
# =============================================================================
S_values = np.linspace(16.3, 16.5, 11)  # Try S = 50, 60, ... 150.
fundamental_map = {}
for S in S_values:
    omega_phys_all = compute_omega_phys(S)
    # We pick the eigenvalue with the smallest positive real part (ignoring any near-zero trivial mode).
    # Here, we filter for modes with 0 < Re(ω) < 1, say.
    candidates = [w for w in omega_phys_all if 0 < w.real < 1 and w.imag < 0]
    if len(candidates) > 0:
        # Pick the one with the smallest real part.
        candidate = min(candidates, key=lambda w: w.real)
        fundamental_map[S] = candidate
        print(f"S = {S:5.1f}: Fundamental ω_phys = {candidate}")
    else:
        print(f"S = {S:5.1f}: No candidate found.")

# =============================================================================
# 9. Plot the candidate fundamental frequency versus S.
# =============================================================================
S_plot = np.array(list(fundamental_map.keys()))
omega_plot = np.array([fundamental_map[S] for S in S_plot])
plt.figure(figsize=(10, 6))
plt.plot(S_plot, omega_plot.real, 'o-', label="Re(ω_phys)")
plt.plot(S_plot, omega_plot.imag, 's-', label="Im(ω_phys)")
plt.xlabel("Scaling factor S")
plt.ylabel("Candidate fundamental ω_phys")
plt.title("Candidate Fundamental Frequency vs. S")
plt.legend()
plt.grid(True)
plt.show()
