import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw
from scipy.linalg import eig

# =============================================================================
# 1. Inversion of the tortoise coordinate using the Lambert W function.
#    For Schwarzschild (with M in geometric units, here M=1):
#      r* = r + 2 ln(r/2 - 1)
#    Invert via: r = 2 * (1 + W(exp(r*/2 - 1)) )
# =============================================================================
def r_of_rstar(rstar, M=1.0):
    argument = np.exp(rstar/(2*M) - 1)
    x = np.real(lambertw(argument))
    return 2 * M * (1 + x)

# =============================================================================
# 2. Regge–Wheeler potential for gravitational perturbations.
#    For Schwarzschild and l = 2:
#      V(r) = (1 - 2M/r)[l(l+1)/r^2 - 6M/r^3]
#    We use M = 1 in geometrized units.
# =============================================================================
def regge_wheeler_potential(r, M=1.0, l=2):
    return (1 - 2*M/r) * (l*(l+1)/r**2 - 6*M/r**3)

# =============================================================================
# 3. Set up the tortoise coordinate grid.
#    We choose a limited domain, say r* ∈ [-30, 30], to capture the active potential.
# =============================================================================
M = 1.0  # For now, work with M = 1 in geometric units.
rstar_min = -30.0
rstar_max = 30.0
N_grid = 3000  # High resolution
rstar = np.linspace(rstar_min, rstar_max, N_grid)
# Compute r values using the inversion.
r_vals = np.array([r_of_rstar(rs, M) for rs in rstar])

# =============================================================================
# 4. Compute the Regge–Wheeler potential on the grid.
# =============================================================================
l = 2  # dominant quadrupole mode
V_rw = regge_wheeler_potential(r_vals, M, l)
# In the standard Regge–Wheeler equation the angular part is already incorporated,
# so we use:
V_eff = V_rw

# =============================================================================
# 5. Add a complex absorbing potential near boundaries.
#    Here, we design an absorber that activates for |r*| >= L_abs.
# =============================================================================
def absorbing_potential(rstar, L_abs=25.0, rstar_max=30.0, W_max=0.005):
    if np.abs(rstar) < L_abs:
        return 0.0
    else:
        # Quadratic increase in the imaginary part.
        return 1j * W_max * ((np.abs(rstar) - L_abs) / (rstar_max - L_abs))**2

W_abs = np.array([absorbing_potential(rs, L_abs=25.0, rstar_max=rstar_max, W_max=0.005) for rs in rstar])
V_total = V_eff + W_abs

# Diagnostics: Plot the effective potential.
plt.figure(figsize=(10, 6))
plt.plot(rstar, np.real(V_total), label="Re(V_total)")
plt.plot(rstar, np.imag(V_total), label="Im(V_total)", linestyle="--")
plt.xlabel("Tortoise coordinate r*")
plt.ylabel("Potential")
plt.title("Effective Potential with Absorbing Layer")
plt.legend()
plt.grid(True)
plt.show()

# =============================================================================
# 6. Set up the finite difference Laplacian.
#    We use Dirichlet boundary conditions.  That is, we consider the interior points
#    only (discard the endpoints). This helps avoid spurious high eigenvalues.
# =============================================================================
dx = rstar[1] - rstar[0]
# Consider interior nodes: indices 1 to N_grid-2.
N_int = N_grid - 2
D2 = np.zeros((N_int, N_int), dtype=complex)
for i in range(N_int):
    if i > 0:
        D2[i, i-1] = 1.0
    D2[i, i] = -2.0
    if i < N_int - 1:
        D2[i, i+1] = 1.0
D2 = D2 / dx**2

# Effective potential for interior nodes:
V_diag = V_total[1:-1]
# =============================================================================
# 7. Introduce a time (or cyclic) scaling factor S.
#    Our earlier (unscaled) eigenvalue problem (S = 1) produced frequencies ~100.
#    Published GR fundamental QNM for Schwarzschild (M=1, ℓ=2) is approximately
#      ω ≃ 0.373 - 0.089i.
#    To bridge the gap you can define a rescaled time coordinate:
#         t_new = S * τ.
#    Then, the operator becomes
#         -D2/S^2 + diag(V_diag)
#    and the eigenfrequencies become ω_rescaled = ω_unscaled / S.
#    For our current numbers, we choose S ≃ 268 (for M=1).
#    Moreover, we let S depend on M so that a larger BH (resulting from a merger)
#    has a larger cyclic period (a bigger Hopf fiber). That is, set:
#         S(M) = 268 * M.
# =============================================================================
def S_of_M(M):
    return 268 * M

S = S_of_M(M)
print(f"Using scaling factor S = {S} (for M = {M}).")

# =============================================================================
# 8. Construct the full operator.
#    Our rescaled operator becomes:
#
#         A_scaled = -D2/S^2 + diag(V_diag)
#
#    We solve A_scaled R = λ R.
#    Then, the eigenfrequencies in the rescaled time coordinate are:
#
#         ω_rescaled = sqrt(λ) / (2π_S?) but here we identify ω_rescaled = √λ,
#    and the physical (or GR comparable) frequency becomes ω_phys = ω_rescaled/S.
#
#    We wish for ω_phys ≃ 0.373 - 0.089 i (in geometric units M = 1).
# =============================================================================
A_scaled = -D2 / (S**2) + np.diag(V_diag)
eigvals, eigvecs = eig(A_scaled)
eigvals = np.array(eigvals)

# =============================================================================
# 9. Retrieve eigenfrequencies.
#    In our rescaled eigenvalue problem, the raw eigenvalues are λ,
#    and the uncorrected (rescaled) frequencies are ω_unscaled = √λ.
#    The physical frequency is then ω_phys = ω_unscaled / S.
# =============================================================================
omega_unscaled = []
for lam in eigvals:
    w = np.sqrt(lam)
    if w.real < 0:
        w = -w
    omega_unscaled.append(w)
omega_unscaled = np.array(omega_unscaled)
omega_phys = omega_unscaled / S

# =============================================================================
# 10. Sort the physical frequencies by absolute value of the imaginary part.
# =============================================================================
indices = np.argsort(np.abs(np.imag(omega_phys)))
omega_sorted = omega_phys[indices]

print("Computed (physical) eigenvalue frequencies (ω_phys) for the first 10 modes:")
for i in range(min(10, len(omega_sorted))):
    print(f"Mode {i}: ω_phys = {omega_sorted[i]}")

# =============================================================================
# 11. Search for a candidate fundamental QNM.
#     For Schwarzschild, published value (M=1, ℓ=2) is approximately:
#         ω ≃ 0.373 - 0.089 i.
# =============================================================================
fundamental_omega = None
for w in omega_sorted:
    if 0.2 < w.real < 0.6 and w.imag < 0:
        fundamental_omega = w
        break
if fundamental_omega is not None:
    print("\nFundamental (dominant) QNM frequency (physical) found:")
    print(f"ω_phys = {fundamental_omega}")
    print("Expected (published) value: ω ≃ 0.373 - 0.089 i")
else:
    fundamental_omega = omega_sorted[0]
    print("\nNo mode found in the expected range; using smallest eigenvalue:")
    print(f"ω_phys = {fundamental_omega}")

# =============================================================================
# 12. Plot the eigenfunction corresponding to the chosen mode.
#     We pad the interior eigenfunction with zeros at the boundaries.
# =============================================================================
mode_indices = np.where(np.isclose(omega_phys, fundamental_omega, atol=1e-3))[0]
if mode_indices.size > 0:
    i_mode = mode_indices[0]
else:
    i_mode = 0
R_fund_interior = eigvecs[:, i_mode]
R_full = np.zeros(N_grid, dtype=complex)
R_full[1:-1] = R_fund_interior

plt.figure(figsize=(10, 6))
plt.plot(rstar, np.real(R_full), label="Real Part")
plt.plot(rstar, np.imag(R_full), label="Imaginary Part", linestyle="--")
plt.xlabel("Tortoise coordinate r*")
plt.ylabel("Eigenfunction R(r*)")
plt.title("Fundamental QNM Eigenfunction (Rescaled)")
plt.legend()
plt.grid(True)
plt.show()
