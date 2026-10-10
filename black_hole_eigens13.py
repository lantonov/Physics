import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.special import lambertw

# =============================================================================
# 1. Chebyshev collocation with a linear mapping.
#    Compute Chebyshev differentiation matrix on [-1,1] and map to r* domain.
# =============================================================================
def cheb(N):
    """Compute the Chebyshev differentiation matrix and grid on [-1, 1]."""
    if N == 0:
        return np.array([[0]]), np.array([1])
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1)
    c[0] = 2; c[-1] = 2
    c = c * ((-1) ** np.arange(N + 1))
    X = np.tile(x, (N + 1, 1))
    dX = X - X.T
    D = np.outer(c, 1/c) / (dX + np.eye(N + 1))
    D = D - np.diag(np.sum(D, axis=1))
    return D, x

def linear_map(x, a, b):
    """Map x in [-1,1] to r* in [a,b]."""
    return 0.5 * (b - a) * x + 0.5 * (a + b)

# Choose Chebyshev resolution:
N = 200  # increased resolution relative to earlier experiments
D_cheb, x_cheb = cheb(N)
# We choose a narrow domain where the potential is active:
rstar_min = -10.0   
rstar_max = 16.6
rstar = linear_map(x_cheb, rstar_min, rstar_max)

# =============================================================================
# 2. Inversion: r* -> r using the analytic formula.
#    For Schwarzschild (M=1), r* = r + 2 ln(r/2 - 1) so that
#    r = 2 (1 + W(exp(r*/2 - 1))).
# =============================================================================
def r_of_rstar(rstar, M=1.0):
    argument = np.exp(rstar/(2*M) - 1)
    x_val = np.real(lambertw(argument))
    return 2 * M * (1 + x_val)

r_vals = np.array([r_of_rstar(rs, M=1.0) for rs in rstar])

# =============================================================================
# 3. Regge–Wheeler potential for ℓ=2:
#    V(r) = (1 - 2M/r)[ℓ(ℓ+1)/r^2 - 6M/r^3] with M=1.
# =============================================================================
def regge_wheeler_potential(r, M=1.0, l=2):
    return (1 - 2*M/r) * (l*(l+1)/r**2 - 6*M/r**3)

l_val = 2
V_RW = regge_wheeler_potential(r_vals, M=1.0, l=l_val)
V_eff = V_RW   # This is our effective potential

# =============================================================================
# 4. Add an absorber near the boundaries to mimic outgoing conditions.
#    We turn the absorber on for |r*| >= L_abs.
# =============================================================================
def absorber(rstar, L_abs=3.0, rstar_max=rstar_max, W_max=0.23):
    # A quadratic ramp absorber.
    return 1j * -W_max * np.maximum(0, (np.abs(rstar) - L_abs)/(rstar_max - L_abs))**2

W_abs = np.array([absorber(rs, L_abs=3.0, rstar_max=rstar_max, W_max=0.23) for rs in rstar])
V_total = V_eff + W_abs

plt.figure(figsize=(8,5))
plt.plot(rstar, np.real(V_total), label="Re(V_total)")
plt.plot(rstar, np.imag(V_total), label="Im(V_total)", linestyle="--")
plt.xlabel("Tortoise coordinate r*")
plt.ylabel("Effective Potential")
plt.title("Effective Potential on [-4,4] with Absorber")
plt.legend()
plt.grid(True)
plt.show()

# =============================================================================
# 5. Construct the differentiation matrices.
#    The Chebyshev differentiation matrix on [-1,1] must be scaled for the physical domain.
#    For r* = 0.5*(b-a)*x + 0.5*(a+b), the derivative d/dx = (2/(b-a)) d/dr*.
# =============================================================================
scale_factor = 2.0 / (rstar_max - rstar_min)   # here 2/8 = 0.25
D_rstar = scale_factor * D_cheb
D2 = np.dot(D_rstar, D_rstar)  # second derivative matrix

# =============================================================================
# 6. Impose Dirichlet boundary conditions by restricting to the interior nodes.
# =============================================================================
D2_int = D2[1:-1, 1:-1]
V_int = V_total[1:-1]   # effective potential on the interior nodes
rstar_int = rstar[1:-1]

# =============================================================================
# 7. Build the operator A = -D2_int + diag(V_int) and solve the eigenvalue problem.
# =============================================================================
A = -D2_int + np.diag(V_int)
eigvals, eigvecs = eig(A)
eigvals = np.array(eigvals)

# Compute eigenfrequencies: ω is defined by ω = sqrt(λ)
omega_raw = []
for lam in eigvals:
    w = np.sqrt(lam)
    # Ensure we choose the branch with nonnegative real part.
    if w.real < 0:
        w = -w
    omega_raw.append(w)
omega_raw = np.array(omega_raw)

# =============================================================================
# 8. Candidate Selection:
#    We now search for an eigenmode that satisfies a physical criterion.
#    For Schwarzschild (M=1, ℓ=2), published fundamental QNM is near:
#         ω ≈ 0.373 - 0.089i.
#    Here we filter for modes with a real part between, say, 0 and 1 and a small negative imaginary part.
def localization(u, rstar, region=(-1,1)):
    mask = (rstar >= region[0]) & (rstar <= region[1])
    return np.linalg.norm(u[mask])

candidate_idx = None
min_loc = np.inf
for i in range(len(omega_raw)):
    if (0 < omega_raw[i].real < 1) and (omega_raw[i].imag < 0):
        # Pad the eigenfunction to full grid (interior zeros at boundaries).
        u = np.zeros(len(rstar))
        u[1:-1] = eigvecs[:, i].real
        loc_norm = localization(u, rstar, region=(-1,1))
        if loc_norm < min_loc:
            min_loc = loc_norm
            candidate_idx = i

if candidate_idx is not None:
    candidate_mode = omega_raw[candidate_idx]
    print("Candidate fundamental mode found: ω =", candidate_mode)
else:
    candidate_mode = omega_raw[np.argmin(np.abs(omega_raw.real))]
    print("No clear candidate; using smallest real part: ω =", candidate_mode)

# =============================================================================
# 9. (Optional) Plot a few of the eigenfunctions for visual inspection.
# =============================================================================
plt.figure(figsize=(10, 6))
for i in range(5):
    u = np.zeros(len(rstar))
    u[1:-1] = eigvecs[:, i].real
    plt.plot(rstar, u, label=f"Mode {i}")
plt.xlabel("r*")
plt.ylabel("Eigenfunction amplitude")
plt.title("Selected Eigenfunctions (Interior Only)")
plt.legend()
plt.grid(True)
plt.show()

# =============================================================================
# Final: Report the candidate mode.
# =============================================================================
print("Final candidate eigenfrequency: ω =", candidate_mode)
