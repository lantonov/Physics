import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.special import lambertw

# Example: Use a nonlinear mapping from Chebyshev nodes to the physical domain.
def cheb(N):
    if N == 0:
        return np.array([[0]]), np.array([1])
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1)
    c[0], c[-1] = 2, 2
    c = c * ((-1) ** np.arange(N + 1))
    X = np.tile(x, (N + 1, 1))
    dX = X - X.T
    D = np.outer(c, 1 / c) / (dX + np.eye(N + 1))
    D = D - np.diag(np.sum(D, axis=1))
    return D, x

# Use Chebyshev collocation with a nonlinear mapping.
def map_nonlinear(x, a, b):
    # Instead of a linear mapping, use a map that clusters points near 0.
    # For example: r* = a + (b-a) * ( (arctanh(x))/arctanh(1) + 1 )/2.
    return a + (b - a) * ((np.arctanh(x))/(np.arctanh(0.9999)) + 1)/2

N = 150
D_cheb, x_cheb = cheb(N)
rstar_min = -10.0
rstar_max = 10.0
# Use a nonlinear mapping to cluster points near 0 (where potential peaks).
rstar_colloc = map_nonlinear(x_cheb, rstar_min, rstar_max)

# Inversion from rstar to r
def r_of_rstar_scalar(rstar, M=1.0):
    argument = np.exp(rstar/(2*M) - 1)
    return 2 * M * (1 + np.real(lambertw(argument)))

r_vals = np.array([r_of_rstar_scalar(rs, M=1.0) for rs in rstar_colloc])

# Define the Regge–Wheeler potential for ℓ = 2:
def regge_wheeler_potential(r, M=1.0, l=2):
    return (1 - 2*M/r) * (l*(l+1)/r**2 - 6*M/r**3)

l_val = 2
V_RW = regge_wheeler_potential(r_vals, M=1.0, l=l_val)
V_eff = V_RW

# Add an absorber for |r*| > L_abs.
def absorber(rstar, L_abs=7.0, rstar_max=rstar_max, W_max=0.01):
    return 1j * W_max * np.maximum(0, (np.abs(rstar) - L_abs)/(rstar_max - L_abs))**2

W_abs = absorber(rstar_colloc, L_abs=7.0, rstar_max=rstar_max, W_max=0.01)
V_total = V_eff + W_abs

plt.figure(figsize=(8,5))
plt.plot(rstar_colloc, np.real(V_total), label="Re(V_total)")
plt.plot(rstar_colloc, np.imag(V_total), label="Im(V_total)", linestyle="--")
plt.xlabel("r*")
plt.ylabel("V_total")
plt.legend()
plt.grid(True)
plt.title("Effective Potential with Nonlinear Mapping & Absorber")
plt.show()

# Build Chebyshev differentiation matrix on the nonlinear grid.
# Here D_cheb is already computed for the Chebyshev nodes;
# we need to scale it by the derivative of the mapping.
# For simplicity, compute finite differences in the rstar_colloc array:
dx = np.diff(rstar_colloc).mean()  # approximate
N_grid = len(rstar_colloc)
# Form a finite-difference matrix with Dirichlet BC (drop first and last row).
D2_fd = np.zeros((N_grid-2, N_grid-2))
for i in range(1, N_grid-1):
    # Use central finite differences
    if i-1 > 0:
        D2_fd[i-1, i-2] = 1.0
    D2_fd[i-1, i-1] = -2.0
    if i < N_grid-2:
        D2_fd[i-1, i] = 1.0
D2_fd /= dx**2

V_int = V_total[1:-1]
A = -D2_fd + np.diag(V_int)

eigvals, eigvecs = eig(A)
eigvals = np.array(eigvals)

# Compute eigenfrequencies: ω = sqrt(λ)
omega_raw = []
for lam in eigvals:
    w = np.sqrt(lam)
    if w.real < 0:
        w = -w
    omega_raw.append(w)
omega_raw = np.array(omega_raw)

# Now select the candidate eigenmode by checking the localization of the eigenfunction.
# For instance, compute the norm of the eigenfunction in the region where |r*|<3.
def localization(u, rstar, region=(-3,3)):
    mask = (rstar >= region[0]) & (rstar <= region[1])
    return np.linalg.norm(u[mask])

candidate_idx = None
min_norm_diff = np.inf
for i in range(len(omega_raw)):
    # Consider only eigenmodes with positive real part and a small negative imaginary part:
    if 0 < omega_raw[i].real < 1 and omega_raw[i].imag < 0:
        # Interpolate eigenfunction onto full rstar_colloc.
        u = np.zeros(N_grid, dtype=complex)
        u[1:-1] = eigvecs[:, i]
        loc_norm = localization(u, rstar_colloc, region=(-3,3))
        # We want the mode that is most concentrated in the active region.
        if loc_norm < min_norm_diff:
            min_norm_diff = loc_norm
            candidate_idx = i

if candidate_idx is not None:
    candidate_mode = omega_raw[candidate_idx]
    print("Candidate fundamental mode (localized eigenfunction): ω =", candidate_mode)
else:
    candidate_mode = omega_raw[np.argmin(np.abs(omega_raw.real))]
    print("No clear candidate found; using smallest real part: ω =", candidate_mode)
