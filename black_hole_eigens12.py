import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.special import lambertw

# ------------------------------------------------------------------
# 1. Chebyshev collocation using a linear mapping.
#    We compute the Chebyshev differentiation matrix on [-1, 1]
#    and then map it linearly to the physical tortoise domain [a,b].
# ------------------------------------------------------------------
def cheb(N):
    if N == 0:
        return np.array([[0]]), np.array([1])
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1)
    c[0] = 2; c[-1] = 2
    c = c * ((-1) ** np.arange(N + 1))
    X = np.tile(x, (N + 1, 1))
    dX = X - X.T
    D = np.outer(c, 1/c) / (dX + np.eye(N+1))
    D = D - np.diag(np.sum(D, axis=1))
    return D, x

# Linear mapping from x in [-1,1] to r* in [a, b]
def linear_map(x, a, b):
    return 0.5*(b-a)*x + 0.5*(a+b)

# Choose Chebyshev resolution
N = 150
D_cheb, x_cheb = cheb(N)  # x_cheb in [-1,1]
rstar_min = -8.0   # Try a smaller domain concentrated on the potential peak
rstar_max = 8.0
rstar = linear_map(x_cheb, rstar_min, rstar_max)

# ------------------------------------------------------------------
# 2. Invert r* -> r using the standard analytic inversion for Schwarzschild.
#    For Schwarzschild, r* = r + 2 ln(r/2 - 1).
#    Thus, r = 2 (1 + W(exp(r*/2 - 1)) ).
#    (This function is robust for our chosen domain.)
# ------------------------------------------------------------------
def r_of_rstar(rstar, M=1.0):
    argument = np.exp(rstar/(2*M) - 1)
    x = np.real(lambertw(argument))
    return 2*M * (1 + x)

r_vals = np.array([r_of_rstar(rs, M=1.0) for rs in rstar])

# ------------------------------------------------------------------
# 3. Compute the Regge–Wheeler potential for ℓ=2.
#    V(r) = (1 - 2M/r)[ℓ(ℓ+1)/r^2 - 6M/r^3]
#    with M = 1.
# ------------------------------------------------------------------
def regge_wheeler_potential(r, M=1.0, l=2):
    return (1 - 2*M/r) * (l*(l+1)/r**2 - 6*M/r**3)

l_val = 2
V_RW = regge_wheeler_potential(r_vals, M=1.0, l=l_val)
V_eff = V_RW  # no extra angular term

# ------------------------------------------------------------------
# 4. Add a simple absorbing potential near the boundaries.
#    We turn on the absorber for |r*| >= L_abs.
# ------------------------------------------------------------------
def absorber(rstar, L_abs=8.0, rstar_max=rstar_max, W_max=0.01):
    # Using a linear (or quadratic) ramp for absorption.
    return 1j * W_max * np.maximum(0, (np.abs(rstar)-L_abs)/(rstar_max - L_abs))**2

W_abs = np.array([absorber(rs, L_abs=8.0, rstar_max=rstar_max, W_max=0.01) for rs in rstar])
V_total = V_eff + W_abs

plt.figure(figsize=(8,5))
plt.plot(rstar, np.real(V_total), label="Re(V_total)")
plt.plot(rstar, np.imag(V_total), label="Im(V_total)", linestyle="--")
plt.xlabel("Tortoise coordinate r*")
plt.ylabel("V_total")
plt.title("Effective Potential (Linear Mapping) with Absorber")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------------
# 5. Build the spectral differentiation matrix.
#    We already have D_cheb; now we must scale it to the physical domain.
#    For a linear mapping from [-1,1] to [a,b]:
#         r* = 0.5*(b-a)*x + 0.5*(a+b), so dx/dr* = 2/(b-a)
#    Hence, D_rstar = (2/(b-a)) * D_cheb.
# ------------------------------------------------------------------
scale_factor = 2.0/(rstar_max - rstar_min)
D_rstar = scale_factor * D_cheb
D2 = np.dot(D_rstar, D_rstar)

# ------------------------------------------------------------------
# 6. Impose Dirichlet boundary conditions by retaining interior points.
# ------------------------------------------------------------------
D2_int = D2[1:-1, 1:-1]
V_int = V_total[1:-1]
rstar_int = rstar[1:-1]

# ------------------------------------------------------------------
# 7. Solve the eigenvalue problem.
#    Our operator is: A = -D2_int + diag(V_int)
# ------------------------------------------------------------------
A = -D2_int + np.diag(V_int)
eigvals, eigvecs = eig(A)
eigvals = np.array(eigvals)

# Compute eigenfrequencies (raw), ω = sqrt(λ)
omega_raw = []
for lam in eigvals:
    w = np.sqrt(lam)
    if w.real < 0:
        w = -w
    omega_raw.append(w)
omega_raw = np.array(omega_raw)

# ------------------------------------------------------------------
# 8. Filter candidate modes by requiring the eigenfunction be localized
#    in the region where the potential is active, e.g., |r*| < 3.
def localization(u, rstar, region=(-3,3)):
    mask = (rstar >= region[0]) & (rstar <= region[1])
    return np.linalg.norm(u[mask])

candidate_idx = None
min_loc_norm = np.inf
for i in range(len(omega_raw)):
    # Pick modes with physically reasonable frequencies, e.g., 0 < Re(ω) < 1 and negative Im(ω).
    if 0 < omega_raw[i].real < 1 and omega_raw[i].imag < 0:
        u = np.zeros(len(rstar))
        u[1:-1] = eigvecs[:, i].real  # use the real part for localization
        norm_loc = localization(u, rstar, region=(-3,3))
        if norm_loc < min_loc_norm:
            min_loc_norm = norm_loc
            candidate_idx = i

if candidate_idx is not None:
    candidate_mode = omega_raw[candidate_idx]
    print("Candidate fundamental mode (spectral) found: ω =", candidate_mode)
else:
    candidate_mode = omega_raw[np.argmin(np.abs(omega_raw.real))]
    print("No clear candidate based on localization; using smallest real part: ω =", candidate_mode)

# At this point, if candidate_mode ≈ 0.373 - 0.089j, we're in good shape.
# Otherwise, further adjustments of the domain, resolution, or absorber parameters are needed.

# ------------------------------------------------------------------
# 9. (Optional) Plot a few eigenfunctions for inspection.
# ------------------------------------------------------------------
plt.figure(figsize=(10, 6))
for i in range(5):
    u = np.zeros(len(rstar))
    u[1:-1] = eigvecs[:, i].real
    plt.plot(rstar, u, label=f"Mode {i}")
plt.xlabel("r*")
plt.ylabel("Eigenfunction amplitude")
plt.title("First Few Eigenfunctions from Spectral Method")
plt.legend()
plt.grid(True)
plt.show()
