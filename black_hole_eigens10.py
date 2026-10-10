import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.special import lambertw

# =============================================================================
# 1. Chebyshev Collocation: Build the Chebyshev differentiation matrix.
# =============================================================================
def cheb(N):
    """
    Compute the Chebyshev differentiation matrix D and grid x (in [-1,1])
    for N+1 points.
    """
    if N == 0:
        return np.array([[0]]), np.array([1])
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1)
    c[0] = 2; c[-1] = 2
    c = c * ((-1) ** np.arange(N + 1))
    X = np.tile(x, (N + 1, 1))
    dX = X - X.T
    D = np.outer(c, 1 / c) / (dX + np.eye(N + 1))
    D = D - np.diag(np.sum(D, axis=1))
    return D, x

# Set the number of Chebyshev points (choose a sufficiently high number)
N = 150
D_cheb, x_cheb = cheb(N)  # x_cheb in [-1,1]

# =============================================================================
# 2. Map the Chebyshev domain to the physical tortoise coordinate domain.
#    Let r* in [a, b]. For example, a = -30, b = 30.
# =============================================================================
rstar_min = -30.0
rstar_max = 30.0

def map_x_to_rstar(x, a, b):
    # Linear mapping: r* = (b-a)/2 * x + (b+a)/2
    return 0.5 * (b - a) * x + 0.5 * (b + a)

rstar_colloc = map_x_to_rstar(x_cheb, rstar_min, rstar_max)

# The differentiation matrix on the physical domain is scaled accordingly:
D_rstar = (2.0 / (rstar_max - rstar_min)) * D_cheb
D2 = np.dot(D_rstar, D_rstar)  # second derivative

# =============================================================================
# 3. Invert the tortoise coordinate to get the radial coordinate r.
#    For Schwarzschild with M=1, we use:
#      r* = r + 2 ln(r/2 - 1)
#    The inversion: r = 2*(1 + W(exp(r*/2 - 1)))
# =============================================================================
def r_of_rstar_scalar(rstar, M=1.0):
    argument = np.exp(rstar/(2*M) - 1)
    w_val = np.real(lambertw(argument))
    return 2 * M * (1 + w_val)

r_values = np.array([r_of_rstar_scalar(rs, M=1.0) for rs in rstar_colloc])

# =============================================================================
# 4. Compute the Regge–Wheeler potential.
#    For Schwarzschild, ℓ = 2:
#      V(r) = (1 - 2M/r)[ℓ(ℓ+1)/r^2 - 6M/r^3]
#    With M=1.
# =============================================================================
def regge_wheeler_potential(r, M=1.0, l=2):
    return (1 - 2 * M / r) * (l * (l + 1) / r**2 - 6 * M / r**3)

l_val = 2
V_RW = regge_wheeler_potential(r_values, M=1.0, l=l_val)
V_eff = V_RW  # We take this as our effective potential.

# =============================================================================
# 5. Optional: Add a complex absorbing potential near the boundaries to mimic
#    outgoing conditions. Here we add an absorber for |r*| >= L_abs.
# =============================================================================
def absorbing_potential(rstar, L_abs=25.0, rstar_max=30.0, W_max=0.005):
    if np.abs(rstar) < L_abs:
        return 0.0
    else:
        return 1j * W_max * ((np.abs(rstar) - L_abs) / (rstar_max - L_abs))**2

W_abs = np.array([absorbing_potential(rs, L_abs=25.0, rstar_max=rstar_max, W_max=0.005) for rs in rstar_colloc])
V_total = V_eff + W_abs

# For visualization:
plt.figure(figsize=(10, 6))
plt.plot(rstar_colloc, np.real(V_total), label="Re(V_total)")
plt.plot(rstar_colloc, np.imag(V_total), label="Im(V_total)", linestyle="--")
plt.xlabel("Tortoise coordinate r*")
plt.ylabel("Effective Potential")
plt.title("Effective Potential (with Absorber) on r* ∈ [{}, {}]".format(rstar_min, rstar_max))
plt.legend()
plt.grid(True)
plt.show()

# =============================================================================
# 6. Impose boundary conditions.
#    We enforce Dirichlet BC (R = 0 at r* = rstar_min and rstar_max) by discarding the endpoints.
# =============================================================================
D2_int = D2[1:-1, 1:-1]  # interior of the differentiation matrix
V_int = V_total[1:-1]    # effective potential at interior nodes

# Construct the operator: A = -D2 + diag(V_int)
A = -D2_int + np.diag(V_int)

# =============================================================================
# 7. Solve the eigenvalue problem.
# =============================================================================
eigvals, eigvecs = eig(A)

# Compute eigenfrequencies as omega = sqrt(lambda)
omega_raw = []
for lam in eigvals:
    w = np.sqrt(lam)
    if w.real < 0:
        w = -w
    omega_raw.append(w)
omega_raw = np.array(omega_raw)

# =============================================================================
# 8. (Dedicated Oversampling/ Spectral) Option:
# If eigenvalues are clustered, one can "oversample" by increasing N or use a spectral method.
# Here, we already used Chebyshev collocation which is spectral.
# You might try a larger N (e.g. N=200 or N=300) to improve resolution.
# =============================================================================

# =============================================================================
# 9. Experiment with boundary conditions:
# In this code we have enforced Dirichlet BC. Alternatively, one may try one-sided differentiation,
# or impose “absorbing” (e.g., non-reflecting) conditions using more advanced methods.
#
# For now, we simply sort the eigenvalues by, say, the absolute value of their imaginary part.
# =============================================================================
indices = np.argsort(np.abs(np.imag(omega_raw)))
omega_sorted = omega_raw[indices]

print("Spectral method computed eigenfrequencies (using Chebyshev collocation):")
for i in range(10):
    print(f"Mode {i}: ω = {omega_sorted[i]}")

# Ideally, for Schwarzschild with M=1, ℓ=2, published QNM for the fundamental mode is roughly:
#   ω_fundamental ≈ 0.373 - 0.089i
# Here you can examine the candidate eigenvalues; with Dirichlet boundary conditions on a compact domain,
# sometimes the discretization captures primarily the potential-dominated mode.
# You may need to adjust N, the domain [rstar_min, rstar_max], or the absorber strength.
# For this illustration, let’s pick the candidate with, say, 0 < Re(ω) < 1 and a small (negative) imaginary part.
fundamental_mode = None
first_overtone = None
for w in omega_sorted:
    if (0 < w.real < 1) and (w.imag < 0):
        if fundamental_mode is None:
            fundamental_mode = w
        elif first_overtone is None:
            first_overtone = w
            break

if fundamental_mode is not None:
    print("\nCandidate Fundamental mode: ω = ", fundamental_mode)
else:
    fundamental_mode = omega_sorted[0]
    print("\nUsing smallest eigenvalue as candidate: ω = ", fundamental_mode)

if first_overtone is not None:
    print("Candidate First overtone: ω = ", first_overtone)
else:
    print("No distinct first overtone found; only using the fundamental mode.")

# =============================================================================
# 10. (Composite Ringdown Signal)
# Construct a composite signal as a sum of damped sinusoids.
# We model the ringdown as:
#     h(t) = A1 exp(-i ω1 t) + A2 exp(-i ω2 t)
# where ω1 is the fundamental mode and ω2 is the first overtone (if available).
# If the overtone is nearly degenerate (or not well separated), you might simply use the published value.
# For illustration, we set:
if fundamental_mode is not None:
    # For demonstration, we use the candidate values if available.
    omega1 = fundamental_mode
else:
    omega1 = 0.373 - 0.089j  # fallback published value for M=1, ℓ=2.
if first_overtone is not None:
    omega2 = first_overtone
else:
    omega2 = 0.372 - 0.20j   # example published overtone value (typically more damped).

A1 = 1.0
A2 = 0.5
phi1 = 0.0
phi2 = 0.0

fs = 1000  # samples per unit time
T_sim = 1.0
t_sim = np.linspace(0, T_sim, int(fs*T_sim), endpoint=False)
h_t = A1*np.exp(-1j*(omega1*t_sim + phi1)) + A2*np.exp(-1j*(omega2*t_sim + phi2))

plt.figure(figsize=(10, 6))
plt.plot(t_sim, np.real(h_t), label="Real part")
plt.plot(t_sim, np.imag(h_t), label="Imaginary part", linestyle="--")
plt.xlabel("Time")
plt.ylabel("h(t)")
plt.title("Composite Ringdown Signal")
plt.legend()
plt.grid(True)
plt.show()

from scipy.signal import periodogram
f_vals, Pxx = periodogram(h_t, fs=fs)
plt.figure(figsize=(10, 6))
plt.semilogy(f_vals, Pxx)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density")
plt.title("Periodogram of the Composite Ringdown Signal")
plt.grid(True)
plt.show()
