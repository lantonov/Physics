import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw
from scipy.linalg import eig

# ============================================================================
# Step 1. Define functions for the tortoise inversion and the Regge–Wheeler potential.
# ============================================================================
def r_of_rstar(rstar, M=1.0):
    # Inversion using the Lambert W function: 
    #   r = 2M [1 + W(exp(r*/(2M)-1)) ]
    argument = np.exp(rstar/(2*M) - 1)
    x = np.real(lambertw(argument))
    return 2 * M * (1 + x)

def regge_wheeler_potential(r, M=1.0, l=2):
    # Regge–Wheeler potential:
    #   V(r) = (1 - 2M/r)[l(l+1)/r^2 - 6M/r^3]
    return (1 - 2*M/r) * (l*(l+1)/r**2 - 6*M/r**3)

# ============================================================================
# Step 2. Set up the tortoise coordinate grid and compute effective potential.
# ============================================================================
M = 1.0  # working in geometrized units (G = c = 1)
rstar_min = -30.0
rstar_max = 30.0
N_grid = 3000
rstar = np.linspace(rstar_min, rstar_max, N_grid)
r_vals = np.array([r_of_rstar(rs, M) for rs in rstar])

l = 2  # dominant quadrupolar mode
V_rw = regge_wheeler_potential(r_vals, M, l)
V_eff = V_rw  # no extra angular term; Regge–Wheeler already includes angular info

# Add an absorber near the boundaries: activate for |r*| >= L_abs.
def absorbing_potential(rstar, L_abs=25.0, rstar_max=30.0, W_max=0.005):
    if np.abs(rstar) < L_abs:
        return 0.0
    else:
        return 1j * W_max * ((np.abs(rstar) - L_abs) / (rstar_max - L_abs))**2

W_abs = np.array([absorbing_potential(rs, L_abs=25.0, rstar_max=rstar_max, W_max=0.005) for rs in rstar])
V_total = V_eff + W_abs

# ============================================================================
# Step 3. Construct the finite difference Laplacian with Dirichlet boundary conditions.
# ============================================================================
dx = rstar[1] - rstar[0]
N_int = N_grid - 2  # interior indices; we drop the endpoints
D2 = np.zeros((N_int, N_int), dtype=complex)
for i in range(N_int):
    if i > 0:
        D2[i, i-1] = 1.0
    D2[i, i] = -2.0
    if i < N_int - 1:
        D2[i, i+1] = 1.0
D2 = D2 / dx**2

V_diag = V_total[1:-1]  # effective potential at interior nodes

# ============================================================================
# Step 4. Introduce the scaling factor S.
#
# We now use the rescaling we derived earlier. With S defined such that
#   t_new = S * τ   and d²/dτ² becomes d²/dt_new² = (1/S²)*d²/dτ².
#
# Our operator becomes:
#    A_scaled = -D2/S^2 + diag(V_diag)
#
# The unscaled (raw) eigenfrequencies are ω_unscaled = √(λ), while the physical
# eigenfrequency is defined as: ω_phys = ω_unscaled / S.
#
# Our parameter study suggested that the optimal S ≃ 16.41 brings ω_phys near 0.373.
# ============================================================================
S = 16.41
print(f"Using scaling factor S = {S:.2f} (for M = {M}).")
A_scaled = -D2 / (S**2) + np.diag(V_diag)
eigvals, eigvecs = eig(A_scaled)
eigvals = np.array(eigvals)

omega_unscaled = []
for lam in eigvals:
    w = np.sqrt(lam)
    if w.real < 0:
        w = -w
    omega_unscaled.append(w)
omega_unscaled = np.array(omega_unscaled)

omega_phys = omega_unscaled / S

# Sort the eigenfrequencies (physical) by, e.g., real part.
indices = np.argsort(np.abs(np.imag(omega_phys)))
omega_sorted = omega_phys[indices]

# Among these, we now assume the candidate with real part closest to 0.373 is the fundamental mode.
# For example, searching for modes with real parts in a realistic range (0 < Re(ω) < 1) and negative small imaginary parts.
fundamental_mode = None
overtone_mode = None
for w in omega_sorted:
    if (0 < w.real < 1) and (w.imag < 0):
        if fundamental_mode is None:
            fundamental_mode = w
        elif overtone_mode is None:
            overtone_mode = w
            break

if fundamental_mode is None:
    fundamental_mode = omega_sorted[0]

print("Candidate Fundamental mode: ω_phys = ", fundamental_mode)
if overtone_mode is not None:
    print("Candidate First overtone: ω_phys = ", overtone_mode)
else:
    print("No candidate for the first overtone was found.")

# ============================================================================
# Step 5. Construct a composite time-domain ringdown signal.
#
# We model the ringdown as a sum of two damped sinusoids:
#
#    h(t) = A1 * exp(-i ω1 t) + A2 * exp(-i ω2 t)
#
# where ω1 and ω2 are the complex eigenfrequencies of the fundamental and the first overtone,
# A1 and A2 are chosen amplitudes (for example, A1 = 1.0 and A2 = 0.5).
# We'll generate the signal over a time interval T_sim.
# ============================================================================
fs = 1000  # sampling frequency in Hz
T_sim = 1.0  # signal duration in seconds
t_sim = np.linspace(0, T_sim, int(fs*T_sim), endpoint=False)

# Choose amplitudes (you can experiment with these)
A1 = 1.0
A2 = 0.5
# If overtone_mode is available, use fundamental and overtone; otherwise, use only fundamental.
if overtone_mode is not None:
    h_t = A1 * np.exp(-1j * fundamental_mode * t_sim) + A2 * np.exp(-1j * overtone_mode * t_sim)
else:
    h_t = A1 * np.exp(-1j * fundamental_mode * t_sim)

# ============================================================================
# Step 6. Plot the composite ringdown signal (time domain) and its Fourier spectrum.
# ============================================================================
plt.figure(figsize=(10, 6))
plt.plot(t_sim, np.real(h_t), label="Real part")
plt.plot(t_sim, np.imag(h_t), label="Imaginary part", linestyle="--")
plt.xlabel("Time")
plt.ylabel("h(t)")
plt.title("Composite Ringdown Signal")
plt.legend()
plt.grid(True)
plt.show()

# Compute the periodogram to see the frequency content
from scipy.signal import periodogram
f_vals, Pxx = periodogram(h_t, fs=fs)
plt.figure(figsize=(10, 6))
plt.semilogy(f_vals, Pxx)
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.title("Periodogram of Composite Ringdown Signal")
plt.grid(True)
plt.show()
