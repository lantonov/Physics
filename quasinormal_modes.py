# Θ⁺-consistent QNM analysis:
# 1) Singular spectrum computation
# 2) Automatic n_max selection
# 3) Overlay Θ⁺ S^3 spectrum
# 4) Demonstrate overfitting instability

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, eig

# ----------------------------
# Utilities
# ----------------------------

def hankel_matrix(h, L):
    N = len(h)
    return np.column_stack([h[i:i+L] for i in range(N-L)])

def singular_spectrum(h, L):
    H = hankel_matrix(h, L)
    _, S, _ = svd(H, full_matrices=False)
    return S

def auto_nmax(S, threshold=1e-2):
    """
    Select effective rank where singular values
    drop below threshold * max(S)
    """
    S_norm = S / S[0]
    return np.sum(S_norm > threshold)

# ----------------------------
# Matrix Pencil with chosen n_max
# ----------------------------

def matrix_pencil(t, h, n_modes, pencil_frac=0.5):
    dt = t[1] - t[0]
    N = len(h)
    L = int(pencil_frac * N)

    # Number of columns for Hankel matrices
    M = N - L

    if M <= 0:
        raise ValueError("Pencil dimension L is too large (N - L <= 0)")
    if L <= n_modes:
        raise ValueError("Pencil dimension L too small for requested n_modes")
    if M < n_modes:
        raise ValueError("Number of columns (N-L) is too small for requested n_modes")

    # Correct construction to ensure Y0 and Y1 have the same number of columns
    Y0 = np.column_stack([h[i:i+L] for i in range(M)])
    Y1 = np.column_stack([h[i+1:i+L+1] for i in range(M)])

    U, S, Vh = svd(Y0, full_matrices=False)
    Ur = U[:, :n_modes]
    Sr = np.diag(S[:n_modes])
    Vr = Vh.conj().T[:, :n_modes]

    A = np.linalg.pinv(Sr) @ Ur.conj().T @ Y1 @ Vr
    eigvals, _ = eig(A)

    omega = np.log(eigvals) / dt
    freqs = np.real(omega) / (2*np.pi)
    taus = -1 / np.imag(omega)

    return freqs, taus

# ----------------------------
# Θ⁺ S³ spectrum
# ----------------------------

def theta_plus_s3_spectrum(n_max, R, c=1.0, gamma=0.05):
    n = np.arange(1, n_max+1)
    omega_R = c / R * np.sqrt(n*(n+2))
    omega_I = -gamma * n
    return omega_R, omega_I

# ----------------------------
# Synthetic ringdown data
# ----------------------------

np.random.seed(1)
t = np.linspace(0, 0.05, 4000)
true_freqs = np.array([180, 260])
true_taus = np.array([0.02, 0.012])

h = (
    np.exp(2j*np.pi*true_freqs[0]*t - t/true_taus[0]) +
    0.7*np.exp(2j*np.pi*true_freqs[1]*t - t/true_taus[1])
).real

h += 0.03*np.random.randn(len(t))

# ----------------------------
# 1) Singular spectrum
# ----------------------------

L = 600
S = singular_spectrum(h, L)
n_eff = auto_nmax(S)

plt.figure()
plt.semilogy(S / S[0], 'o-')
plt.axvline(n_eff-1, color='r', linestyle='--', label=f'n_max = {n_eff}')
plt.xlabel("Index")
plt.ylabel("Normalized singular value")
plt.title("Singular value spectrum (mode count diagnostic)")
plt.legend()
plt.show()

# ----------------------------
# 2) Extract modes for n_max and n_max+1
# ----------------------------

freqs_ok, taus_ok = matrix_pencil(t, h, n_eff)
freqs_bad, taus_bad = matrix_pencil(t, h, n_eff + 1)

# ----------------------------
# 3) Θ⁺ spectrum overlay
# ----------------------------

R_eff = 0.005
omega_R, omega_I = theta_plus_s3_spectrum(n_eff, R_eff)

plt.figure()
plt.scatter(freqs_ok, 1/taus_ok, label="Extracted modes", zorder=3)
plt.plot(omega_R/(2*np.pi), -omega_I, 'x', label="Θ⁺ S³ prediction")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Damping rate [1/s]")
plt.title("Θ⁺ spectrum vs extracted QNMs")
plt.legend()
plt.show()

# ----------------------------
# 4) Overfitting demonstration
# ----------------------------

plt.figure()
plt.scatter(freqs_ok, taus_ok, label=f'n = {n_eff}', zorder=3)
plt.scatter(freqs_bad, taus_bad, label=f'n = {n_eff+1}', marker='x')
plt.xlabel("Frequency [Hz]")
plt.ylabel("Tau [s]")
plt.title("Adding one mode destabilizes the fit")
plt.legend()
plt.show()

n_eff