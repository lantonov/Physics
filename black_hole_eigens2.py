import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt

# Physical constants (SI units)
G = 6.67e-11       # m^3/(kg*s^2)
c = 3e8            # m/s

# Black hole mass and derived period T (light-crossing time at Schwarzschild radius)
M = 6e31           # kg, example mass
T = 2 * G * M / c**3  # seconds

# Number of discretization points for the eigenvalue solver
N = 1000
dtau = T / N

# Time grid over one period
tau = np.linspace(0, T, N, endpoint=False)

def construct_operator(V0, V1, ell, M, T, N, dtau):
    """
    Constructs the finite-difference operator for L = -d^2/dtau^2 + V_eff(tau)
    with an effective potential that includes a periodic modulation and a spherical-harmonics term.
    
    V_eff(tau) = V0 + V1 * cos(2*pi*tau/T) + [ell(ell+1) * c^4 / (2GM)^2]
    
    Periodic boundary conditions are imposed.
    """
    # Spherical harmonic contribution for mode ell
    V_ang = (ell * (ell + 1) * c**4) / ( (2 * G * M)**2 )
    # Construct effective potential at each tau
    V_eff = V0 + V1 * np.cos(2 * np.pi * tau / T) + V_ang
    V_matrix = np.diag(V_eff)
    
    # Finite-difference second derivative with periodic BC
    D2 = np.zeros((N, N))
    for i in range(N):
        D2[i, i] = -2.0
        D2[i, (i+1) % N] = 1.0
        D2[i, (i-1) % N] = 1.0
    D2 = D2 / dtau**2

    # Construct the operator: L = -D2 + V_eff(tau)
    L = -D2 + V_matrix
    return L

def compute_eigensystem(L):
    """Compute eigenvalues and return the sorted eigenvalues and eigenvectors."""
    eigenvals, eigenvecs = eig(L)
    eigenvals = np.real(eigenvals)  # Discard tiny imaginary parts
    sorted_indices = np.argsort(eigenvals)
    return eigenvals[sorted_indices], eigenvecs[:, sorted_indices]

def compute_frequencies(eigenvals):
    """
    Return oscillation frequencies (Hz) computed from eigenvalues.
    Frequency (in Hz) for a mode is f = sqrt(lambda)/(2*pi).
    """
    return np.sqrt(eigenvals) / (2 * np.pi)

# Naive prediction (without additional potential or angular contribution) for comparison:
f_naive = c**3 / (2 * G * M)
print("Naive prediction f ~ c^3/(2GM): {:.3f} Hz".format(f_naive))

# --- Parameter Study ---
# We choose ranges for V0 and V1 (units 1/s^2). Here we use arbitrary ranges
V0_vals = np.linspace(0, 1e10, 10)      # baseline potential component
V1_vals = np.linspace(-5e9, 5e9, 10)      # modulation amplitude

# We also choose an angular parameter; for a quadrupole mode we set ell=2.
ell = 2

# Prepare arrays to store the fundamental frequencies and inferred dimensionless factor kappa.
fundamental_frequencies = np.zeros((len(V0_vals), len(V1_vals)))
kappa_matrix = np.zeros((len(V0_vals), len(V1_vals)))

for i, V0 in enumerate(V0_vals):
    for j, V1 in enumerate(V1_vals):
        L = construct_operator(V0, V1, ell, M, T, N, dtau)
        eigenvals, _ = compute_eigensystem(L)
        # Assume eigenvals[0] might be a trivial zero/constant mode; take the first nonzero one.
        # (In practice, check the eigenvalue magnitude.)
        fundamental_lambda = eigenvals[1]
        f_dom = np.sqrt(fundamental_lambda) / (2*np.pi)
        fundamental_frequencies[i, j] = f_dom
        # Determine the additional dimensionless factor kappa, defined so that:
        # f = (kappa/(2*pi))*(c^3/(G*M))
        kappa_matrix[i, j] = (2*np.pi * f_dom * G * M) / c**3

# Plot kappa versus V0 for several fixed V1 values.
plt.figure(figsize=(10, 6))
for j in range(len(V1_vals)):
    plt.plot(V0_vals, kappa_matrix[:, j], 'o-', label=f"V1 = {V1_vals[j]:.1e} 1/s^2")
plt.xlabel("V0 (1/s^2)")
plt.ylabel("Dimensionless factor kappa")
plt.title("Dependence of kappa on V0 (for fixed V1) [ell = {}]".format(ell))
plt.legend()
plt.grid(True)
plt.show()

# Plot kappa versus V1 for several fixed V0 values.
plt.figure(figsize=(10, 6))
for i in range(len(V0_vals)):
    plt.plot(V1_vals, kappa_matrix[i, :], 's--', label=f"V0 = {V0_vals[i]:.1e} 1/s^2")
plt.xlabel("V1 (1/s^2)")
plt.ylabel("Dimensionless factor kappa")
plt.title("Dependence of kappa on V1 (for fixed V0) [ell = {}]".format(ell))
plt.legend()
plt.grid(True)
plt.show()

# --- Modeling Two Characteristic Peaks ---
# Here we assume that our eigenvalue spectrum has at least two non-trivial modes.
# We now choose a fixed set of V0 and V1 that yield two distinct eigenvalues.
V0_fixed = 5e9
V1_fixed = 0  # For clarity, start with no modulation; later you can add modulation.
L = construct_operator(V0_fixed, V1_fixed, ell, M, T, N, dtau)
eigenvals, evecs = compute_eigensystem(L)
frequencies = compute_frequencies(eigenvals)

# Display the first several frequencies.
print("The first 5 frequencies (Hz):", frequencies[:5])

# For our composite model, we take the first (n=1) and second (n=2) nontrivial modes.
if len(frequencies) >= 3:
    f1 = frequencies[1]
    f2 = frequencies[2]
    print("Dominant frequency f1: {:.3f} Hz".format(f1))
    print("Secondary frequency f2: {:.3f} Hz".format(f2))
else:
    print("Not enough modes computed.")

# Simulate a composite signal (as in a black hole collision) using these two modes.
fs_sim = 10000  # simulation sampling frequency
T_sim = 1.0     # simulation duration in seconds
t_sim = np.linspace(0, T_sim, int(fs_sim * T_sim), endpoint=False)
A1 = 1.0
A2 = 0.5
signal_composite = A1 * np.sin(2*np.pi * f1 * t_sim) + A2 * np.sin(2*np.pi * f2 * t_sim)

# Plot a short portion of the composite signal.
plt.figure(figsize=(10, 6))
plt.plot(t_sim[:1000], signal_composite[:1000])
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Composite Signal: Two Characteristic Peaks (f1 and f2)")
plt.show()

# Plot the periodogram of the composite signal.
from scipy.signal import periodogram, find_peaks
f_vals, Pxx = periodogram(signal_composite, fs_sim)
plt.figure(figsize=(10, 6))
plt.semilogy(f_vals, Pxx, label="PSD")
peaks, props = find_peaks(Pxx, height=np.max(Pxx)*0.1)
plt.scatter(f_vals[peaks], Pxx[peaks], color='red', zorder=5, label="Detected Peaks")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.title("Periodogram of the Composite Signal")
plt.legend()
plt.show()
