import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt

# Physical and simulation parameters
G = 6.67e-11       # m^3/kg/s^2
c = 3e8            # m/s
M = 6e31           # Example mass in kg
T = 2 * G * M / c**3  # Light-crossing time at Schwarzschild radius (s)
N = 1000           # Number of discretization points
dtau = T / N

# Define the constant effective potential V_eff (in s^-2 units)
# V0 is chosen so that dimensions match: V0 ~ (frequency)^2.
# For demonstration, we can take V0 = alpha*(2*pi/T)^2. For example:
alpha = 0.1
V0 = alpha * (2 * np.pi / T)**2

# Construct the second derivative finite difference matrix with periodic boundary conditions
D2 = np.zeros((N, N))
for i in range(N):
    D2[i, i] = -2.0
    D2[i, (i+1) % N] = 1.0
    D2[i, (i-1) % N] = 1.0
D2 = D2 / dtau**2

# The full operator L = -D2 + V0 * I.
L = -D2 + V0 * np.eye(N)

# Solve the eigenvalue problem L psi = lambda psi.
eigenvalues, eigenvectors = eig(L)
eigenvalues = np.real(eigenvalues)

# Sort eigenvalues
sorted_indices = np.argsort(eigenvalues)
eigenvalues = eigenvalues[sorted_indices]

# The fundamental (lowest non-zero) eigenvalue gives us the mode frequency.
# (Often the zero eigenvalue corresponds to a trivial gauge mode.)
lambda_dom = eigenvalues[1]  # assuming eigenvalues[0] is near zero
omega_dom = np.sqrt(lambda_dom)
f_dom = omega_dom / (2 * np.pi)

print("Dominant eigenvalue (λ):", lambda_dom)
print("Dominant angular frequency (ω):", omega_dom)
print("Dominant oscillation frequency, f = ω/(2π):", f_dom, "Hz")

# Compare with the naive prediction
f_naive = c**3 / (2 * G * M)
print("Naive prediction f ~ c^3/(2GM):", f_naive, "Hz")

# Plot a few eigenfunctions to see the spatial structure
plt.figure(figsize=(10, 6))
for i in range(3):
    plt.plot(np.linspace(0, T, N), eigenvectors[:, sorted_indices[i]], label=f"Mode {i}")
plt.xlabel("τ (s)")
plt.ylabel("ψ(τ)")
plt.title("Representative Eigenfunctions")
plt.legend()
plt.show()
