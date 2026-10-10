import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt

# Constants (SI units)
G = 6.67e-11       # m^3/(kg*s^2)
c = 3e8            # m/s

# Black hole mass and derived period T (light-crossing time at Schwarzschild radius)
M = 6e31           # kg
T = 2 * G * M / c**3  # seconds

# Number of discretization points
N = 1000
dtau = T / N

# Define the time grid
tau = np.linspace(0, T, N, endpoint=False)

def construct_operator(V0, V1, T, N, dtau):
    """
    Constructs the finite difference matrix for L = -d^2/dtau^2 + V_eff(tau),
    where:
      V_eff(tau) = V0 + V1 * cos(2*pi*tau/T).
    Periodic boundary conditions are enforced.
    """
    # Create effective potential as a diagonal matrix:
    V_eff = V0 + V1 * np.cos(2 * np.pi * tau / T)
    V_matrix = np.diag(V_eff)
    
    # Construct finite difference approximation for second derivative (D2)
    D2 = np.zeros((N, N))
    for i in range(N):
        D2[i, i] = -2.0
        D2[i, (i+1) % N] = 1.0
        D2[i, (i-1) % N] = 1.0
    D2 = D2 / dtau**2

    # Build full operator L = -D2 + V_matrix
    L = -D2 + V_matrix
    return L

# Function to compute fundamental frequency from operator L
def compute_frequency(L):
    eigenvals, _ = eig(L)
    eigenvals = np.real(eigenvals)
    eigenvals.sort()
    # Assume the first eigenvalue is near zero (or trivial); take the second smallest.
    lambda_dom = eigenvals[1]
    omega_dom = np.sqrt(lambda_dom)
    f_dom = omega_dom / (2 * np.pi)
    return f_dom, lambda_dom

# Naive prediction (without additional potential) for comparison:
f_naive = c**3 / (2 * G * M)
print("Naive prediction f ~ c^3/(2GM):", f_naive, "Hz")

# Parameter study: vary V0 and V1
V0_vals = np.linspace(0, 1e10, 10)   # Adjust range and units (units: 1/s^2)
V1_vals = np.linspace(-5e9, 5e9, 10)   # Adjust range as needed
kappa_matrix = np.zeros((len(V0_vals), len(V1_vals)))
f_matrix = np.zeros((len(V0_vals), len(V1_vals)))

for i, V0 in enumerate(V0_vals):
    for j, V1 in enumerate(V1_vals):
        L = construct_operator(V0, V1, T, N, dtau)
        f_dom, lam_dom = compute_frequency(L)
        f_matrix[i, j] = f_dom
        # Compute the dimensionless factor: if f = (kappa/2pi)*(c^3/(GM)), then:
        kappa_matrix[i, j] = (2 * np.pi * f_dom * G * M) / (c**3)

# Plot the dependence of kappa on V0 and V1.
plt.figure(figsize=(10, 6))
for j, V1 in enumerate(V1_vals):
    plt.plot(V0_vals, kappa_matrix[:, j], label=f"V1 = {V1:.1e} 1/s^2")
plt.xlabel("V0 (1/s^2)")
plt.ylabel("Dimensionless factor kappa")
plt.title("Dependence of kappa on V0 (for various V1)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
for i, V0 in enumerate(V0_vals):
    plt.plot(V1_vals, kappa_matrix[i, :], label=f"V0 = {V0:.1e} 1/s^2")
plt.xlabel("V1 (1/s^2)")
plt.ylabel("Dimensionless factor kappa")
plt.title("Dependence of kappa on V1 (for various V0)")
plt.legend()
plt.grid(True)
plt.show()

# Optionally, plot a surface plot of f_dom vs V0 and V1.
from mpl_toolkits.mplot3d import Axes3D
V0_grid, V1_grid = np.meshgrid(V0_vals, V1_vals)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(V0_grid, V1_grid, kappa_matrix.T, cmap='viridis')
ax.set_xlabel("V0 (1/s^2)")
ax.set_ylabel("V1 (1/s^2)")
ax.set_zlabel("kappa")
ax.set_title("3D plot: kappa vs V0 and V1")
plt.show()
