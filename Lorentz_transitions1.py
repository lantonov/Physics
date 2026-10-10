import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# Define the function f(x) corresponding to the invariant equation
# without quantum corrections:
# f(x) := (\beta*x + c0)^2 - x^2 = 0.
def f_classical(x, beta, c0):
    return (beta*x + c0)**2 - x**2

# Now include a quantum correction.
# For example, let F(x) = k * (1 - x^2) with a small parameter eps.
def f_quantum(x, beta, c0, eps, k):
    # Invariant: (beta*x + c0)^2 + (1-x^2) + eps*k*(1-x^2) = 1, subtract 1:
    return (beta*x + c0)**2 - x**2 + eps * k * (1 - x**2)

# Define parameters for a run.
beta = 0.2           # moderate boost
c0 = 0.5             # offset constant
eps = 0.01           # quantum correction parameter
k = 0.1              # quantum correction strength

# Discretize the domain x in [-1, 1].
x_grid = np.linspace(-1, 1, 1000)

# Find sign changes and use brentq to extract roots for the classical equation.
roots_classical = []
for i in range(len(x_grid)-1):
    if f_classical(x_grid[i], beta, c0) * f_classical(x_grid[i+1], beta, c0) < 0:
        root = brentq(f_classical, x_grid[i], x_grid[i+1], args=(beta, c0))
        roots_classical.append(root)

# Similarly, find roots for the quantum-corrected model.
roots_quantum = []
for i in range(len(x_grid)-1):
    if f_quantum(x_grid[i], beta, c0, eps, k) * f_quantum(x_grid[i+1], beta, c0, eps, k) < 0:
        root = brentq(f_quantum, x_grid[i], x_grid[i+1], args=(beta, c0, eps, k))
        roots_quantum.append(root)

tolerance = 1e-8  # if successive estimates differ by less than tol, treat as degenerate

# In the root-finding loop, check the absolute function value close to the root:
root_candidate = brentq(f_classical, x_grid[i], x_grid[i+1], args=(beta, c0))
if np.abs(f_classical(root_candidate, beta, c0)) < tolerance:
    # Save as double root (or mark as degenerate)
    roots_classical.append(root_candidate)
    print("Degenerate (parabolic) case near x =", root_candidate)



print("Classical roots (x-values):", roots_classical)
print("Quantum-corrected roots (x-values):", roots_quantum)

# Plot the functions for visualization.
plt.figure(figsize=(8, 5))
plt.plot(x_grid, f_classical(x_grid, beta, c0), label="f_classical(x)")
plt.plot(x_grid, f_quantum(x_grid, beta, c0, eps, k), label="f_quantum(x)", linestyle='--')
plt.axhline(0, color='black', lw=1)
plt.xlabel("x = cos(theta)")
plt.ylabel("f(x)")
plt.legend()
plt.title(f"f(x) for beta={beta}, c0={c0} (eps={eps}, k={k})")
plt.show()
