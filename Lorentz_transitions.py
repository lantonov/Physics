import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq  # a robust bisection method

# Parameters
beta = 0.2  # boost parameter
c0 = 0.5    # offset constant
eps = 0.01  # quantum correction small parameter
k = 0.1     # quantum correction strength

# Define the function f(x) corresponding to the modified equation:
def f(x, beta, c0, eps, k):
    return (beta*x + c0)**2 - (1 - eps*k)*x**2 - eps*k

# Create an x-grid
x_vals = np.linspace(-1, 1, 500)

# Find roots in subintervals where the sign of f(x) changes
roots = []
for i in range(len(x_vals)-1):
    if f(x_vals[i], beta, c0, eps, k) * f(x_vals[i+1], beta, c0, eps, k) < 0:
        root = brentq(f, x_vals[i], x_vals[i+1], args=(beta, c0, eps, k))
        roots.append(root)

print("Found roots (x-values):", roots)

# Plot f(x) for visualization
plt.plot(x_vals, f(x_vals, beta, c0, eps, k), label='f(x)')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('x = cos(theta)')
plt.ylabel('f(x)')
plt.title('Modified invariant equation for beta={} c0={} eps={}, k={}'.format(beta, c0, eps, k))
plt.legend()
plt.show()

T_vals = np.linspace(0, 2, 50)  # transition time from S1^3 (T=0) to S2^3 (T=2)

# For each T, define beta(T) and c0(T) as a simple interpolation:
beta_T = 0.2 + 0.4 * T_vals / 2.0    # for example, beta goes from 0.2 to 0.6
c0_T = 0.5 + 0.3 * T_vals / 2.0        # c0 goes from 0.5 to 0.8

# For each T, find roots and store them. Plot evolution.
roots_vs_T = []
for i, T in enumerate(T_vals):
    b = beta_T[i]
    c_val = c0_T[i]
    # Find roots for current T
    current_roots = []
    for j in range(len(x_vals)-1):
        if f(x_vals[j], b, c_val, eps, k) * f(x_vals[j+1], b, c_val, eps, k) < 0:
            root = brentq(f, x_vals[j], x_vals[j+1], args=(b, c_val, eps, k))
            current_roots.append(root)
    roots_vs_T.append(current_roots)

# One may now produce a scatter plot with T along the horizontal axis
for i, roots in enumerate(roots_vs_T):
    T_cur = T_vals[i]
    for r in roots:
        plt.scatter(T_cur, r, color='blue')
plt.xlabel('Transition parameter T')
plt.ylabel('x = cos(theta)')
plt.title('Evolution of allowed x-values vs. transition time')
plt.show()
