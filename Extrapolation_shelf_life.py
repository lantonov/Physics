import numpy as np
import matplotlib.pyplot as plt

# Simulated extrapolation distance (difference from mean observed temp, in K^-1 units)
d = np.linspace(0, 0.0025, 50)

# Simulated uncertainty growth
# Linear (fixed slope, weak propagation)
width_linear = 0.5 + 40 * d

# Hierarchical nonlinear (full uncertainty propagation)
width_hier = 0.5 + 40 * d + 300 * d**2

plt.figure()
plt.plot(d, width_linear, label="Linear Arrhenius (fixed slope)")
plt.plot(d, width_hier, label="Hierarchical non-linear Arrhenius")
plt.xlabel("Extrapolation distance d = |1/T_ref − 1/T|")
plt.ylabel("Shelf-life distribution width (months)")
plt.legend()
plt.show()
