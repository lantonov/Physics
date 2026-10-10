import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Generate Hopf fibers on S^3 projected to R^3
phi = np.linspace(0, 2 * np.pi, 400)
alpha = np.linspace(-np.pi / 2, np.pi / 2, 100)
Phi, Alpha = np.meshgrid(phi, alpha)

# Parameters for a Hopf torus
r_base = 1
fiber_radius = 0.3

# Parametric equations for a Hopf torus
X = (r_base + fiber_radius * np.cos(2 * Phi)) * np.cos(2 * Alpha)
Y = (r_base + fiber_radius * np.cos(2 * Phi)) * np.sin(2 * Alpha)
Z = fiber_radius * np.sin(2 * Phi)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, color='cyan', alpha=0.6, edgecolor='k')
ax.set_title('Hopf Fiber with 2+2+2 Winding Structure')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()