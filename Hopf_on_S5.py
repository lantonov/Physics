import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define a sample of points on S^5 via Hopf fibration over S^2 ⊂ CP^2
# Parametrize S^2 by (theta, phi) and lift to S^5 via Hopf map
theta = np.linspace(0, np.pi, 40)
phi = np.linspace(0, 2 * np.pi, 80)
Theta, Phi = np.meshgrid(theta, phi)

# Base S^2 in CP^2: pick homogeneous coordinates [z1:z2:z3] = [cosθ:sinθ e^{iϕ}:0]
z1 = np.cos(Theta)
z2 = np.sin(Theta) * np.exp(1j * Phi)
z3 = np.zeros_like(z1)

# Now lift to S^5: Hopf fiber in each coordinate direction — introduce a fiber phase ψ
psi_vals = np.linspace(0, 2 * np.pi, 15)
points = []

for ψ in psi_vals:
    phase = np.exp(1j * ψ)
    w1 = z1 * phase
    w2 = z2 * phase
    w3 = z3 * phase
    # Project complex 3D vector to R^3 using real parts of (w1, w2, w3)
    x = np.real(w1)
    y = np.real(w2)
    z = np.imag(w2)
    points.append((x, y, z))

# Plot the projected fibers over S^2 base
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for x, y, z in points:
    ax.plot_surface(x, y, z, color='blue', alpha=0.1, rstride=1, cstride=1, linewidth=0)

ax.set_title("Hopf Fibers over S² ⊂ CP² inside S⁵ (Projected to ℝ³)")
ax.set_xlabel("Re(z₁)")
ax.set_ylabel("Re(z₂)")
ax.set_zlabel("Im(z₂)")
ax.view_init(elev=30, azim=40)
plt.tight_layout()
plt.show()
