# Conformal Villarceau Rotor Simulation

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters for the torus
R = 2  # major radius
r = 1  # minor radius

# Define the rotor angles (base and fiber contributions)
phi = np.linspace(0, 2 * np.pi, 400)  # base (horizontal)
psi = np.linspace(0, 2 * np.pi, 400)  # fiber (vertical)

Phi, Psi = np.meshgrid(phi, psi)

# Torus coordinates (composite Villarceau rotor)
X = (R + r * np.cos(Phi)) * np.cos(Psi)
Y = (R + r * np.cos(Phi)) * np.sin(Psi)
Z = r * np.sin(Phi)

# Plotting the torus and Villarceau circles
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the torus surface
ax.plot_surface(X, Y, Z, color='lightblue', alpha=0.5, edgecolor='gray')

# Plot Villarceau circles (combined rotor paths)
for offset in np.linspace(0, 2 * np.pi, 6, endpoint=False):
    theta = np.linspace(0, 2 * np.pi, 400)
    x = (R + r * np.cos(theta)) * np.cos(theta + offset)
    y = (R + r * np.cos(theta)) * np.sin(theta + offset)
    z = r * np.sin(theta)
    ax.plot(x, y, z, 'r', linewidth=2)

ax.set_title('Conformal Villarceau Rotor: Contact Fiber Mapping')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# Table of coordinate correspondences
correspondence = {
    "Contact Base-Base (2 turns)": "Torus parallels (phi motion)",
    "Contact Base-Fiber (2 turns)": "Meridians / Villarceau circle directions",
    "Contact Fiber-Fiber (2 turns)": "Fiber motion (psi motion)",
    "Reeb Flow (fiber)": "Isoclinic rotor (conformal time circles)",
    "Total 6 windings": "Composite conformal rotor phase"
}

for key, value in correspondence.items():
    print(f"{key} <=> {value}")
