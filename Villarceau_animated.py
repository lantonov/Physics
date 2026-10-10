# Conformal Villarceau Rotor Simulation

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

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

# Create the figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the torus surface
ax.plot_surface(X, Y, Z, color='lightblue', alpha=0.3, edgecolor='gray')

# Prepare Villarceau circle line
villarceau_line, = ax.plot([], [], [], 'r', linewidth=2)

# Animation setup
def init():
    villarceau_line.set_data([], [])
    villarceau_line.set_3d_properties([])
    return villarceau_line,

# Animation function
def animate(frame):
    offset = 2 * np.pi * frame / 200  # Smooth evolution
    theta = np.linspace(0, 2 * np.pi, 400)
    x = (R + r * np.cos(theta)) * np.cos(theta + offset)
    y = (R + r * np.cos(theta)) * np.sin(theta + offset)
    z = r * np.sin(theta)
    villarceau_line.set_data(x, y)
    villarceau_line.set_3d_properties(z)
    return villarceau_line,

# Configure plot limits and labels
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_zlim(-2, 2)
ax.set_title('Animated Conformal Villarceau Rotor Evolution')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Run animation
ani = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=200, interval=50, blit=True)

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
