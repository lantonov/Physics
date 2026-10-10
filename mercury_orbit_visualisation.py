import numpy as np
import matplotlib.pyplot as plt

# Constants
num_orbits = 100
delta_phi_rad = 0.1031 / 3600 * np.pi / 180  # Convert arcseconds to radians

# Orbital parameters
a = 1  # Scaled semi-major axis
e = 0.2056  # Eccentricity of Mercury

# Function to generate an orbit with given phase shift
def generate_orbit(phase_shift, num_points=500):
    phi = np.linspace(0, 2 * np.pi, num_points)
    r = a * (1 - e**2) / (1 + e * np.cos(phi - phase_shift))
    return r, phi

# Generate orbits and accumulate perihelion shifts
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
colors = plt.cm.viridis(np.linspace(0, 1, num_orbits))

for i in range(num_orbits):
    shift = i * delta_phi_rad
    r, phi = generate_orbit(shift)
    ax.plot(phi, r, color=colors[i], alpha=0.6)

# Aesthetic settings
ax.set_title("Mercury's Orbital Precession (Glome Model)", fontsize=14)
ax.set_rticks([])
ax.grid(True)

plt.show()
