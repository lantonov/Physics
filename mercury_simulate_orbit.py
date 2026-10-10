import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# Constants
a = 0.387  # AU, semi-major axis of Mercury
e = 0.206  # eccentricity
orbits = 10
delta_phi = 0.001  # precession rate (radians per orbit)

# Angular domain
phi = np.linspace(0, 2 * np.pi * orbits, 5000)
delta_phi_total = delta_phi * phi

# Relativistic precession formula
p = a * (1 - e**2)
r = p / (1 + e * np.cos(phi - delta_phi_total))

# Convert to Cartesian
x = r * np.cos(phi)
y = r * np.sin(phi)

# Detect perihelia
perihelion_indices = argrelextrema(r, np.less)[0]
perihelion_phis = phi[perihelion_indices]
delta_phis = np.diff(perihelion_phis)
precession_per_orbit_arcsec = np.mean(delta_phis - 2 * np.pi) * (3600 * 180 / np.pi)

# Plot
plt.figure(figsize=(8, 8))
plt.plot(x, y, label="Precessing Orbit")
plt.plot(0, 0, 'yo', label="Sun")
plt.plot(x[perihelion_indices], y[perihelion_indices], 'ro', label="Perihelia")
for i, idx in enumerate(perihelion_indices[1:], 1):
    plt.annotate(f"{i}", (x[idx], y[idx]), textcoords="offset points", xytext=(5,5), fontsize=8)
plt.title("Simulated Mercury Orbit with Precession")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.xlabel("x [AU]")
plt.ylabel("y [AU]")
plt.show()

print(f"Simulated precession per orbit: {precession_per_orbit_arcsec:.4f} arcseconds")
