import numpy as np
import matplotlib.pyplot as plt

# Parameters for a spiral with precession (Poinsot-like)
# These values are illustrative and can be adjusted to match Mercury's real values
r0 = 0.387  # AU, average distance of Mercury to the Sun
epsilon = 0.01  # small precession factor per orbit

# Number of orbits and angular range
orbits = 10
phi = np.linspace(0, 2 * np.pi * orbits, 5000)

# Define a Poinsot-like spiral to simulate precession
r = r0 / (1 + epsilon * phi)

# Convert to Cartesian coordinates for plotting
x = r * np.cos(phi)
y = r * np.sin(phi)

# Plotting
plt.figure(figsize=(8, 8))
plt.plot(x, y, label="Poinsot Spiral (Precession)")
plt.plot(0, 0, 'yo', label="Sun (focus)")
plt.title("Simulated Precession of Mercury's Orbit\nvia Poinsot Spiral in Clifford Framework")
plt.xlabel("x [AU]")
plt.ylabel("y [AU]")
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()
