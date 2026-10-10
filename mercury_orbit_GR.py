import numpy as np
import matplotlib.pyplot as plt

# Constants
GM = 1.32712440018e20  # Gravitational parameter for the Sun in m^3/s^2
c = 3e8  # Speed of light in m/s
a = 5.790e10  # Semi-major axis of Mercury's orbit in meters
e = 0.2056  # Orbital eccentricity of Mercury
M = 1.989e30  # Mass of the Sun in kg

# Relativistic correction factor
def relativistic_correction(a, e):
    return (6 * np.pi * GM) / (c**2 * a * (1 - e**2))

# Initial conditions
r_initial = a * (1 - e)  # Perihelion (closest point)
phi_initial = 0  # Initial angle at perihelion

# Number of orbits to simulate
num_orbits = 50

# Time step for the calculation
dt = 365.25 * 24 * 3600  # Time step in seconds (one year)

# Precession calculation
precession_per_orbit = relativistic_correction(a, e)

# List to store precession angles
precession_angles = [phi_initial]

# Simulate the precession over multiple orbits
for i in range(num_orbits):
    phi_initial += precession_per_orbit  # Add the precession to the angle
    precession_angles.append(phi_initial)

# Plot the precession over time
plt.figure(figsize=(8, 6))
plt.plot(np.arange(num_orbits + 1), np.array(precession_angles) * (180/np.pi), label='Precession of Perihelion')
plt.xlabel('Orbit Number')
plt.ylabel('Perihelion Precession (degrees)')
plt.title('Mercury Orbital Precession (Relativistic Correction)')
plt.grid(True)
plt.legend()
plt.show()
