import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.674e-11  # gravitational constant
GM_earth = 3.986e14
GM_moon = 4.904e12

# Orbital parameters
AU = 1.496e11  # m
R_moon = 3.84e8  # m
omega_earth = 2 * np.pi / (365.25 * 24 * 3600)  # rad/s
omega_moon = 2 * np.pi / (27.3 * 24 * 3600)  # rad/s

# Simulation parameters
dt = 3600  # time step: 1 hour
steps = 24 * 100  # simulate 10 days

# Initialize arrays
pos = np.zeros((steps, 2))
pos_earth = np.zeros((steps, 2))
pos_moon = np.zeros((steps, 2))

# Initial positions
pos_earth[0] = np.array([AU, 0])
pos_moon[0] = pos_earth[0] + np.array([0, R_moon])
# S-IVB at Earth L2: slightly beyond Earth away from Sun
r_L2 = AU * (GM_earth / (3 * 1.327e20))**(1/3)
pos[0] = np.array([AU + r_L2, 0])
# Initial velocity: same as Earth's orbital around Sun
v = np.array([0, AU * omega_earth])

trajectory = np.zeros((steps, 2))

for i in range(1, steps):
    t = i * dt
    # Update Earth and Moon positions
    theta_e = omega_earth * t
    pos_earth[i] = np.array([AU * np.cos(theta_e), AU * np.sin(theta_e)])
    theta_m = omega_moon * t
    pos_moon[i] = pos_earth[i] + np.array([R_moon * np.cos(theta_m), R_moon * np.sin(theta_m)])
    
    # Compute gravitational acceleration on S-IVB by Earth and Moon only
    a_earth = -GM_earth * (pos[i-1] - pos_earth[i-1]) / np.linalg.norm(pos[i-1] - pos_earth[i-1])**3
    a_moon = -GM_moon * (pos[i-1] - pos_moon[i-1]) / np.linalg.norm(pos[i-1] - pos_moon[i-1])**3
    a_total = a_earth + a_moon
    
    # Update velocity and position
    v = v + a_total * dt
    pos[i] = pos[i-1] + v * dt
    
    trajectory[i] = pos[i]

# Plot trajectory excluding the Sun
plt.figure(figsize=(6, 6))
plt.plot(trajectory[:, 0], trajectory[:, 1])
plt.plot(pos_earth[:, 0], pos_earth[:, 1], 'o', markersize=3)  # Earth path
plt.plot(pos_moon[:, 0], pos_moon[:, 1], 'o', markersize=2)  # Moon path
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Simulated trajectory of J002E3 over 10 days (Earth & Moon only)')
plt.grid(True)
plt.show()
