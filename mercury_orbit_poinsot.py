import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
G = 6.67430e-11  # gravitational constant (m^3 kg^-1 s^-2)
M = 1.989e30  # mass of the Sun (kg)
c = 299792458  # speed of light (m/s)

# Orbital parameters (semi-major axis, eccentricity)
a = 5.790e10  # semi-major axis (m)
e = 0.205  # eccentricity

# Equations of motion
def equations_of_motion(t, y):
    r, theta, v_r, v_theta = y
    # Radial equation
    drdt = v_r
    # Angular equation
    dthetadt = v_theta / r**2
    # Radial velocity equation (with relativistic correction for perihelion precession)
    dv_rdt = r * v_theta**2 / (r**2) - G * M / r**2 + (6 * np.pi * G * M / (c**2 * a * (1 - e**2)))
    # Angular velocity equation (no relativistic correction needed here)
    dv_thetadt = 0
    return [drdt, dthetadt, dv_rdt, dv_thetadt]

# Initial conditions (r, theta, v_r, v_theta)
r0 = a * (1 - e)
theta0 = 0  # initial angle
v_r0 = 0  # initial radial velocity
v_theta0 = np.sqrt(G * M / r0)  # initial tangential velocity

# Initial state vector
y0 = [r0, theta0, v_r0, v_theta0]

# Time span for the simulation (in seconds)
t_span = (0, 5e7)  # simulate for 50 years
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Solve the ODE
sol = solve_ivp(equations_of_motion, t_span, y0, t_eval=t_eval)

# Plot the result
r_values = sol.y[0]
theta_values = sol.y[1]

plt.figure(figsize=(6, 6))
plt.plot(r_values * np.cos(theta_values), r_values * np.sin(theta_values), label="Orbit with Precession")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Mercury's Orbit with Relativistic Precession")
plt.grid(True)
plt.legend()
plt.show()

# Find perihelion positions (minima in r)
perihelion_indices = np.argwhere(np.diff(np.sign(np.diff(r_values))) > 0).flatten() + 1
perihelion_angles = theta_values[perihelion_indices]

# Plot perihelion positions and angular advance
plt.plot(perihelion_angles, np.zeros_like(perihelion_angles), 'ro', label="Perihelion")
plt.xlabel("True Anomaly (radians)")
plt.ylabel("Precession (radians)")
plt.title("Perihelion Precession Over Time")
plt.grid(True)
plt.legend()
plt.show()
