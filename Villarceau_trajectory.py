import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
c = 3.0
k = 2.0
theta0 = np.pi/4  # Initial latitude
phi0 = 0.0         # Initial longitude
tmax = 20.0        # Time range

# Define the ODE system
def hopf_odes(t, y):
    theta, theta_dot, phi = y
    dtheta_dt = theta_dot
    dtheta_ddot = -c * k * np.sin(theta)
    dphi_dt = k - c * np.cos(theta)
    return [dtheta_dt, dtheta_ddot, dphi_dt]

# Initial conditions: [theta(0), theta'(0), phi(0)]
y0 = [theta0, 0.0, phi0]

# Solve the ODE
sol = solve_ivp(hopf_odes, [0, tmax], y0, t_eval=np.linspace(0, tmax, 1000))

# Extract solutions
theta = sol.y[0]
phi = sol.y[2]

# Compute trajectory on S^2
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

# Plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, 'r-', lw=2)

# Draw S^2
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
sphere_x = np.cos(u)*np.sin(v)
sphere_y = np.sin(u)*np.sin(v)
sphere_z = np.cos(v)
ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color="k", alpha=0.1)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
plt.title('Quasiperiodic Motion on S²')
plt.show()