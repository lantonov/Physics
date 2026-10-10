# Re-import necessary libraries after reset
import numpy as np
import matplotlib.pyplot as plt

# Parameters for the rotational analogy
r0 = np.array([10, 0, 0])  # Initial position vector
omega_vec = np.array([0, 0, 1])  # Angular velocity vector along the z-axis
t_values = np.linspace(0, 10, 500)  # Time values

# Compute r' using the rotational analogy formula: r' = r + (omega x r) * t
def rotational_translation(omega, r, t):
    """Compute r' = r + (omega x r) * t."""
    omega_cross_r = np.cross(omega, r)  # omega x r is constant
    return r + omega_cross_r * t  # Add displacement over time

# Generate trajectory
r_prime_linear = np.array([rotational_translation(omega_vec, r0, t) for t in t_values])

# Extract x, y, and z components
x_linear = r_prime_linear[:, 0]
y_linear = r_prime_linear[:, 1]
z_linear = r_prime_linear[:, 2]

# Plot the trajectory
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_linear, y_linear, z_linear, label=r'$\vec{r} + (\vec{\omega} \times \vec{r}) t$', color='blue')
ax.scatter(r0[0], r0[1], r0[2], color='r', label='Initial position $\vec{r}$')  # Mark the initial position

# Labeling the axes
ax.set_title("Rotational Analogy: Linear Trajectory")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.legend()

plt.show()

