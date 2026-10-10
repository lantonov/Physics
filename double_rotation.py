import numpy as np
import matplotlib.pyplot as plt

# Parameters for extended rotational analogy
r0 = np.array([10, 0, 0])  # Initial position vector
omega_vec = np.array([0, 0, 1])  # Angular velocity vector (z-axis rotation)
t_values = np.linspace(0, 10 * np.pi, 500)  # Time array

# Compute r' using the extended rotational analogy formula
def double_cross_product(omega, r, t):
    """Compute r' = r + (omega x r) x t."""
    omega_cross_r = np.cross(omega, r)  # First cross product: omega x r
    result = r + np.cross(omega_cross_r, t)  # Second cross product: (omega x r) x t
    return result

# Generate trajectory
r_prime_extended = np.zeros((len(t_values), 3))
for i, t in enumerate(t_values):
    t_vec = np.array([t, t, t])  # Treat time as a vector [t, t, t] for computation
    r_prime_extended[i] = double_cross_product(omega_vec, r0, t_vec)

# Extract x and y components for plotting
x_extended = r_prime_extended[:, 0]
y_extended = r_prime_extended[:, 1]
z_extended = r_prime_extended[:, 2]

# Plot the trajectory
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_extended, y_extended, z_extended, label=r'\vec{r} + (\vec{\omega} \times \vec{r}) \times \vec{t}', color='purple')
ax.scatter(r0[0], r0[1], r0[2], color='r', label='Initial position \vec{r}')  # Mark the initial position

# Labeling the axes
ax.set_title("Rotational Analogy with Double Cross Product")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.legend()
plt.show()
