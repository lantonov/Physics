import numpy as np
import matplotlib.pyplot as plt

# Parameters
omega = np.array([0, 0, 1])  # Angular velocity vector (z-axis rotation)
r0 = np.array([1, 0, 0])     # Initial position vector
t_values = np.linspace(0, 10, 100)  # Time values

# Compute r' using the rotational analogy formula
r_prime = np.zeros((len(t_values), 3))
for i, t in enumerate(t_values):
    cross_term = np.cross(omega, r0) * t  # omega x r * t
    r_prime[i] = r0 + cross_term          # r' = r + (omega x r) t

# Extract x and y components for plotting
x_values = r_prime[:, 0]
y_values = r_prime[:, 1]

# Plot the trajectory
plt.figure(figsize=(8, 8))
plt.plot(x_values, y_values, label=r'$\vec{r} + (\vec{\omega} \times \vec{r})t$', color='b')
plt.scatter(r0[0], r0[1], color='r', label='Initial position $\vec{r}$')  # Mark the initial position

plt.title("Rotational Analogy of Galilean Transformation")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.axis('equal')  # Equal scaling for x and y axes
plt.legend()
plt.show()
