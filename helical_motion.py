import numpy as np
import matplotlib.pyplot as plt

# Extending the analogy to include elliptical and helical motion under rotational transformation.
# The elliptical motion is achieved by scaling the components of the position vector, and helical motion includes z-motion.

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# Parameters
omega = 1  # Angular velocity
t = np.linspace(0, 10 * np.pi, 1000)  # Time array

# Circular motion - Initial position
r0_x = 10  # x-component
r0_y = 0   # y-component
r0_z = 0   # z-component

# Elliptical motion - Scale x and y differently
a, b = 10, 5  # Semi-major and semi-minor axes

# Helical motion - Add z-dependence
z_amplitude = 0.5

# Rotational motion under analogy
def rotation_matrix(omega, t):
    """Returns a 2D rotation matrix for angular velocity omega and time t."""
    cos_term = np.cos(omega * t)
    sin_term = np.sin(omega * t)
    return np.array([[cos_term, -sin_term],
                     [sin_term, cos_term]])

# Generate motion
circular_motion = np.array([np.cos(omega * t), np.sin(omega * t), np.zeros_like(t)])
elliptical_motion = np.array([a * np.cos(omega * t), b * np.sin(omega * t), np.zeros_like(t)])
helical_motion = np.array([a * np.cos(omega * t), b * np.sin(omega * t), z_amplitude * t])

# Plotting the motion
fig = plt.figure(figsize=(12, 6))

# Circular Motion
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.plot(circular_motion[0], circular_motion[1], circular_motion[2], label='Circular Motion')
ax1.set_title("Circular Motion")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")
ax1.legend()

# Elliptical Motion
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.plot(elliptical_motion[0], elliptical_motion[1], elliptical_motion[2], label='Elliptical Motion', color='orange')
ax2.set_title("Elliptical Motion")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("z")
ax2.legend()

# Helical Motion
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.plot(helical_motion[0], helical_motion[1], helical_motion[2], label='Helical Motion', color='green')
ax3.set_title("Helical Motion")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_zlabel("z")
ax3.legend()

plt.tight_layout()
plt.show()
