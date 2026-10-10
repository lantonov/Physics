import matplotlib.pyplot as plt
import numpy as np

# Create a grid of (x, y) values
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
x, y = np.meshgrid(x, y)

# Fixed values for s and z
s = 1  # Example radius in the 4D hypersphere
z = 0  # Fixing z to 0 for projection

# Compute t from the equation s^2 = (1/2) * (t^2 + x^2 + y^2 + z^2)
# Solve for t^2
t_squared = 2 * s**2 - x**2 - y**2 - z**2

# Mask invalid values where t^2 < 0 (outside the hypersphere)
valid = t_squared >= 0
t = np.sqrt(t_squared, where=valid, out=np.full_like(t_squared, np.nan))

# Plot the surface
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the positive branch of t
ax.plot_surface(x, y, t, rstride=5, cstride=5, alpha=0.6, cmap="viridis")

# Plot the negative branch of t
ax.plot_surface(x, y, -t, rstride=5, cstride=5, alpha=0.6, cmap="viridis")

# Add labels and formatting
ax.set_title("Projection of $s^2 = \\frac{1}{2}(t^2 + x^2 + y^2 + z^2)$ (z=0)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("t")
plt.show()
