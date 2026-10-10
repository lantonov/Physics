import matplotlib.pyplot as plt
import numpy as np

# Define parameters
R = 20  # Radius of the large circle
r = 1  # Radius of the small circle
d = r  # Distance from the center of the small circle

# Define the epicycloid parametric equations
t = np.linspace(0, 2*np.pi, 1000)
x = (R + r) * np.cos(t) - d * np.cos((R + r) * t / r)
y = (R + r) * np.sin(t) - d * np.sin((R + r) * t / r)

# Define the large circle parametric equations
x_circle = R * np.cos(t)
y_circle = R * np.sin(t)

# Plot the epicycloid
plt.figure(figsize=(6,6))
plt.plot(x, y, label='r circle')
plt.plot(x_circle, y_circle, label='t circle', linestyle='--')

plt.gca().set_aspect('equal', adjustable='box')
plt.legend()

# Remove the frame and axes
plt.axis('off')

plt.show()
