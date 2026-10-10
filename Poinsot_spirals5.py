import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
theta = np.linspace(-15, 15, 2000)
k = 1/3
r = 1/np.cosh(k*theta)  # bounded Poinsot spiral

# Spiral in 3D (projected in xy-plane, lifted along z-axis for cone analogy)
x = r * np.cos(theta)
y = r * np.sin(theta)
z = np.zeros_like(theta)

# Define cones
phi = np.linspace(0, 2*np.pi, 60)
h = np.linspace(0, 2, 60)
Phi, H = np.meshgrid(phi, h)
R_body = 0.5*H
X_body = R_body*np.cos(Phi)
Y_body = R_body*np.sin(Phi)
Z_body = H

R_space = 0.5*(2-H)
X_space = R_space*np.cos(Phi)
Y_space = R_space*np.sin(Phi)
Z_space = H

# Plot
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

# Body cone
ax.plot_surface(X_body, Y_body, Z_body, alpha=0.3, color='blue')
# Space cone
ax.plot_surface(X_space, Y_space, Z_space, alpha=0.3, color='red')
# Spiral projection in xy-plane
ax.plot(x, y, z, color='darkorange', linewidth=2)

ax.set_title("Poinsot cones with spiral projection")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.view_init(elev=25, azim=45)
plt.show()
