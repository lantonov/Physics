import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define quaternionic left multiplications
def F_i(v):
    """Left multiplication by i on a quaternion v = (x0, x1, x2, x3)"""
    x0, x1, x2, x3 = v
    return np.array([-x1, x0, -x3, x2])

def F_j(v):
    x0, x1, x2, x3 = v
    return np.array([-x2, x3, x0, -x1])

def F_k(v):
    x0, x1, x2, x3 = v
    return np.array([-x3, -x2, x1, x0])

# Define a sample point on S^3 and compute contact plane
q = np.array([1, 0, 0, 0])  # north pole on S^3
tangent_basis = np.array([
    [0, 1, 0, 0],  # x1 direction
    [0, 0, 1, 0],  # x2 direction
    [0, 0, 0, 1]   # x3 direction
])

# Generate Villarceau-style flows in 3D (projecting onto 3-space)
t_vals = np.linspace(0, 8 * np.pi, 1000)
radius = 1.0

# Define a Villarceau-style rotor in the contact plane
def generate_villarceau_path(F, base_vec, steps=1000):
    traj = []
    v = base_vec.copy()
    for _ in range(steps):
        v = F(v)
        v = v / np.linalg.norm(v)  # reproject onto unit sphere
        traj.append(v[1:])  # drop x0, use 3D projection
    return np.array(traj)

# Generate three flows in the different quaternionic contact planes
traj_i = generate_villarceau_path(F_i, np.array([0, 0, 1, 0]))
traj_j = generate_villarceau_path(F_j, np.array([0, 1, 0, 0]))
traj_k = generate_villarceau_path(F_k, np.array([0, 0, 0, 1]))

# Plot the 3 projected Villarceau trajectories
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(*traj_i.T, label="F_i rotor", color='r')
ax.plot(*traj_j.T, label="F_j rotor", color='g')
ax.plot(*traj_k.T, label="F_k rotor", color='b')

ax.set_title("Quaternionic Villarceau Rotors on Contact S³ (Projected to ℝ³)")
ax.set_xlabel("x₁")
ax.set_ylabel("x₂")
ax.set_zlabel("x₃")
ax.legend()
plt.tight_layout()
plt.show()
