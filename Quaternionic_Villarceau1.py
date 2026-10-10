import numpy as np
import matplotlib.pyplot as plt

# Reeb flow (right multiplication)
t = np.linspace(0, 4*np.pi, 200)
q0 = np.array([np.sqrt(0.6), np.sqrt(0.4), 0, 0])  # Avoids pole and highlights spiral
reeb = np.array([
    q0[0]*np.cos(t) - q0[1]*np.sin(t),
    q0[0]*np.sin(t) + q0[1]*np.cos(t),
    np.zeros_like(t),
    np.zeros_like(t)
])

# Villarceau spiral on Clifford torus
gamma = np.array([
    np.cos(t)/np.sqrt(2),
    np.sin(t)/np.sqrt(2),
    -np.sin(t)/np.sqrt(2),
    np.cos(t)/np.sqrt(2)
])

# Safe stereographic projection
def stereographic(q):
    denom = 1 - q[0]
    denom[abs(denom) < 1e-12] = 1e-12  # Pole avoidance
    return q[1:]/denom

# Project and plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(*stereographic(reeb), 'r-', lw=2, label='Reeb flow')
ax.plot(*stereographic(gamma), 'b-', lw=1.5, label='Villarceau spiral')
ax.legend(); plt.show()