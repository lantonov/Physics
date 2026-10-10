import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Setup figure ---
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection="3d")

# Light cone
r = np.linspace(0, 2.0, 60)
phi = np.linspace(0, 2*np.pi, 120)
R, PHI = np.meshgrid(r, phi)
X = R * np.cos(PHI)
Y = R * np.sin(PHI)
T = R

ax.plot_surface(T, X, Y, color="#F39C12", alpha=0.15, linewidth=0)
ax.plot_surface(-T, X, Y, color="#F39C12", alpha=0.15, linewidth=0)

# Parameters for circle/ellipse
phi_line = np.linspace(0, 2*np.pi, 400)
t0 = 1.5
line, = ax.plot([], [], [], color="red", linewidth=2)

def init():
    line.set_data([], [])
    line.set_3d_properties([])
    return line,

def update(frame):
    v = frame/50.0  # velocity parameter from 0 to ~1
    x = np.cos(phi_line) * t0
    y = np.sin(phi_line) * t0
    t = t0 + v * x
    line.set_data(x, y)
    line.set_3d_properties(t)
    ax.set_title(f"Hypercircle → Ellipse (v={v:.2f})")
    return line,

ani = FuncAnimation(fig, update, frames=50, init_func=init,
                    blit=True, interval=200, repeat=True)

# Axes labels and limits
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("t")
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
ax.set_zlim(0,2.5)
ax.view_init(elev=22, azim=32)

plt.show()
