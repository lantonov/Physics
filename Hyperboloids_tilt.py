import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

plt.rcParams.update({"figure.figsize": (8, 7)})

def light_cone(ax, rmax=2.0, color="#F39C12", alpha=0.15):
    r = np.linspace(0, rmax, 80)
    phi = np.linspace(0, 2*np.pi, 160)
    R, PHI = np.meshgrid(r, phi)
    X = R * np.cos(PHI)
    Y = R * np.sin(PHI)
    T = R
    ax.plot_surface(T, X, Y, color=color, alpha=alpha, linewidth=0)
    ax.plot_surface(-T, X, Y, color=color, alpha=alpha, linewidth=0)

def circle_of_present(ax, t0=1.5, color="blue"):
    phi = np.linspace(0, 2*np.pi, 200)
    x = np.cos(phi) * t0
    y = np.sin(phi) * t0
    t = np.full_like(x, t0)
    ax.plot(t, x, y, color=color, linewidth=2, label="v=0 (circle)")

def ellipse_of_present(ax, t0=1.5, v=0.6, color="red"):
    # Tilted plane: t = t0 + v*x
    phi = np.linspace(0, 2*np.pi, 400)
    # Parametrize ellipse in x-y plane
    x = np.cos(phi) * t0
    y = np.sin(phi) * t0
    t = t0 + v * x
    ax.plot(t, x, y, color=color, linewidth=2, label="v>0 (ellipse)")

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Draw cone
light_cone(ax)

# Circle (rest frame)
circle_of_present(ax, t0=1.5, color="blue")

# Ellipse (boosted frame)
ellipse_of_present(ax, t0=1.5, v=0.6, color="red")

# Labels
ax.set_title("Hypercircle of the Present vs. Tilted Ellipse (Dual Perspective)")
ax.set_xlabel("t")
ax.set_ylabel("x")
ax.set_zlabel("y")
ax.legend()

ax.set_xlim(0, 2.0)
ax.set_ylim(-2.0, 2.0)
ax.set_zlim(-2.0, 2.0)
ax.view_init(elev=22, azim=32)
plt.tight_layout()
plt.show()
