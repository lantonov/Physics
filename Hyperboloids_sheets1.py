import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

plt.rcParams.update({
    "figure.figsize": (8, 7),
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
})

def light_cone(ax, rmax=2.5, color="#F39C12", alpha=0.15):
    r = np.linspace(0, rmax, 100)
    phi = np.linspace(0, 2*np.pi, 200)
    R, PHI = np.meshgrid(r, phi)
    X = R * np.cos(PHI)
    Y = R * np.sin(PHI)
    T = R
    ax.plot_surface(T, X, Y, color=color, alpha=alpha, linewidth=0)
    ax.plot_surface(-T, X, Y, color=color, alpha=alpha, linewidth=0)

def timelike_hyperboloid(ax, a=1.0, umax=1.6, nu=80, nv=120, color="#2E86C1", alpha=0.35):
    # Parametrization: t^2 - r^2 = a^2
    u = np.linspace(0.01, umax, nu)
    v = np.linspace(0, 2*np.pi, nv)
    U, V = np.meshgrid(u, v)
    r = np.sinh(U) * a
    t_pos = np.cosh(U) * a
    t_neg = -np.cosh(U) * a
    x = r * np.cos(V)
    y = r * np.sin(V)
    ax.plot_surface(t_pos, x, y, color=color, alpha=alpha, linewidth=0)
    ax.plot_surface(t_neg, x, y, color=color, alpha=alpha, linewidth=0)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Draw cone
light_cone(ax, rmax=2.5)

# Draw several nested hyperboloid sheets with different constants a
for a, col in zip([0.5, 1.0, 1.5, 2.0], ["#1F618D", "#2874A6", "#5499C7", "#85C1E9"]):
    timelike_hyperboloid(ax, a=a, color=col, alpha=0.45)

ax.set_title("Multiple timelike hyperboloid sheets inside the light cone")
ax.set_xlabel("t")
ax.set_ylabel("x")
ax.set_zlabel("y")

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_zlim(-2.5, 2.5)
ax.set_box_aspect((1,1,1))
ax.view_init(elev=22, azim=32)

plt.tight_layout()
plt.show()
