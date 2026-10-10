import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Aesthetics
plt.rcParams.update({
    "figure.figsize": (8, 7),
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
})

def light_cone(ax, rmax=2.2, color="#F39C12", alpha=0.15):
    r = np.linspace(0, rmax, 100)
    phi = np.linspace(0, 2*np.pi, 200)
    R, PHI = np.meshgrid(r, phi)
    X = R * np.cos(PHI)
    Y = R * np.sin(PHI)
    T = R
    ax.plot_surface(T, X, Y, color=color, alpha=alpha, linewidth=0, antialiased=True)
    ax.plot_surface(-T, X, Y, color=color, alpha=alpha, linewidth=0, antialiased=True)

def timelike_hyperboloid(ax, umax=1.6, nu=100, nv=160, color="#2E86C1", alpha=0.35):
    # Parametrization of t^2 - r^2 = 1:
    # r = sinh(u), t = ±cosh(u), 0<=u
    u = np.linspace(0.01, umax, nu)
    v = np.linspace(0, 2*np.pi, nv)
    U, V = np.meshgrid(u, v)
    r = np.sinh(U)
    t_pos = np.cosh(U)
    t_neg = -np.cosh(U)
    x = r * np.cos(V)
    y = r * np.sin(V)
    ax.plot_surface(t_pos, x, y, color=color, alpha=alpha, linewidth=0, antialiased=True)
    ax.plot_surface(t_neg, x, y, color=color, alpha=alpha, linewidth=0, antialiased=True)

def main():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Draw cone first (translucent), then hyperboloid sheets
    light_cone(ax, rmax=2.2, color="#F39C12", alpha=0.18)
    timelike_hyperboloid(ax, umax=1.7, color="#2874A6", alpha=0.42)

    # Axes labels and annotations
    ax.set_title("Timelike hyperboloid sheets inside the light cone")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_zlabel("y")

    # Annotations
    ax.text(1.35, 0.0, 0.0, "Timelike hyperboloid\n(two sheets)", color="#1B4F72")
    ax.text(0.15, 0.0, 1.7, "Light cone", color="#AF601A")

    # View and limits
    lim = 2.2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1.05, 1.0, 1.0))
    ax.view_init(elev=22, azim=32)

    # Subtle grid
    ax.grid(False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
