import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ------------------------------
# Helper: aesthetic defaults
# ------------------------------
plt.rcParams.update({
    "figure.figsize": (14, 6),
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
})

# ------------------------------
# Left panel: Minkowski fibers (3D slice)
# Model surfaces in (t, x, y) with z=0 slice:
#   Timelike unit:  -t^2 + x^2 + y^2 = -1  -> t^2 - (x^2+y^2) = 1  (two-sheeted)
#   Spacelike unit: -t^2 + x^2 + y^2 = +1  -> (x^2+y^2) - t^2 = 1  (one-sheeted)
#   Null cone:      -t^2 + x^2 + y^2 = 0   -> t = ±sqrt(x^2+y^2)
# ------------------------------
def minkowski_surfaces(ax):
    # Timelike hyperboloid (two sheets): param via t = ±cosh(u), r = sinh(u)
    u = np.linspace(0.02, 1.6, 80)
    v = np.linspace(0, 2*np.pi, 120)
    U, V = np.meshgrid(u, v)
    r_t = np.sinh(U)
    t_pos = np.cosh(U)
    t_neg = -np.cosh(U)
    x_t = r_t * np.cos(V)
    y_t = r_t * np.sin(V)

    ax.plot_surface(t_pos, x_t, y_t, rstride=1, cstride=1, color="#5DADE2",
                    alpha=0.35, linewidth=0, antialiased=True)
    ax.plot_surface(t_neg, x_t, y_t, rstride=1, cstride=1, color="#5DADE2",
                    alpha=0.35, linewidth=0, antialiased=True)

    # Spacelike hyperboloid (one sheet): (x^2+y^2) - t^2 = 1 -> r^2 - t^2 = 1
    # Param via r = cosh(u), t = sinh(u)
    r_s = np.cosh(U)
    t_s = np.sinh(U)
    x_s = r_s * np.cos(V)
    y_s = r_s * np.sin(V)
    ax.plot_surface(t_s, x_s, y_s, rstride=1, cstride=1, color="#52BE80",
                    alpha=0.28, linewidth=0, antialiased=True)

    # Null cone: t = ±sqrt(x^2 + y^2)
    r = np.linspace(0, 2.0, 80)
    phi = np.linspace(0, 2*np.pi, 180)
    R, PHI = np.meshgrid(r, phi)
    Xc = R * np.cos(PHI)
    Yc = R * np.sin(PHI)
    Tc = R
    ax.plot_surface(Tc, Xc, Yc, color="#F5B041", alpha=0.15, linewidth=0)
    ax.plot_surface(-Tc, Xc, Yc, color="#F5B041", alpha=0.15, linewidth=0)

    # Guides and labels
    ax.set_title("Minkowski unit cotangent fibers (slice)")
    ax.set_xlabel("t (time-like)")
    ax.set_ylabel("x")
    ax.set_zlabel("y")
    ax.text(1.45, 0, 0, "Timelike\n(two-sheeted)", color="#2E86C1")
    ax.text(0.9, 1.7, 0, "Spacelike\n(one-sheeted)", color="#1E8449")
    ax.text(0.1, 0, 1.7, "Null cone", color="#AF601A")

    # Aspect and view
    ax.set_box_aspect((1.2, 1, 1))
    ax.view_init(elev=22, azim=30)
    lim = 2.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)


# ------------------------------
# Right panel: Hopf fibration via stereographic projection
# Represent S^3 ⊂ C^2 by (z1, z2) = (cos θ e^{iφ}, sin θ e^{iψ})
# Stereographic projection from (-1,0,0,0) onto R^3:
#   Given (x1,x2,x3,x4) ∈ S^3, map to R^3 by:
#   X = x1 / (1 + x4), Y = x2 / (1 + x4), Z = x3 / (1 + x4)
# A Hopf fiber is obtained by fixing a point on S^2 via the Hopf map
# and sweeping a circle in S^3; below we draw several linked fibers.
# ------------------------------
def hopf_fibers(ax, n_fibers=8, points_per_fiber=400):
    # Sample fibers by fixing (theta0, delta) pairs
    # Param on S^3: (x1,x2,x3,x4) from angles θ, φ, ψ
    # Fiber circles: fix θ and (φ-ψ) = const; parameterize by α along the fiber
    def stereographic(x1, x2, x3, x4):
        denom = 1.0 + x4
        # Avoid singularity at x4 = -1 by small epsilon
        denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
        return x1/denom, x2/denom, x3/denom

    colors = plt.cm.hsv(np.linspace(0, 1, n_fibers, endpoint=False))
    alphas = np.linspace(0, 2*np.pi, points_per_fiber)

    # Choose a set of base points on S^2 (via θ) and relative phases (delta)
    thetas = np.linspace(0.3, np.pi/2 - 0.3, n_fibers//2)
    deltas = np.linspace(0, np.pi, n_fibers - len(thetas))
    params = []
    for th in thetas:
        params.append(("theta", th))
    for de in deltas:
        params.append(("delta", de))

    for i in range(n_fibers):
        # Alternate between fixing θ and fixing δ to spread fibers
        mode, val = params[i]
        if mode == "theta":
            theta = val
            delta = 0.8
        else:
            theta = 0.9
            delta = val

        # Construct a fiber by sweeping α; set φ = α, ψ = φ - δ
        phi = alphas
        psi = phi - delta
        # S^3 coordinates
        x1 = np.cos(theta) * np.cos(phi)
        x2 = np.cos(theta) * np.sin(phi)
        x3 = np.sin(theta) * np.cos(psi)
        x4 = np.sin(theta) * np.sin(psi)

        X, Y, Z = stereographic(x1, x2, x3, x4)
        ax.plot3D(X, Y, Z, color=colors[i], linewidth=2.0, alpha=0.95)

    # Draw a faint torus guide to suggest fiber organization
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, 2*np.pi, 30)
    U, V = np.meshgrid(u, v)
    R, r = 1.2, 0.45
    Xt = (R + r*np.cos(V)) * np.cos(U)
    Yt = (R + r*np.cos(V)) * np.sin(U)
    Zt = r * np.sin(V)
    ax.plot_wireframe(Xt, Yt, Zt, rstride=3, cstride=3, color="#BFC9CA", alpha=0.25, linewidth=0.5)

    ax.set_title("Hopf fibration on S3 (stereographic projection)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.text(0, 0, 1.3, "Linked S¹ fibers", color="#616A6B")
    ax.set_box_aspect((1, 1, 0.9))
    ax.view_init(elev=24, azim=38)
    lim = 2.2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

# ------------------------------
# Compose figure
# ------------------------------
fig = plt.figure()
ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122, projection="3d")

minkowski_surfaces(ax1)
hopf_fibers(ax2, n_fibers=10, points_per_fiber=500)

# Global annotation
fig.suptitle("Minkowski causal fibers vs. Glome Hopf fibration", y=0.98)
plt.tight_layout()
plt.show()
