import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# === PARAMETERS ===
I1, I2, I3 = 1, 2, 3  # Principal moments of inertia
w = np.array([1.5, 2.0, 0.5])  # Initial angular velocity

# === FUNCTION: ELLIPSOID MESH ===
def build_ellipsoid_mesh():
    a, b, c = 1/I1, 1/I2, 1/I3
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    x = a * np.cos(u) * np.sin(v)
    y = b * np.sin(u) * np.sin(v)
    z = c * np.cos(v)
    return x, y, z

# === FUNCTION: UPDATE ORIENTATION ===
def update_ellipsoid_orientation(ax, x, y, z, R):
    x_rot = R[0, 0]*x + R[0, 1]*y + R[0, 2]*z
    y_rot = R[1, 0]*x + R[1, 1]*y + R[1, 2]*z
    z_rot = R[2, 0]*x + R[2, 1]*y + R[2, 2]*z
    ax.clear()  # Clear the previous plot
    ax.plot_surface(x_rot, y_rot, z_rot, color='r', alpha=0.3)

# === ROTATION MATRIX FROM ANGULAR VELOCITY ===
def skew_symmetric(w):
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])

def integrate_rotation(R, w, dt):
    W = skew_symmetric(w)
    return R + dt * np.dot(W, R)

# === INITIAL SETUP ===
R = np.eye(3)  # Initial orientation matrix
x, y, z = build_ellipsoid_mesh()

# === ANIMATION SETUP ===
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# === TIME LOOP ===
dt = 0.01
t = 0

def update_plot(frame):
    global w, R, t
    dw = np.array([
        ((I2 - I3) * w[1] * w[2]) / I1,
        ((I3 - I1) * w[2] * w[0]) / I2,
        ((I1 - I2) * w[0] * w[1]) / I3
    ])
    w += dw * dt
    R = integrate_rotation(R, w, dt)

    # Re-orthonormalize (simple Gram-Schmidt)
    u, v_ = R[:, 0], R[:, 1]
    u = u / np.linalg.norm(u)
    v_ = v_ - np.dot(v_, u) * u
    v_ = v_ / np.linalg.norm(v_)
    w_ = np.cross(u, v_)
    R = np.column_stack([u, v_, w_])

    update_ellipsoid_orientation(ax, x, y, z, R)
    return ax,

ani = FuncAnimation(fig, update_plot, frames=100, interval=50)

plt.show()
