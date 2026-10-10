from vpython import *
import numpy as np

# === PARAMETERS ===
I1, I2, I3 = 1, 2, 3  # Principal moments of inertia
w = np.array([1.5, 2.0, 0.5])  # Initial angular velocity

# === SCENE SETUP ===
scene = canvas(title="Poinsot Rolling with Inertia Ellipsoid")
scene.width = 800
scene.height = 600
scene.center = vector(0, 0, 0)
scene.background = color.white

# === FUNCTION: ELLIPSOID MESH ===
def build_ellipsoid_mesh():
    a, b, c = 1/I1, 1/I2, 1/I3
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    x = a * np.cos(u) * np.sin(v)
    y = b * np.sin(u) * np.sin(v)
    z = c * np.cos(v)

    mesh_objs = []
    for i in range(u.shape[0]-1):
        for j in range(u.shape[1]-1):
            p1 = vector(x[i, j], y[i, j], z[i, j])
            p2 = vector(x[i+1, j], y[i+1, j], z[i+1, j])
            p3 = vector(x[i+1, j+1], y[i+1, j+1], z[i+1, j+1])
            p4 = vector(x[i, j+1], y[i, j+1], z[i, j+1])
            mesh_objs.append(quad(
                v0=vertex(pos=p1, color=color.red, opacity=0.3),
                v1=vertex(pos=p2, color=color.red, opacity=0.3),
                v2=vertex(pos=p3, color=color.red, opacity=0.3),
                v3=vertex(pos=p4, color=color.red, opacity=0.3)
            ))
    return compound(mesh_objs)

# === FUNCTION: UPDATE ORIENTATION ===
def update_ellipsoid_orientation(mesh, R):
    x_axis = vector(R[0,0], R[1,0], R[2,0])
    y_axis = vector(R[0,1], R[1,1], R[2,1])
    mesh.axis = x_axis
    mesh.up = y_axis

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
inertia_mesh = build_ellipsoid_mesh()
inertia_mesh.pos = vector(0, 0, 0)

# === TIME LOOP ===
dt = 0.01
t = 0

while True:
    rate(100)

    # Rigid body dynamics (Euler equations, simplified)
    dw = np.array([
        ((I2 - I3) * w[1] * w[2]) / I1,
        ((I3 - I1) * w[2] * w[0]) / I2,
        ((I1 - I2) * w[0] * w[1]) / I3
    ])
    w += dw * dt

    # Update rotation matrix
    R = integrate_rotation(R, w, dt)

    # Re-orthonormalize (simple Gram-Schmidt)
    u, v_ = R[:,0], R[:,1]
    u = u / np.linalg.norm(u)
    v_ = v_ - np.dot(v_, u) * u
    v_ = v_ / np.linalg.norm(v_)
    w_ = np.cross(u, v_)
    R = np.column_stack([u, v_, w_])

    update_ellipsoid_orientation(inertia_mesh, R)
    t += dt
