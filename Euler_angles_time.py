import numpy as np
from vpython import *
import math

# === SETUP ===
scene.background = color.white
scene.title = "Poinsot Rolling with Euler Angles and Interactive Controls"
scene.width = 1000
scene.height = 700

# === INERTIA TENSOR ===
I1, I2, I3 = 1.0, 2.0, 3.0
I = np.diag([I1, I2, I3])

# === INITIAL STATE ===
omega = np.array([1.0, 0.2, 1.0])
dt = 0.01
t = 0
paused = False

# === GUI CONTROLS ===
def toggle_pause(evt):
    global paused
    paused = not paused

def reset_sim(evt):
    global omega, t, R
    omega[:] = [1.0, 0.2, 1.0]
    t = 0
    R = np.eye(3)
    tip.clear_trail()
    inertia_mesh.pos = vector(0, 0, 0)

scene.append_to_caption("\n")
button(text="⏯ Pause/Play", bind=toggle_pause)
scene.append_to_caption("    ")
button(text="🔄 Reset", bind=reset_sim)
scene.append_to_caption("\n\n")

# === VISUAL ELEMENTS ===
tip = sphere(radius=0.05, color=color.green, make_trail=True)
L_arrow = arrow(pos=vector(0, 0, 0), axis=vector(*omega * np.diag(I)), color=color.blue)

# === ELLIPSOID MESH ===
def build_ellipsoid_mesh():
    # Parameters
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

inertia_mesh = build_ellipsoid_mesh()

# === ROTATION MATRIX (BODY TO SPACE) ===
R = np.eye(3)  # Start aligned with inertial frame

# === EULER ANGLE COMPUTATION ===
def get_euler_angles(R):
    """Extract (phi, theta, psi) from rotation matrix using ZXZ convention"""
    theta = np.arccos(R[2, 2])
    if np.sin(theta) > 1e-6:
        phi = np.arctan2(R[0, 2], -R[1, 2])
        psi = np.arctan2(R[2, 0], R[2, 1])
    else:
        phi = 0
        psi = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees([phi, theta, psi])

# === RIGID BODY DYNAMICS ===
def d_omega(omega):
    return np.array([
        ((I2 - I3) * omega[1] * omega[2]) / I1,
        ((I3 - I1) * omega[2] * omega[0]) / I2,
        ((I1 - I2) * omega[0] * omega[1]) / I3
    ])

# === SIMULATION LOOP ===
while True:
    rate(100)

    if paused:
        continue

    # Update dynamics
    omega += d_omega(omega) * dt

    # Build skew-symmetric matrix for angular velocity
    omega_skew = np.array([
        [0, -omega[2], omega[1]],
        [omega[2], 0, -omega[0]],
        [-omega[1], omega[0], 0]
    ])

    # Update rotation matrix using Rodrigues' rule (approx)
    R += omega_skew @ R * dt

    # Keep R orthonormal (via QR re-orthogonalization)
    R, _ = np.linalg.qr(R)

    # Update Euler angles
    phi, theta, psi = get_euler_angles(R)

    # Display or log Euler angles (optional)
    scene.caption = f"t = {t:.2f} s\nφ = {phi:.2f}°, θ = {theta:.2f}°, ψ = {psi:.2f}°"

    # Polhode path (angular velocity tip on inertia ellipsoid)
    omega_point = omega / np.sqrt([I1, I2, I3])
    tip.pos = vector(*omega_point)

    # Update ellipsoid position so tip rides the angular momentum direction
    L_vec = omega * np.diag(I)
    L_arrow.axis = vector(*L_vec)
    inertia_mesh.pos = L_arrow.axis - tip.pos

    t += dt
