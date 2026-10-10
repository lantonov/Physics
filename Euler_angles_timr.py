import numpy as np
from vpython import *

# Scene setup
scene.background = color.white
scene.width = 800
scene.height = 600
scene.center = vector(0, 0, 0)

# Inertia components (asymmetric)
I1, I2, I3 = 1.0, 2.0, 3.0

# Initial angular velocity
omega = np.array([1.5, 0.2, 1.0])  # [wx, wy, wz]

# Time step
dt = 0.01
t_max = 20

# Inertia tensor in principal axes
I = np.array([I1, I2, I3])

# Body frame angular momentum
L = omega * I

# Objects for visualization
ellipsoid = ellipsoid(pos=vector(0, 0, 0), length=2, height=1.5, width=1,
                      color=color.red, opacity=0.2, make_trail=False)

tip = sphere(radius=0.05, color=color.green, make_trail=True)
L_arrow = arrow(pos=vector(0, 0, 0), axis=vector(*L), color=color.blue)

def d_omega(omega):
    # Euler's equations for a rigid body in principal axes
    return np.array([
        ((I2 - I3) * omega[1] * omega[2]) / I1,
        ((I3 - I1) * omega[2] * omega[0]) / I2,
        ((I1 - I2) * omega[0] * omega[1]) / I3
    ])

def inertia_ellipsoid_point(omega):
    # Surface of ellipsoid representing inertia ellipsoid
    return np.array([
        omega[0] / I1,
        omega[1] / I2,
        omega[2] / I3
    ])

t = 0
while t < t_max:
    rate(100)

    # Update angular velocity using Euler's equations
    omega += d_omega(omega) * dt

    # Angular momentum is constant in inertial space
    L_body = omega * I
    L_arrow.axis = vector(*L_body)

    # Polhode point (tip of omega vector on ellipsoid surface)
    polhode_point = inertia_ellipsoid_point(omega)
    polhode_vec = vector(*polhode_point)

    # Move tip of omega (green dot)
    tip.pos = polhode_vec

    # Move ellipsoid to simulate rolling (translate to keep tip on L vector)
    ellipsoid.pos = L_arrow.axis - polhode_vec

    t += dt
