import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def stable_stereographic_proj(q, epsilon=1e-8):
    denom = 1 - q[0]
    if np.abs(denom) < epsilon:
        denom = epsilon if denom >= 0 else -epsilon
    return q[1:] / denom

def gamma(s):
    # Option: Shift s slightly to avoid the north pole singularity
    # Or modify the curve so q[0] doesn't hit 1.
    return np.array([np.cos(s + 0.1), np.sin(s + 0.1)/np.sqrt(2), np.sin(s + 0.1)/np.sqrt(2), 0])

def numerical_derivative(f, s, h=1e-5):
    # You might experiment with adaptive h here.
    return (f(s+h) - f(s-h)) / (2*h)

def normalize(v, tol=1e-10):
    norm = np.linalg.norm(v)
    if norm < tol:
        return np.zeros_like(v)
    return v / norm

s_vals = np.linspace(0, 4*np.pi, 500)
curve_3d = np.array([stable_stereographic_proj(gamma(s)) for s in s_vals])

# Select a point for visualization, making sure it avoids the singular region
s0 = 2.0
p = stable_stereographic_proj(gamma(s0))

# Tangent vector using a more robust derivative estimate
t = normalize(numerical_derivative(lambda s: stable_stereographic_proj(gamma(s)), s0))

# Second derivative at s0 for the normal (using a higher order formula can be applied similarly)
t1_candidate = numerical_derivative(lambda s: normalize(numerical_derivative(lambda x: stable_stereographic_proj(gamma(x)), s)), s0)
t1 = normalize(t1_candidate)

# Binormal vector as cross product
b = np.cross(t, t1)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# Plot curve
ax.plot(curve_3d[:,0], curve_3d[:,1], curve_3d[:,2], 'b-', label='Projected path on $S^3$')

# Plot Frenet frame at p
scale = 0.5
ax.quiver(p[0], p[1], p[2], t[0], t[1], t[2], color='r', length=scale, normalize=True, label='Tangent')
ax.quiver(p[0], p[1], p[2], t1[0], t1[1], t1[2], color='g', length=scale, normalize=True, label='Normal')
ax.quiver(p[0], p[1], p[2], b[0], b[1], b[2], color='m', length=scale, normalize=True, label='Binormal')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Refined Stereographic Projection and Frenet Frame')
ax.legend()
plt.show()
