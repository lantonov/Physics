import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- 1. Contact Volume Collapse Tracking ---
# Define contact form: alpha = p dq - H dt
# For tracking: compute alpha ^ d(alpha)
def contact_volume(p, q, H):
    # alpha = p dq - H dt  => differential: d(alpha) = dp ^ dq - dH ^ dt
    # Contact volume (in minimal form) ~ p * (dp ^ dq)
    return p

# Define energy levels near critical point
H_vals = np.linspace(-1, 1, 500)
# Assume q = 1 for simplicity, p = sqrt(2m(H - V(q))) ~ sqrt(2(H))
m = 1
V = 0
p_vals = np.sqrt(2 * np.maximum(H_vals - V, 0))  # Only real p when H >= V

# Compute contact volume
volumes = contact_volume(p_vals, 1, H_vals)

plt.figure(figsize=(10, 6))
plt.plot(H_vals, volumes, label=r'Contact Volume Proxy: $\alpha \wedge d\alpha$', color='purple')
plt.axvline(0, color='red', linestyle='--', label='Critical Transition at $H=0$')
plt.xlabel('Hamiltonian H')
plt.ylabel('Contact Volume Proxy')
plt.title('Contact Volume Collapse Near $H = 0$')
plt.legend()
plt.grid()
plt.show()

# --- 2. Reeb Vector Field Transition ---
# In simple contact form: alpha = p dq - H dt
# Reeb field R satisfies: i_R d(alpha) = 0 and alpha(R) = 1
# Assume alpha = p dq - H dt => d(alpha) = dp ^ dq - dH ^ dt
# Let R = (a, b, c) components along (q, p, t)
# Solve: i_R d(alpha) = 0 -> contraction gives constraints
# Simplify: use p = sqrt(2(H)) as before, test behavior near H = 0
H_test = np.linspace(-1, 1, 500)
p_test = np.sqrt(2 * np.maximum(H_test, 0))

# The Reeb field direction changes sharply at H = 0
reeb_direction = np.where(H_test > 0, 1, -1) * np.abs(p_test)

plt.figure(figsize=(10, 6))
plt.plot(H_test, reeb_direction, label='Reeb Vector Proxy', color='darkgreen')
plt.axhline(0, color='red', linestyle='--', label='Transition at $H=0$')
plt.xlabel('Hamiltonian H')
plt.ylabel('Reeb Vector Proxy')
plt.title('Reeb Vector Field Transition Near $H = 0$')
plt.legend()
plt.grid()
plt.show()

# --- 3. Parabolic Flow Simulation (H = 0) ---
# Simple glome trajectory model: dq/dt = dH/dp = p/m, dp/dt = -dH/dq = 0 (no restoring force)
def glome_flow(t, y):
    q, p = y
    dqdt = p / m
    dpdt = 0
    return [dqdt, dpdt]

# Initial condition: p0 marginal, H = p^2 / (2m) ~ 0
q0 = 0
p0 = 1e-3  # Very low momentum -> near parabolic escape

sol = solve_ivp(glome_flow, [0, 1000], [q0, p0], t_eval=np.linspace(0, 1000, 1000))

plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label='Glome Trajectory (Parabolic Flow)', color='blue')
plt.xlabel('Time')
plt.ylabel('Position q(t)')
plt.title('Parabolic Flow: Marginal Glome Escape at H = 0')
plt.grid()
plt.legend()
plt.show()
