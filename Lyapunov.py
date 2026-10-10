#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from math import log

# =============================================================================
# Modified Lorenz System with Extra Forcing Terms
# =============================================================================
def lorenz_modified(t, state, sigma, rho, beta, epsilon):
    x, y, z = state
    # Base Lorenz equations:
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    # Additional forcing (from metric perturbations, for example)
    Fx = np.sin(x + y + z)
    Fy = np.cos(x - z)
    Fz = np.sin(z)
    dxdt += epsilon * Fx
    dydt += epsilon * Fy
    dzdt += epsilon * Fz
    return np.array([dxdt, dydt, dzdt])

# =============================================================================
# Jacobian of the Modified Lorenz System
# =============================================================================
def jacobian_lorenz(state, sigma, rho, beta, epsilon):
    x, y, z = state
    cos_xyz = np.cos(x + y + z)
    sin_xz  = np.sin(x - z)
    J = np.array([
        [-sigma + epsilon * cos_xyz, sigma + epsilon * cos_xyz, epsilon * cos_xyz],
        [(rho - z) - epsilon * sin_xz, -1, -x + epsilon * sin_xz],
        [y, x, -beta + epsilon * np.cos(z)]
    ])
    return J

# =============================================================================
# One Runge-Kutta 4 Step
# =============================================================================
def rk4_step(f, state, t, dt, *args):
    k1 = f(t, state, *args)
    k2 = f(t + dt/2.0, state + dt * k1/2.0, *args)
    k3 = f(t + dt/2.0, state + dt * k2/2.0, *args)
    k4 = f(t + dt, state + dt * k3, *args)
    return state + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

# =============================================================================
# Function to Compute the Maximum Lyapunov Exponent
# =============================================================================
def compute_lyapunov(epsilon, sigma, rho, beta, t_max=100.0, dt=0.01):
    state = np.array([1.0, 1.0, 20.0])
    delta0 = 1e-7
    deviation = delta0 * np.array([1.0, 0.0, 0.0])
    
    total_time = 0.0
    sum_ln = 0.0
    t = 0.0
    n_steps = int(t_max / dt)
    
    times = []
    le_values = []
    
    for i in range(n_steps):
        # Advance state by one RK4 step:
        state = rk4_step(lorenz_modified, state, t, dt, sigma, rho, beta, epsilon)
        # Linearized evolution:
        J = jacobian_lorenz(state, sigma, rho, beta, epsilon)
        k1 = J.dot(deviation)
        k2 = J.dot(deviation + dt*k1/2.0)
        k3 = J.dot(deviation + dt*k2/2.0)
        k4 = J.dot(deviation + dt*k3)
        deviation_next = deviation + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        norm = np.linalg.norm(deviation_next)
        sum_ln += np.log(norm / delta0)
        deviation = (delta0 / norm) * deviation_next
        
        total_time += dt
        t += dt
        
        if i % 1000 == 0:
            current_LE = sum_ln / total_time
            times.append(t)
            le_values.append(current_LE)
    
    max_LE = sum_ln / total_time
    return max_LE, times, le_values

# =============================================================================
# Case A: Nearly Integrable (Solar System Regime)
# Use parameters that yield a non-chaotic (or weakly perturbed) system.
# We choose: sigma = 10, rho = 15, beta = 8/3, epsilon = 1e-7.
# For these values, the two nonzero fixed points of the Lorenz system are stable.
# Hence the maximum LE should be near zero (or negative).
# =============================================================================
sigma_A = 1.32e-14
rho_A = 2.29e-4    # Reduced ? to yield a stable fixed point (non-chaotic)
beta_A = 2.28e-25
epsilon_A = 1e-7

LE_A, times_A, le_vals_A = compute_lyapunov(epsilon_A, sigma_A, rho_A, beta_A, t_max=100.0, dt=0.01)
print("Nearly Integrable Regime:")
print("Parameters: sigma=10, rho=15, beta=8/3, epsilon=1e-7")
print("Maximum Lyapunov Exponent (LE): {:.3e}".format(LE_A))

# =============================================================================
# Case B: Chaotic (Black Hole Regime)
# Use standard Lorenz parameters: sigma = 10, rho = 28, beta = 8/3, epsilon = 0.1.
# In this case the system is well known to be chaotic.
# =============================================================================
sigma_B = 1.94e-5
rho_B = 4.5e8
beta_B = 6.58e-16
epsilon_B = 0.1

LE_B, times_B, le_vals_B = compute_lyapunov(epsilon_B, sigma_B, rho_B, beta_B, t_max=100.0, dt=0.01)
print("\nChaotic Regime:")
print("Parameters: sigma=10, rho=28, beta=8/3, epsilon=0.1")
print("Maximum Lyapunov Exponent (LE): {:.3e}".format(LE_B))

# =============================================================================
# Plot Running Lyapunov Exponents for Comparison
# =============================================================================
plt.figure(figsize=(10, 5))
plt.plot(times_A, le_vals_A, label="Nearly Integrable (rho=15, e=1e-7)", color='blue')
plt.plot(times_B, le_vals_B, label="Chaotic (rho=28, e=0.1)", color='red')
plt.xlabel("Time")
plt.ylabel("Running LE Estimate")
plt.title("Comparison of Maximum Lyapunov Exponents")
plt.legend()
plt.grid(True)
plt.show()
