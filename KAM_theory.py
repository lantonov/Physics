#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =============================================================================
# Modified Lorenz System with Perturbations
#
# We consider a system of the form:
#
#   dx/dt = sigma*(y - x) + epsilon * F_x(x, y, z)
#   dy/dt = x*(rho - z) - y + epsilon * F_y(x, y, z)
#   dz/dt = x*y - beta*z + epsilon * F_z(x, y, z)
#
# Here, F_x, F_y, F_z are additional forcing terms that represent the 
# effects of metric (and nonlinearity) perturbations that arise from reducing 
# our 4D Hamiltonian (with constant spacetime norm) via normal form methods.
#
# In our toy model we choose:
#
#   F_x = sin(x+y+z)
#   F_y = cos(x-z)
#   F_z = sin(z)
#
# The parameter epsilon controls the perturbation strength. For the solar 
# system, epsilon will be extremely small (e.g. 1e-7), while near a black hole
# epsilon is taken to be moderate (e.g. 0.1), in which case chaos is expected.
#
# We use standard Lorenz parameters sigma, rho, beta:
#   sigma = 10, rho = 28, beta = 8/3 ~ 2.667.
# =============================================================================

def lorenz_modified(t, state, sigma, rho, beta, epsilon):
    x, y, z = state
    # Standard Lorenz terms:
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    # Additional forcing terms:
    Fx = np.sin(x + y + z)
    Fy = np.cos(x - z)
    Fz = np.sin(z)
    # Add perturbations scaled by epsilon:
    dxdt += epsilon * Fx
    dydt += epsilon * Fy
    dzdt += epsilon * Fz
    return [dxdt, dydt, dzdt]

# =============================================================================
# Helper function to integrate and plot the system
# =============================================================================
def simulate_and_plot(epsilon, title_suffix=''):
    # Define Lorenz parameters:
    sigma = 10.0
    rho = 28.0
    beta = 8.0/3.0

    # Time span for simulation:
    t0, tf = 0, 50
    t_eval = np.linspace(t0, tf, 5000)
    
    # Initial condition (chosen to be near a typical orbit of the Lorenz system)
    state0 = [1.0, 1.0, 20.0]
    
    # Integrate the ODE using solve_ivp.
    sol = solve_ivp(lorenz_modified, (t0, tf), state0, t_eval=t_eval,
                    args=(sigma, rho, beta, epsilon), rtol=1e-9, atol=1e-10)
    
    # Produce a table (sampled every ~5 time units)
    sample_indices = np.linspace(0, len(t_eval)-1, 11, dtype=int)
    table_data = []
    for i in sample_indices:
        t_val = sol.t[i]
        x_val, y_val, z_val = sol.y[:, i]
        table_data.append([t_val, x_val, y_val, z_val])
    
    # Print table header & data:
    print(f"Simulation Table for epsilon = {epsilon} {title_suffix}")
    print("{:<8} {:<10} {:<10} {:<10}".format("Time", "x(t)", "y(t)", "z(t)"))
    for row in table_data:
        print("{:<8.3f} {:<10.3f} {:<10.3f} {:<10.3f}".format(*row))
    
    # Plot the attractor (3D trajectory)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(sol.y[0], sol.y[1], sol.y[2], lw=0.75)
    ax.set_title("Modified Lorenz Attractor " + title_suffix)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()
    
    # Also plot a 2D projection for clarity
    plt.figure(figsize=(8,6))
    plt.plot(sol.y[0], sol.y[2], lw=0.75)
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("Projection (x vs. z) " + title_suffix)
    plt.grid(True)
    plt.show()

# =============================================================================
# Simulate for a small perturbation (solar system-like: epsilon ~ 1e-7)
# =============================================================================
simulate_and_plot(1e-7, title_suffix=": Nearly Integrable (Solar System Regime)")

# =============================================================================
# Simulate for a larger perturbation (black hole-like: epsilon ~ 0.1)
# =============================================================================
simulate_and_plot(0.1, title_suffix=": Chaotic (Black Hole Regime)")
