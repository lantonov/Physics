#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def lorenz_modified(t, state, sigma, rho, beta, epsilon):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    # Additional forcing mimicking metric perturbations:
    Fx = np.sin(x + y + z)
    Fy = np.cos(x - z)
    Fz = np.sin(z)
    dxdt += epsilon * Fx
    dydt += epsilon * Fy
    dzdt += epsilon * Fz
    return [dxdt, dydt, dzdt]

def poincare_section(epsilon, num_crossings=500, z_plane=15.0, t_max=500):
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    t0, tf = 0, t_max
    t_eval = np.linspace(t0, tf, 200000)  # even finer resolution
    state0 = [1.0, 1.0, z_plane]  # choose z = z_plane initially
    
    sol = solve_ivp(lorenz_modified, (t0, tf), state0, t_eval=t_eval,
                    args=(sigma, rho, beta, epsilon), rtol=1e-9, atol=1e-10)
    
    x, y, z, t = sol.y[0], sol.y[1], sol.y[2], sol.t
    poincare_x = []
    poincare_y = []
    poincare_t = []  # record crossing time for color-coding

    for i in range(len(z)-1):
        if z[i] < z_plane and z[i+1] >= z_plane:  # upward crossing
            alpha = (z_plane - z[i]) / (z[i+1]-z[i])
            x_cross = x[i] + alpha * (x[i+1]-x[i])
            y_cross = y[i] + alpha * (y[i+1]-y[i])
            t_cross = t[i] + alpha * (t[i+1]-t[i])
            poincare_x.append(x_cross)
            poincare_y.append(y_cross)
            poincare_t.append(t_cross)
            if len(poincare_x) >= num_crossings:
                break
    return np.array(poincare_x), np.array(poincare_y), np.array(poincare_t)

# Poincare section for nearly integrable case (epsilon small)
epsilon_solar = 1e-7
px_solar, py_solar, pt_solar = poincare_section(epsilon_solar)
# Poincare section for chaotic case (epsilon larger)
epsilon_black = 0.1
px_black, py_black, pt_black = poincare_section(epsilon_black)

# Plotting with color-coding by crossing time:
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

sc0 = axs[0].scatter(px_solar, py_solar, c=pt_solar, cmap='viridis', s=1)
axs[0].set_title("Poincare Section: Nearly Integrable (Solar System)\n(epsilon = 1e-7)")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[0].grid(True)
fig.colorbar(sc0, ax=axs[0], label='Crossing Time')

sc1 = axs[1].scatter(px_black, py_black, c=pt_black, cmap='inferno', s=1)
axs[1].set_title("Poincare Section: Chaotic (Black Hole Regime)\n(epsilon = 0.1)")
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")
axs[1].grid(True)
fig.colorbar(sc1, ax=axs[1], label='Crossing Time')

plt.tight_layout()
plt.show()
