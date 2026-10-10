import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
num_glomes = 10
coupling_strength = 0.05
energy_threshold = 0.01

# Hamiltonian function with coupling
def hamiltonian(q, p, neighbors):
    H0 = 0.5 * p**2 + np.cos(q)  # Internal glome energy
    coupling = coupling_strength * sum(np.cos(q - qn) for qn in neighbors)
    return H0 + coupling

# Reeb vector field generator (canonical form)
def reeb_vector_field(q, p, neighbors):
    dqdt = p
    dpdt = -np.sin(q) - coupling_strength * sum(np.sin(q - qn) for qn in neighbors)
    return dqdt, dpdt

# Derivative function for all glomes
def multi_glome_dynamics(t, y):
    q = y[:num_glomes]
    p = y[num_glomes:]
    dqdt = np.zeros(num_glomes)
    dpdt = np.zeros(num_glomes)

    for i in range(num_glomes):
        # Periodic boundary conditions (glome lattice)
        neighbors = [q[(i - 1) % num_glomes], q[(i + 1) % num_glomes]]
        dqdt[i], dpdt[i] = reeb_vector_field(q[i], p[i], neighbors)

    return np.concatenate([dqdt, dpdt])

# Initial conditions: small random displacements near H ≈ 0
q0 = np.random.uniform(-0.1, 0.1, num_glomes)
p0 = np.random.uniform(-0.1, 0.1, num_glomes)
y0 = np.concatenate([q0, p0])

# Time span
T = 500
t_eval = np.linspace(0, T, 5000)

# Solve the ODE
sol = solve_ivp(multi_glome_dynamics, [0, T], y0, t_eval=t_eval, method='RK45')

# Track contact volume collapse
contact_volumes = []
for i in range(len(sol.t)):
    q = sol.y[:num_glomes, i]
    p = sol.y[num_glomes:, i]
    volumes = np.array([1 / (1 + abs(hamiltonian(q[j], p[j], [q[(j - 1) % num_glomes], q[(j + 1) % num_glomes]]))) for j in range(num_glomes)])
    contact_volumes.append(np.sum(volumes))

# Plot contact volume evolution
plt.figure(figsize=(12, 5))
plt.plot(sol.t, contact_volumes, label='Contact Volume Sum')
plt.axhline(y=0, color='r', linestyle='--', label='Collapse Threshold')
plt.xlabel('Time')
plt.ylabel('Contact Volume Sum')
plt.title('Contact Volume Collapse Near H = 0')
plt.legend()
plt.grid()
plt.show()

# Plot glome trajectories
plt.figure(figsize=(12, 6))
for i in range(num_glomes):
    plt.plot(sol.t, sol.y[i, :], label=f'Glome {i+1}')
plt.xlabel('Time')
plt.ylabel('q (Position)')
plt.title('Glome Trajectories (Parabolic and Escape Dynamics)')
plt.legend()
plt.grid()
plt.show()

# Participation ratio to quantify localization
mode_energies = np.zeros((num_glomes, len(sol.t)))
for i in range(len(sol.t)):
    q = sol.y[:num_glomes, i]
    p = sol.y[num_glomes:, i]
    for j in range(num_glomes):
        mode_energies[j, i] = hamiltonian(q[j], p[j], [q[(j - 1) % num_glomes], q[(j + 1) % num_glomes]])

participation_ratios = []
for i in range(len(sol.t)):
    energy_sq = np.sum(mode_energies[:, i]**2)
    energy_sum_sq = (np.sum(mode_energies[:, i]))**2
    PR = energy_sum_sq / (num_glomes * energy_sq + 1e-10)
    participation_ratios.append(PR)

plt.figure(figsize=(12, 5))
plt.plot(sol.t, participation_ratios, label='Participation Ratio')
plt.xlabel('Time')
plt.ylabel('Participation Ratio')
plt.title('Localization of Energy Across Glomes')
plt.legend()
plt.grid()
plt.show()

print("Simulation completed: higher-dimensional glome coupling, parabolic escape, and localization dynamics computed.")
