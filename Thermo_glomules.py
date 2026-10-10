import numpy as np
import matplotlib.pyplot as plt

# Updated toy model: N=3 glomules with distinct Θ-energies
epsilon = np.array([0.8, 1.2, 0.9])
epsilon_bind = 3.5  # binding threshold

def partition_function(T):
    beta = 1 / T
    # Gas state: all independent glomules
    Z_gas = np.exp(-beta * epsilon.sum())
    # Clustered state: entire set coalesces if allowed
    if epsilon.sum() <= epsilon_bind:
        Z_cluster = np.exp(-beta * (epsilon.sum() - epsilon_bind))
    else:
        Z_cluster = 0
    return Z_gas + Z_cluster, Z_cluster / (Z_gas + Z_cluster)

# Temperature range
T_vals = np.linspace(0.1, 5.0, 100)
Z_vals = np.zeros_like(T_vals)
phi_eq = np.zeros_like(T_vals)

for i, T in enumerate(T_vals):
    Z_vals[i], phi_eq[i] = partition_function(T)

# 1. Plot Partition Function vs Temperature
plt.figure()
plt.plot(T_vals, Z_vals)
plt.xlabel('Temperature T')
plt.ylabel('Partition Function Z')
plt.title('Updated Z vs T for N=3 Glomules')

# 2. Plot Equilibrium Cluster Fraction vs Temperature
plt.figure()
plt.plot(T_vals, phi_eq)
plt.xlabel('Temperature T')
plt.ylabel('Equilibrium Cluster Fraction φ_eq')
plt.title('φ_eq vs T with Binding Threshold ε_bind=2.5')

# 3. Quench simulation: from T_i to T_f
T_i, T_f = 0.5, 3.0
phi_i = np.interp(T_i, T_vals, phi_eq)
phi_target = np.interp(T_f, T_vals, phi_eq)

t = np.linspace(0, 10, 200)
tau = 2.0  # relaxation timescale
phi_t = phi_target + (phi_i - phi_target) * np.exp(-t / tau)

plt.figure()
plt.plot(t, phi_t)
plt.xlabel('Time t')
plt.ylabel('φ(t)')
plt.title(f'Relaxation after Quench {T_i}→{T_f}')

# 4. Berry connection with varying polar angle α(θ)
theta = np.linspace(0, 2*np.pi, 200)
alpha_theta = (np.pi / 4) + 0.3 * np.sin(theta)  # varying tilt
A_theta = (1 - np.cos(alpha_theta)) / 2
berry_phase = np.trapz(A_theta, theta)

plt.figure()
plt.plot(theta, A_theta)
plt.axhline(berry_phase / (2*np.pi), linestyle='--')
plt.xlabel('θ (fiber coordinate)')
plt.ylabel('Berry Connection A_θ(θ)')
plt.title(f'Berry Connection with Varying α(θ)\nTotal phase = {berry_phase:.2f}')

plt.show()
