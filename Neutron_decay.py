# Simulation of Beta Decay from Neutron Glome (S^7) in Theta-Glome Theory with Octonionic Coupling

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- Parameters ---
num_modes = 3  # u, d, d
coupling_strength = 1.0
leakage_threshold = 0.01  # Contact volume collapse threshold

# Octonionic-inspired SU(3) coupling matrix
OCT_SU3 = np.array([
    [0, 1, -1],
    [-1, 0, 1],
    [1, -1, 0]
])

# Initial energy distribution (neutron composite)
H0 = np.array([-0.6, -0.6, -0.6])  # Bound standing waves

# Contact volume dynamics: V_c ~ |H_total|
def contact_volume(H_modes):
    return np.abs(np.sum(H_modes))

# Octonionic SU(3) interaction
def octonionic_interaction(H_modes, t):
    dHdt = np.zeros_like(H_modes)
    for i in range(num_modes):
        for j in range(num_modes):
            if i != j:
                dHdt[i] += coupling_strength * OCT_SU3[i, j] * np.sin(H_modes[j] - H_modes[i])
    return dHdt

# Time evolution
T = np.linspace(0, 50, 5000)
H_modes = odeint(octonionic_interaction, H0, T)

# Contact volume over time
V_contact = np.array([contact_volume(H) for H in H_modes])

# Find decay moment when contact volume approaches zero
decay_index = np.argmin(np.abs(V_contact))
decay_time = T[decay_index]

# --- Plotting ---
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
for i in range(num_modes):
    plt.plot(T, H_modes[:, i], label=f'Mode {i+1}')
plt.axvline(x=decay_time, color='k', linestyle='--', label='Decay Threshold')
plt.xlabel('Time')
plt.ylabel('Mode Energy H')
plt.title('Mode Energy Evolution (Octonionic SU(3) Coupling)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(T, V_contact, label='Contact Volume |H_total|')
plt.axhline(y=leakage_threshold, color='r', linestyle='--', label='Leakage Threshold')
plt.axvline(x=decay_time, color='k', linestyle='--', label='Decay Time')
plt.xlabel('Time')
plt.ylabel('Contact Volume')
plt.title('Contact Volume Evolution and Decay Point')
plt.legend()

plt.tight_layout()
plt.show()

print(f"Decay occurs approximately at time t = {decay_time:.3f}")
