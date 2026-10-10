import numpy as np
import matplotlib.pyplot as plt

# Time parameters
t = np.linspace(0, 10, 1000)
decay_time = 3.0  # decay happens at t = 3

# Neutron glome volume (contact volume)
neutron_volume = np.exp(-0.5 * t)  # Exponential collapse

# W⁻ boson mediator field (emerges at decay time)
W_boson_field = np.exp(-20 * (t - decay_time)**2)  # Sharp Gaussian pulse

# Proton and electron glome volume growth post-decay
proton_volume = np.zeros_like(t)
electron_volume = np.zeros_like(t)
neutrino_volume = np.zeros_like(t)

# Residual weak potential coupling (simplified as exponential decay)
residual_coupling = np.zeros_like(t)

# Octonionic spin alignment (abstract measure: alignment factor oscillating)
spin_alignment = np.zeros_like(t)

# Growth rates (tunable)
proton_growth_rate = 0.5
electron_growth_rate = 0.3
neutrino_emission_rate = 0.2

for i in range(len(t)):
    if t[i] > decay_time:
        dt_post_decay = t[i] - decay_time
        proton_volume[i] = proton_growth_rate * dt_post_decay
        electron_volume[i] = electron_growth_rate * dt_post_decay
        neutrino_volume[i] = neutrino_emission_rate * dt_post_decay
        # Residual weak potential decays over time post-decay
        residual_coupling[i] = np.exp(-0.5 * dt_post_decay)
        # Octonionic spin alignment (oscillating pattern simulating spin precession)
        spin_alignment[i] = np.cos(5 * dt_post_decay) * np.exp(-0.2 * dt_post_decay)

# Plotting glome volumes and W⁻ field
plt.figure(figsize=(14, 8))
plt.plot(t, neutron_volume, label='Neutron Glome Contact Volume', color='blue')
plt.plot(t, W_boson_field, label='W⁻ Boson Field', color='purple')
plt.plot(t, proton_volume, label='Proton Glome Volume', color='green')
plt.plot(t, electron_volume, label='Electron Glome Volume', color='red')
plt.plot(t, neutrino_volume, label='Neutrino Glome Emission', color='orange')
plt.axvline(decay_time, color='black', linestyle='--', label='Decay Trigger (t = {:.2f})'.format(decay_time))
plt.xlabel('Time')
plt.ylabel('Glome Volume / Field Strength')
plt.title('Theta-Glome Beta Decay with Neutrino Emission and Weak Coupling')
plt.legend()
plt.grid(True)
plt.show()

# Plotting weak potential and spin alignment
plt.figure(figsize=(14, 6))
plt.plot(t, residual_coupling, label='Residual Weak Potential', color='brown')
plt.plot(t, spin_alignment, label='Octonionic Spin Alignment', color='cyan')
plt.axvline(decay_time, color='black', linestyle='--', label='Decay Trigger (t = {:.2f})'.format(decay_time))
plt.xlabel('Time')
plt.ylabel('Coupling Strength / Spin Alignment')
plt.title('Post-Decay Weak Potential and Spin Alignment Dynamics')
plt.legend()
plt.grid(True)
plt.show()
