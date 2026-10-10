import numpy as np
import matplotlib.pyplot as plt

# Time parameters
t = np.linspace(0, 10, 1000)
decay_time = 3.0  # decay happens at t = 3

# Neutron glome volume (contact volume)
neutron_volume = np.exp(-0.5 * t)  # Exponential collapse

# W- boson mediator field (emerges at decay time)
W_boson_field = np.exp(-20 * (t - decay_time)**2)  # Sharp Gaussian pulse

# Proton and electron glome volume growth post-decay
proton_volume = np.zeros_like(t)
electron_volume = np.zeros_like(t)

# Growth rates (tunable)
proton_growth_rate = 0.5
electron_growth_rate = 0.3

# Post-decay dynamics
for i in range(len(t)):
    if t[i] > decay_time:
        proton_volume[i] = proton_growth_rate * (t[i] - decay_time)
        electron_volume[i] = electron_growth_rate * (t[i] - decay_time)

# Plotting
plt.figure(figsize=(12, 7))
plt.plot(t, neutron_volume, label='Neutron Glome Contact Volume', color='blue')
plt.plot(t, W_boson_field, label='W⁻ Boson Field', color='purple')
plt.plot(t, proton_volume, label='Proton Glome Volume', color='green')
plt.plot(t, electron_volume, label='Electron Glome Volume', color='red')

plt.axvline(decay_time, color='black', linestyle='--', label='Decay Trigger (t = {:.2f})'.format(decay_time))

plt.xlabel('Time')
plt.ylabel('Glome Volume / Field Strength')
plt.title('Theta-Glome Beta Decay Simulation (Neutron → Proton + Electron + W⁻)')
plt.legend()
plt.grid(True)
plt.show()
