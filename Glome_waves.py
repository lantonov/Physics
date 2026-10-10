import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.signal import welch

# Parameters
num_glomes = 20
num_modes = 5
time_steps = 10000
dt = 0.01

# Initialize frequencies and damping
frequencies = np.array([[1 + 0.1 * l + 0.05 * np.random.randn() for l in range(num_modes)] for _ in range(num_glomes)])
damping = 0.01 * np.ones((num_glomes, num_modes))

# Random coupling matrix (off-diagonal disorder)
coupling = np.zeros((num_glomes * num_modes, num_glomes * num_modes))
for i in range(num_glomes * num_modes):
    for j in range(i + 1, num_glomes * num_modes):
        if np.random.rand() < 0.2:  # sparse coupling
            coupling[i, j] = 0.02 * (1 + 0.2 * np.random.randn())
            coupling[j, i] = coupling[i, j]

# Communication delays
propagation_speed = 10.0
distances = np.abs(np.subtract.outer(np.arange(num_glomes * num_modes), np.arange(num_glomes * num_modes)))
delays = distances / propagation_speed

# Time-dependent coupling (dynamic disorder)
dynamic_coupling_strength = 0.005

# Initialize positions and velocities
positions = np.zeros((num_glomes * num_modes, time_steps))
velocities = np.zeros((num_glomes * num_modes, time_steps))

# Initial excitation
positions[num_glomes * 2] = 1.0

# Simulation loop
for t in range(1, time_steps - 1):
    dynamic_coupling = dynamic_coupling_strength * np.sin(0.05 * t * dt) * (np.random.rand(num_glomes * num_modes, num_glomes * num_modes) - 0.5)
    dynamic_coupling = (dynamic_coupling + dynamic_coupling.T) / 2  # ensure symmetry

    total_coupling = coupling + dynamic_coupling

    for i in range(num_glomes * num_modes):
        omega = frequencies[i // num_modes, i % num_modes]
        force = -omega**2 * positions[i, t] - damping[i // num_modes, i % num_modes] * velocities[i, t]

        for j in range(num_glomes * num_modes):
            if total_coupling[i, j] != 0 and t - int(delays[i, j] / dt) >= 0:
                delay_index = int(delays[i, j] / dt)
                force += total_coupling[i, j] * (positions[j, t - delay_index] - positions[i, t])

        acceleration = force
        velocities[i, t + 1] = velocities[i, t] + acceleration * dt
        positions[i, t + 1] = positions[i, t] + velocities[i, t + 1] * dt

# Compute energy localization over time
energy_per_glome = np.zeros((num_glomes, time_steps))
for g in range(num_glomes):
    for l in range(num_modes):
        idx = g * num_modes + l
        energy_per_glome[g] += 0.5 * (velocities[idx]**2 + (frequencies[g, l]**2) * (positions[idx]**2))

# Participation ratio
participation_ratio = np.zeros(time_steps)
for t in range(time_steps):
    norm = np.sum(energy_per_glome[:, t])
    if norm > 0:
        participation_ratio[t] = 1.0 / np.sum((energy_per_glome[:, t] / norm)**2)

# Time-resolved mode decomposition
from scipy.fft import fft
mode_spectra = []
for g in range(num_glomes):
    for l in range(num_modes):
        idx = g * num_modes + l
        mode_spectrum = np.abs(fft(positions[idx]))[:time_steps // 2]
        mode_spectra.append(mode_spectrum)

# Plot results
fig, axs = plt.subplots(2, 2, figsize=(16, 10))

# Energy localization
for g in range(num_glomes):
    axs[0, 0].plot(np.arange(time_steps) * dt, energy_per_glome[g], label=f'Glome {g}')
axs[0, 0].set_title('Energy per Glome Over Time')
axs[0, 0].set_xlabel('Time')
axs[0, 0].set_ylabel('Energy')
axs[0, 0].legend()

# Participation ratio
axs[0, 1].plot(np.arange(time_steps) * dt, participation_ratio)
axs[0, 1].set_title('Participation Ratio Over Time')
axs[0, 1].set_xlabel('Time')
axs[0, 1].set_ylabel('Participation Ratio')

# Spectral analysis (example glome)
g_example = 0
for l in range(num_modes):
    axs[1, 0].plot(mode_spectra[g_example * num_modes + l], label=f'Mode {l}')
axs[1, 0].set_title(f'Spectral Analysis of Glome {g_example}')
axs[1, 0].set_xlabel('Frequency Bin')
axs[1, 0].set_ylabel('Amplitude')
axs[1, 0].legend()

# Energy map (spatial localization)
axs[1, 1].imshow(energy_per_glome, aspect='auto', origin='lower', extent=[0, time_steps * dt, 0, num_glomes])
axs[1, 1].set_title('Energy Localization Map')
axs[1, 1].set_xlabel('Time')
axs[1, 1].set_ylabel('Glome Index')

plt.tight_layout()
plt.show()
