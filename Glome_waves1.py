import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

# Simulation parameters
N_glomes = 10
N_modes = 5
t_max = 200
dt = 0.05
t = np.arange(0, t_max, dt)
coupling_speed = 2.0  # finite propagation speed

# Initial conditions
np.random.seed(42)
frequencies = np.random.uniform(0.5, 2.0, (N_glomes, N_modes))
couplings = np.random.normal(0, 0.05, (N_glomes, N_glomes, N_modes))
off_diagonal_disorder = np.random.normal(0, 0.02, (N_glomes, N_glomes, N_modes))
dynamic_disorder_amplitude = 0.02
positions = np.random.uniform(-1, 1, (N_glomes, 2))

# Initialize amplitudes and energies
amplitudes = np.zeros((N_glomes, N_modes, len(t)))
energies = np.zeros((N_glomes, len(t)))

# Initial energy in glome 0, mode 0
amplitudes[0, 0, 0] = 1.0

# Distance matrix and communication delays
distances = np.zeros((N_glomes, N_glomes))
delays = np.zeros((N_glomes, N_glomes))
for i in range(N_glomes):
    for j in range(N_glomes):
        distances[i, j] = np.linalg.norm(positions[i] - positions[j])
        delays[i, j] = distances[i, j] / coupling_speed

# Simulation loop
for ti in range(1, len(t)):
    for i in range(N_glomes):
        for m in range(N_modes):
            omega = frequencies[i, m]
            damping = 0.01 * amplitudes[i, m, ti - 1]

            # Non-linear self-interaction (cubic)
            nonlinear_term = -0.005 * amplitudes[i, m, ti - 1] ** 3

            # Dynamic disorder
            dynamic_disorder = dynamic_disorder_amplitude * np.random.randn()

            coupling_sum = 0.0
            for j in range(N_glomes):
                if i != j:
                    delay_steps = int(delays[i, j] / dt)
                    delayed_index = max(ti - delay_steps, 0)
                    coupling_sum += (
                        (couplings[i, j, m] + off_diagonal_disorder[i, j, m]) * amplitudes[j, m, delayed_index]
                    )

            amplitudes[i, m, ti] = (
                amplitudes[i, m, ti - 1]
                + dt * (
                    -omega ** 2 * amplitudes[i, m, ti - 1]
                    + coupling_sum
                    + nonlinear_term
                    + dynamic_disorder
                    - damping
                )
            )
            energies[i, ti] += amplitudes[i, m, ti] ** 2

# Participation ratios
participation_ratios = np.zeros(len(t))
for ti in range(len(t)):
    total_energy = np.sum(energies[:, ti])
    if total_energy > 0:
        participation_ratios[ti] = 1 / np.sum((energies[:, ti] / total_energy) ** 2)

# Spectral analysis
spectra = []
for i in range(N_glomes):
    fft_result = fft(np.sum(amplitudes[i], axis=0))
    spectra.append(np.abs(fft_result[:len(fft_result)//2]))

# Plot energy localization
plt.figure(figsize=(12, 6))
for i in range(N_glomes):
    plt.plot(t, energies[i], label=f'Glome {i}')
plt.xlabel('Time')
plt.ylabel('Energy per Glome')
plt.title('Energy Localization Across Glomes')
plt.legend()
plt.grid()
plt.show()

# Plot participation ratios
plt.figure(figsize=(10, 4))
plt.plot(t, participation_ratios)
plt.xlabel('Time')
plt.ylabel('Participation Ratio')
plt.title('Localization Measure Over Time')
plt.grid()
plt.show()

# Plot example spectra
plt.figure(figsize=(12, 6))
for i in range(min(N_glomes, 4)):
    plt.plot(np.linspace(0, 1 / (2 * dt), len(spectra[i])), spectra[i], label=f'Glome {i}')
plt.xlabel('Frequency')
plt.ylabel('Spectral Amplitude')
plt.title('Spectral Analysis of Selected Glomes')
plt.legend()
plt.grid()
plt.show()
