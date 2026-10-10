import numpy as np
import matplotlib.pyplot as plt

# Time array (simulate inspiral)
t = np.linspace(0, 50, 10000)

# Standard GR phase (chirping signal)
Phi_GR = 20 * t ** (5/8)

# Fiber-induced phase modulation
epsilon = 0.01  # Small amplitude
omega_f = 1.5   # Fiber mode frequency
phi_f = epsilon * np.cos(omega_f * t)

# Composite waveform
h = np.cos(Phi_GR + phi_f)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(t, h, label='Contact-Extended Gravitational Waveform')
plt.xlabel('Time')
plt.ylabel('Strain')
plt.title('Gravitational Waveform with Contact-Induced Fiber Phase Modulation')
plt.legend()
plt.grid()
plt.show()

# Plot phase residuals
plt.figure(figsize=(12, 6))
plt.plot(t, phi_f, label='Fiber Phase Residual')
plt.xlabel('Time')
plt.ylabel('Phase Residual')
plt.title('Predicted Fiber Phase Modulation (Contact Geometry)')
plt.legend()
plt.grid()
plt.show()