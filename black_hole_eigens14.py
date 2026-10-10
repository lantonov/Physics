import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram

#--------------------------------------------------------------------------
# Candidate frequencies
# Fundamental mode (from your extraction):
omega1 = 0.37337764545009466 - 0.08924997006612352j

# First overtone: Since our extraction did not yield a distinct overtone,
# we adopt a literature value (e.g., from Berti et al.) for Schwarzschild, ℓ=2:
omega2 = 0.34671 - 0.27391j  # typical published value for the first overtone

#--------------------------------------------------------------------------
# Define simulation parameters
fs = 1000         # Sampling frequency (samples per unit time)
T_sim = 2.0       # Total simulation duration in time units
t = np.linspace(0, T_sim, int(fs*T_sim), endpoint=False)  # time vector

#--------------------------------------------------------------------------
# Choose amplitudes and phases for the two modes.
# You may adjust these as desired.
A1 = 1.0           # Amplitude for the fundamental mode
A2 = 1.0           # Amplitude for the first overtone
phi1 = 10000         # Phase for the fundamental mode (in radians)
phi2 = 10000         # Phase for the first overtone

#--------------------------------------------------------------------------
# Construct the composite ringdown signal.
# Time dependence: h(t) = A1 * exp[-i*(omega1*t + phi1)] + A2 * exp[-i*(omega2*t + phi2)]
h_t = A1 * np.exp(-1j*(omega1*t + phi1)) + A2 * np.exp(-1j*(omega2*t + phi2))

#--------------------------------------------------------------------------
# Plot the time domain signal: its real and imaginary parts.
plt.figure(figsize=(10, 6))
plt.plot(t, np.real(h_t), label="Real part")
plt.plot(t, np.imag(h_t), label="Imaginary part", linestyle="--")
plt.xlabel("Time")
plt.ylabel("h(t)")
plt.title("Composite Ringdown Signal")
plt.legend()
plt.grid(True)
plt.show()

#--------------------------------------------------------------------------
# Compute and plot the periodogram to see the frequency content.
# Note: Since the input signal is complex, scipy's periodogram will return
# a two-sided spectrum.
f_vals, Pxx = periodogram(h_t, fs=fs)
plt.figure(figsize=(10, 6))
plt.semilogy(f_vals, Pxx)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density")
plt.title("Periodogram of the Composite Ringdown Signal")
plt.grid(True)
plt.show()

#--------------------------------------------------------------------------
# Optionally, you can also plot the amplitude envelope of the signal.
amplitude_envelope = np.abs(h_t)
plt.figure(figsize=(10, 6))
plt.plot(t, amplitude_envelope, 'r-', label="Amplitude Envelope")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Amplitude Envelope of the Composite Signal")
plt.legend()
plt.grid(True)
plt.show()
