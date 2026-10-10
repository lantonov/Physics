import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram, find_peaks

# Parameters for the test signal
fs = 1000         # Sampling frequency in Hz
T_total = 5       # Total time in seconds
t = np.linspace(0, T_total, int(T_total * fs), endpoint=False)
mod_freq = 5      # Modulation frequency in Hz
mod_amp = 0.5     # Modulation amplitude

# Create a base signal and add the sinusoidal modulation
base_signal = np.ones_like(t)  # constant baseline, can also add a trend if desired
modulation = mod_amp * np.sin(2 * np.pi * mod_freq * t)
signal = base_signal + modulation

# Plot the signal to verify the sinusoid
plt.plot(t, signal)
plt.xlabel("Time (s)")
plt.ylabel("Signal amplitude")
plt.title("Test Signal with Sinusoidal Modulation")
plt.show()

# Compute the periodogram
f, Pxx = periodogram(signal, fs)
plt.semilogy(f, Pxx)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density")
plt.title("Periodogram of the Test Signal")
plt.show()

# Detect peaks in the periodogram
peaks, _ = find_peaks(Pxx, height=np.max(Pxx)*0.1)
print("Peak Frequencies:", f[peaks])
