import numpy as np
from scipy.signal import periodogram, find_peaks
import matplotlib.pyplot as plt

# Suppose "signal_windowed" is your preprocessed, mean-subtracted, windowed time series.
# For demonstration, we re-simulate a test signal:
fs = 1000         # sampling frequency in Hz
T_total = 5       
t = np.linspace(0, T_total, int(T_total * fs), endpoint=False)
mod_freq = 5      # modulation frequency in Hz
mod_amp = 0.5     # modulation amplitude

signal = mod_amp * np.sin(2 * np.pi * mod_freq * t)
signal = signal - np.mean(signal)  # remove DC
# Optionally, apply a window (here, Hann window)
window = np.hanning(len(signal))
signal_windowed = signal * window

# Compute the periodogram.
f, Pxx = periodogram(signal_windowed, fs)

# Find peaks (ignoring the zero-frequency peak, which should be small after mean subtraction).
peaks, properties = find_peaks(Pxx, height=np.max(Pxx) * 0.1)

plt.semilogy(f, Pxx)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density")
plt.title("Periodogram with Detected Peaks")
plt.show()

# Extract and print peak information.
if peaks.size > 0:
    peak_freqs = f[peaks]
    peak_heights = properties["peak_heights"]
    print("Detected peak frequencies (Hz):", peak_freqs)
    print("Peak heights:", peak_heights)
else:
    print("No significant peaks detected.")

# For instance, you expect a peak at 5 Hz. Compare this with your predifined mod_freq.
