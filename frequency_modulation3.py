import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram, find_peaks
from scipy.optimize import curve_fit

# Constants in SI units
c = 3e8           # Speed of light [m/s]
G = 6.67e-11      # Gravitational constant [m^3/kg/s^2]

def characteristic_frequency(M):
    """
    Calculate the characteristic frequency of the cyclic invariant as:
    f = c^3 / (2 * G * M)
    
    Parameters:
        M : float
            Black hole mass in kg.
    
    Returns:
        f : float
            Characteristic frequency in Hz.
    """
    return c**3 / (2 * G * M)

def simulate_sinusoid(frequency, amplitude, t):
    """
    Generate a sinusoidal signal with a given frequency and amplitude.
    
    Parameters:
        frequency : float
            Frequency of the sinusoid (Hz).
        amplitude : float
            Amplitude of the sinusoid.
        t : numpy.ndarray
            Time vector.
    
    Returns:
        signal : numpy.ndarray
            The sinusoidal signal.
    """
    return amplitude * np.sin(2 * np.pi * frequency * t)

# Set up simulation parameters
fs = 10000         # Sampling frequency in Hz
T = 1.0            # Duration of the signal in seconds
t = np.linspace(0, T, int(T * fs), endpoint=False)

# First mass value (for instance, 6e31 kg ~ 30 solar masses)
M1 = 6e31  
f1 = characteristic_frequency(M1)
print(f"For mass M1 = {M1:.3e} kg, predicted characteristic frequency: {f1:.3e} Hz")

# Generate signal for M1
signal1 = simulate_sinusoid(f1, amplitude=1.0, t=t)

# Compute periodogram for signal1
f_vals1, Pxx1 = periodogram(signal1, fs)
peaks1, _ = find_peaks(Pxx1, height=np.max(Pxx1) * 0.1)
if peaks1.size > 0:
    recovered_freq1 = f_vals1[peaks1[0]]
    print(f"Recovered frequency from FFT for M1: {recovered_freq1:.3e} Hz")
else:
    print("No clear peak detected for M1!")

# Now double the mass: M2 = 2 * M1
M2 = 2 * M1
f2 = characteristic_frequency(M2)
print(f"\nFor mass M2 = {M2:.3e} kg, predicted characteristic frequency: {f2:.3e} Hz")

# Generate signal for M2
signal2 = simulate_sinusoid(f2, amplitude=1.0, t=t)

# Compute periodogram for signal2
f_vals2, Pxx2 = periodogram(signal2, fs)
peaks2, _ = find_peaks(Pxx2, height=np.max(Pxx2) * 0.1)
if peaks2.size > 0:
    recovered_freq2 = f_vals2[peaks2[0]]
    print(f"Recovered frequency from FFT for M2: {recovered_freq2:.3e} Hz")
else:
    print("No clear peak detected for M2!")

# Plot example segments of the two signals for visual inspection
plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plt.plot(t[:1000], signal1[:1000])
plt.title(f"Signal for M1: f = {f1:.3e} Hz")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.subplot(2,1,2)
plt.plot(t[:1000], signal2[:1000])
plt.title(f"Signal for M2: f = {f2:.3e} Hz (should be roughly half of f for M1)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()
