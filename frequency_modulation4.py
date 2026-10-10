import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram, find_peaks
from scipy.optimize import curve_fit

# Constants (SI units)
c = 3e8           # Speed of light in m/s
G = 6.67e-11      # Gravitational constant in m^3/kg/s^2

def characteristic_frequency(M):
    """
    Calculate the characteristic frequency from the invariant potential scale:
        f = c^3 / (2 * G * M)
    M : mass in kg
    Returns f in Hz.
    """
    return c**3 / (2 * G * M)

def simulate_sinusoid(frequency, amplitude, t):
    """
    Generate a sinusoidal signal with a given frequency and amplitude.
    """
    return amplitude * np.sin(2 * np.pi * frequency * t)

def simulate_double_sinusoid(f1, f2, amp1, amp2, t):
    """
    Generate a signal that is a superposition of two sinusoids with frequencies f1 and f2.
    """
    return amp1 * np.sin(2 * np.pi * f1 * t) + amp2 * np.sin(2 * np.pi * f2 * t)

# --- PART 1: Mass-Sweep and Scaling Demo ---

def mass_sweep_demo():
    fs = 10000          # Sampling frequency in Hz
    T = 1.0             # Signal duration in seconds
    t = np.linspace(0, T, int(T * fs), endpoint=False)
    
    # Define a range of masses (in kg). (For example, from 4e31 to 1.2e32)
    mass_values = np.linspace(4e31, 1.2e32, 10)
    predicted_freqs = []
    recovered_freqs = []
    
    for M in mass_values:
        # Calculate predicted frequency f = c^3/(2GM)
        f_pred = characteristic_frequency(M)
        predicted_freqs.append(f_pred)
        
        # Simulate a sinusoidal signal at that frequency (amplitude arbitrary, say 1)
        signal = simulate_sinusoid(f_pred, amplitude=1.0, t=t)
        # Noisy signals can be simulated by adding gaussian noise if desired.
        
        # Perform Fourier analysis
        f_vals, Pxx = periodogram(signal, fs)
        # Identify the peak frequency (ignoring DC)
        peaks, _ = find_peaks(Pxx[1:], height=np.max(Pxx[1:])*0.1)
        if peaks.size > 0:
            recovered = f_vals[peaks[0] + 1]  # add 1 since we skipped first bin
        else:
            recovered = np.nan
        recovered_freqs.append(recovered)
        print(f"M = {M:.3e} kg: predicted f = {f_pred:.3e} Hz, recovered f = {recovered:.3e} Hz")
    
    # Plot predicted vs. recovered frequency vs mass
    plt.figure(figsize=(8, 6))
    plt.plot(mass_values, predicted_freqs, 'bo-', label="Predicted frequency")
    plt.plot(mass_values, recovered_freqs, 'rs--', label="Recovered frequency")
    plt.xlabel("Mass (kg)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Characteristic Frequency vs. Mass")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot f vs 1/M to show inverse scaling:
    inv_mass = 1.0 / mass_values
    plt.figure(figsize=(8, 6))
    plt.plot(inv_mass, predicted_freqs, 'bo-', label="Predicted frequency")
    plt.xlabel("1/M (1/kg)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Inverse Scaling: Frequency vs. Inverse Mass")
    plt.legend()
    plt.grid(True)
    plt.show()

# --- PART 2: Extra Frequency Simulation ---

def double_frequency_demo(M, extra_ratio=0.5):
    """
    Simulate a signal that is the sum of two sinusoids:
      f1 = c^3 / (2GM)  (dominant frequency)
      f2 = extra_ratio * f1 (secondary frequency, e.g., 50%).
    Fit the signal and plot the periodogram.
    """
    fs = 10000         # Sampling frequency in Hz
    T = 1.0            # Duration (seconds)
    t = np.linspace(0, T, int(T * fs), endpoint=False)
    
    f1 = characteristic_frequency(M)
    f2 = extra_ratio * f1  # Secondary frequency
    print(f"For mass M = {M:.3e} kg, f1 = {f1:.3e} Hz, f2 = {f2:.3e} Hz")
    
    amplitude1 = 1.0   # amplitude of dominant sinusoid
    amplitude2 = 0.5   # amplitude of secondary sinusoid
    signal = simulate_double_sinusoid(f1, f2, amplitude1, amplitude2, t)
    
    # Fourier analysis to extract both peaks.
    f_vals, Pxx = periodogram(signal, fs)
    peaks, properties = find_peaks(Pxx[1:], height=np.max(Pxx[1:])*0.1)
    # Adjust index (ignore DC component)
    peak_freqs = f_vals[peaks + 1]
    print("Extracted peak frequencies: ", peak_freqs)
    
    # Plot the time-domain signal segment.
    plt.figure(figsize=(10, 6))
    plt.plot(t[:1000], signal[:1000])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Composite Signal (Double Frequency)")
    plt.show()
    
    # Plot the periodogram
    plt.figure(figsize=(10, 6))
    plt.semilogy(f_vals, Pxx, label="PSD")
    plt.scatter(peak_freqs, Pxx[peaks+1], color='red', zorder=5, label="Detected Peaks")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.title("Periodogram of Composite Signal")
    plt.legend()
    plt.show()
    
    # Optional: Fit a two-frequency model using non-linear sinusoids.
    # For simplicity, we can display that two peaks show clearly.
    return peak_freqs

# --- Run the Demonstrations ---

if __name__ == "__main__":
    print("== MASS SWEEP DEMO ==")
    mass_sweep_demo()
    
    print("\n== DOUBLE FREQUENCY DEMO ==")
    # Choose a representative mass (e.g., M1 = 6e31 kg)
    mass_example = 6e31
    double_frequency_demo(mass_example, extra_ratio=0.5)
