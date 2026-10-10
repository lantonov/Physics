import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the sinusoidal model.
def sinusoid(t, A, f, phi, C):
    return A * np.sin(2 * np.pi * f * t + phi) + C

# Assume you have your time series 't' and the preprocessed signal 'signal'
# (already mean-subtracted and possibly windowed).
# For demonstration, we simulate a signal with modulation at 5 Hz:
fs = 1000  # Sampling frequency in Hz.
T_total = 5  # seconds.
t = np.linspace(0, T_total, int(T_total * fs), endpoint=False)
true_amp = 0.5  # True amplitude
true_freq = 5.0  # True frequency in Hz
true_phi = 0.3  # True phase (radians)
true_offset = 0.0
signal = true_amp * np.sin(2 * np.pi * true_freq * t + true_phi) + true_offset
# In a realistic case, your 'signal' comes from your simulation.
# Remove any residual DC:
signal = signal - np.mean(signal)

# Plot the time-domain signal:
plt.figure()
plt.plot(t, signal, label="Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Time-Domain Signal")
plt.legend()
plt.show()

# Provide an initial guess for parameters: amplitude, frequency, phase, offset.
initial_guess = [0.4, 5.0, 0.0, 0.0]
params, covariance = curve_fit(sinusoid, t, signal, p0=initial_guess)

A_fit, f_fit, phi_fit, C_fit = params
print("Fitted parameters:")
print("Amplitude =", A_fit)
print("Frequency =", f_fit, "Hz")
print("Phase =", phi_fit, "radians")
print("Offset =", C_fit)

# Plot the fitted model:
fitted_signal = sinusoid(t, *params)
plt.figure()
plt.plot(t, signal, label="Signal", alpha=0.6)
plt.plot(t, fitted_signal, label="Fitted Model", linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Sinusoidal Fit to the Signal")
plt.legend()
plt.show()
