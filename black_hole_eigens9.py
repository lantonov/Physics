import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram

# Assume that you have already computed the physical eigenfrequencies, omega_phys,
# and that from your sweep you have:
fundamental_mode = 0.37149620984292353  # candidate fundamental, using the real part
first_overtone = 0.37196057292828066     # candidate overtone

# For demonstration, choose amplitudes and phases.
A1 = 1.0      # amplitude of the fundamental mode
A2 = 0.5      # amplitude of the overtone (you may adjust this)
phi1 = 0.0    # phase for the fundamental
phi2 = 0.0    # phase for the overtone

# Also, we include the damping by using their imaginary parts:
# (here your computed imaginary parts are extremely small, 
# so you may want to insert a typical damping value for illustration such as -0.089 for the fundamental)
# For illustration, we force the fundamental to have the published damping:
omega1 = fundamental_mode - 0.089j  # published fundamental, if you wish
omega2 = first_overtone - 0.20j       # example value for the overtone (usually with higher damping)

# Define the time base for your composite signal.
fs = 1000         # sampling frequency (Hz)
T_sim = 1.0       # duration in seconds
t = np.linspace(0, T_sim, int(fs * T_sim), endpoint=False)

# Construct the composite ringdown signal:
#  h(t) = A1 * exp(-i*(omega1*t + phi1)) + A2 * exp(-i*(omega2*t + phi2))
h_t = A1 * np.exp(-1j * (omega1 * t + phi1)) + A2 * np.exp(-1j * (omega2 * t + phi2))

# Plot the time-domain signal (real and imaginary parts)
plt.figure(figsize=(10, 6))
plt.plot(t, np.real(h_t), label="Real part")
plt.plot(t, np.imag(h_t), label="Imaginary part", linestyle="--")
plt.xlabel("Time (s)")
plt.ylabel("h(t)")
plt.title("Composite Ringdown Signal")
plt.legend()
plt.grid(True)
plt.show()

# Compute and plot the spectrum.
f_vals, Pxx = periodogram(h_t, fs=fs)
plt.figure(figsize=(10, 6))
plt.semilogy(f_vals, Pxx)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density")
plt.title("Periodogram of Composite Ringdown Signal")
plt.grid(True)
plt.show()
