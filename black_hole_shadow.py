import numpy as np
import matplotlib.pyplot as plt

def effective_intensity(r, epsilon, r_s=1.0):
    # A toy model for an effective potential modification.
    return 1.0 - (r_s / r) + epsilon * np.exp(-r / r_s)

def simulate_black_hole_shadow(grid_size, r_s=1.0, epsilon=0.05):
    x = np.linspace(-5 * r_s, 5 * r_s, grid_size)
    y = np.linspace(-5 * r_s, 5 * r_s, grid_size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    image = effective_intensity(np.maximum(R, 1e-4), epsilon, r_s)
    return X, Y, image

grid_size = 300
X, Y, shadow_img = simulate_black_hole_shadow(grid_size)
plt.figure()
plt.imshow(shadow_img, extent=[-5, 5, -5, 5], cmap="inferno")
plt.xlabel("X (normalized)")
plt.ylabel("Y (normalized)")
plt.title("Simulated Black Hole Shadow (Cyclic-Invariant Model)")
plt.colorbar(label="Effective Intensity")
plt.show()

def simulate_pulsar_arrivals(total_time, baseline_interval, mod_period, mod_amplitude):
    # baseline pulse arrival times at a constant interval
    arrivals = np.arange(0, total_time, baseline_interval)
    # add a sinusoidal modulation to mimic cyclic effects
    modulated = arrivals + mod_amplitude * np.sin(2 * np.pi * arrivals / mod_period)
    return modulated

def analyze_timing_residuals(arrivals):
    residuals = np.diff(arrivals)
    f, Pxx = periodogram(residuals, fs=1.0)
    return f, Pxx

total_time = 3600  # seconds, e.g., 1 hour
baseline_interval = 1.0  # one pulse per second
mod_period = 2.0  # seconds cyclic period
mod_amplitude = 0.005  # seconds modulation amplitude

arrivals = simulate_pulsar_arrivals(total_time, baseline_interval, mod_period, mod_amplitude)
plt.figure()
plt.plot(arrivals, 'o-', markersize=2)
plt.xlabel("Pulse Index")
plt.ylabel("Arrival Time (s)")
plt.title("Simulated Pulsar Timing")
plt.show()

residuals = np.diff(arrivals)
f_res, Pxx_res = periodogram(residuals, fs=1.0)
plt.figure()
plt.plot(f_res, Pxx_res)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.title("Pulsar Timing Residual Spectrum")
plt.show()

