import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram, find_peaks

def simulate_gw_echo(T_echo, num_echoes, dt, amplitude=1.0, decay=0.8):
    t_total = num_echoes * T_echo * 2
    t = np.arange(0, t_total, dt)
    gw_signal = np.zeros_like(t)
    omega = 2 * np.pi / T_echo
    base_signal = amplitude * np.exp(-t / 5.0) * np.sin(omega * t)
    for k in range(num_echoes):
        delay = k * T_echo
        shift = int(delay / dt)
        echo = np.pad(base_signal, (shift, 0), mode='constant')[:len(t)] * (decay ** k)
        gw_signal += echo
    return t, gw_signal

# Simulation parameters
T_echo = 0.1  # echo period in seconds (scale accordingly)
num_echoes = 5
dt = 0.001

t, gw_signal = simulate_gw_echo(T_echo, num_echoes, dt)
plt.figure()
plt.plot(t, gw_signal)
plt.xlabel("Time (s)")
plt.ylabel("Strain")
plt.title("Simulated GW Echo Signal")
plt.show()

f, Pxx = periodogram(gw_signal, fs=1/dt)
plt.figure()
plt.semilogy(f, Pxx)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density")
plt.title("GW Echo Signal: Periodogram")
plt.show()

peaks, _ = find_peaks(Pxx, height=np.max(Pxx)*0.1)
if peaks.size:
    detected_freq = f[peaks[0]]
    detected_period = 1 / detected_freq
    print("Detected echo period: {:.4f} s".format(detected_period))
else:
    print("No significant echo period detected.")


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

