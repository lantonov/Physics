import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def sinusoid(t, A, f, phi, C):
    return A * np.sin(2 * np.pi * f * t + phi) + C

# Suppose t and signal_no_dc (mean-removed signal) are defined already.
# Let’s simulate a noisy version for robustness:
fs = 1000
T_total = 5
t = np.linspace(0, T_total, int(T_total * fs), endpoint=False)
true_params = [0.5, 5.0, 0.3, 0.0]
signal = sinusoid(t, *true_params)
# Add a small Gaussian noise
noise_level = 0.01
signal_noisy = signal + np.random.normal(0, noise_level, len(signal))
signal_no_dc = signal_noisy - np.mean(signal_noisy)

# Number of bootstrap samples
num_bootstraps = 1000
bootstrap_params = []

for _ in range(num_bootstraps):
    indices = np.random.choice(len(t), size=len(t), replace=True)
    t_sample = t[indices]
    sig_sample = signal_no_dc[indices]
    try:
        params, _ = curve_fit(sinusoid, t_sample, sig_sample, p0=[0.5, 5.0, 0.0, 0.0])
        bootstrap_params.append(params)
    except RuntimeError:
        # Skip samples that do not converge:
        continue

bootstrap_params = np.array(bootstrap_params)
param_means = np.mean(bootstrap_params, axis=0)
param_std = np.std(bootstrap_params, axis=0)

print("Bootstrap parameter estimates:")
print("Amplitude = {:.4f} +/- {:.4f}".format(param_means[0], param_std[0]))
print("Frequency = {:.4f} Hz +/- {:.4f} Hz".format(param_means[1], param_std[1]))
print("Phase = {:.4f} +/- {:.4f} radians".format(param_means[2], param_std[2]))
print("Offset = {:.4e} +/- {:.4e}".format(param_means[3], param_std[3]))

# Optionally, plot histograms of the bootstrap distributions:
plt.figure(figsize=(10,8))
titles = ["Amplitude", "Frequency", "Phase", "Offset"]
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.hist(bootstrap_params[:, i], bins=30, alpha=0.7)
    plt.title(titles[i])
plt.tight_layout()
plt.show()
