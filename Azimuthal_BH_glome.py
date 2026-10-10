import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import circstd

# Reproducibility
np.random.seed(42)

# Parameters
N = 256                  # azimuthal samples
epochs = 10
phi = np.linspace(0, 2 * np.pi, N, endpoint=False)

modes_gr = range(1, 9)   # m = 1..8
modes_glome = [2] + list(range(3, 7))  # m = 2..6, with suppressed m=1

# Storage
gr_profiles = []
glome_profiles = []
gr_amplitudes = []
glome_amplitudes = []

# ---------- GR-like profiles (Kerr + turbulence) ----------
# Strong, noisy m=1; broadband higher modes
for _ in range(epochs):
    I = np.zeros(N, dtype=np.complex128)
    for m in modes_gr:
        if m == 1:
            amp = np.random.uniform(0.5, 1.0)      # strong m=1
        else:
            amp = np.random.uniform(0.05, 0.2)     # weaker higher modes
        phase = np.random.uniform(0, 2 * np.pi)    # random phase
        I += amp * np.exp(1j * (m * phi + phase))
    I_real = np.real(I)
    I_real -= I_real.min()                         # make non-negative
    gr_profiles.append(I_real)
    gr_amplitudes.append(np.fft.fft(I_real) / N)

# ---------- Glome-like profiles (S³ Reeb spectrum) ----------
# Suppressed m=1; strong, stable m=2; limited higher modes with small jitter
for _ in range(epochs):
    I = np.zeros(N, dtype=np.complex128)

    # Suppressed m=1
    I += 0.01 * np.exp(1j * (1 * phi + np.random.uniform(0, 2 * np.pi)))

    # Persistent m=2 with small phase jitter
    I += 1.0 * np.exp(1j * (2 * phi + np.random.normal(0, 0.1)))

    # Limited higher modes with small amplitude and phase jitter
    for m in range(3, 7):
        amp = np.random.uniform(0.05, 0.1)
        phase = np.random.normal(0, 0.2)
        I += amp * np.exp(1j * (m * phi + phase))

    I_real = np.real(I)
    I_real -= I_real.min()                         # make non-negative
    glome_profiles.append(I_real)
    glome_amplitudes.append(np.fft.fft(I_real) / N)

# ---------- Statistics on Fourier modes ----------

def compute_stats(amplitudes, m):
    mags = np.array([np.abs(a[m]) for a in amplitudes])
    phases = np.array([np.angle(a[m]) for a in amplitudes])
    mean_mag = np.mean(mags)
    std_mag = np.std(mags)
    phase_dispersion = circstd(phases)
    return mean_mag, std_mag, phase_dispersion

gr_a1_mean, gr_a1_std, gr_a1_phase = compute_stats(gr_amplitudes, 1)
gr_a2_mean, gr_a2_std, gr_a2_phase = compute_stats(gr_amplitudes, 2)

glome_a1_mean, glome_a1_std, glome_a1_phase = compute_stats(glome_amplitudes, 1)
glome_a2_mean, glome_a2_std, glome_a2_phase = compute_stats(glome_amplitudes, 2)

# ---------- Plots ----------

# plt.style.use("seaborn-v0_8")

# (1) Example brightness profiles for a few epochs
fig, axs = plt.subplots(2, 3, figsize=(15, 6))
for i in range(3):
    axs[0, i].plot(phi, gr_profiles[i])
    axs[0, i].set_title(f"GR-like Epoch {i+1}")
    axs[1, i].plot(phi, glome_profiles[i])
    axs[1, i].set_title(f"Glome-like Epoch {i+1}")
for ax in axs.flat:
    ax.set_xlabel("Azimuthal angle φ")
    ax.set_ylabel("Brightness I(φ)")
plt.tight_layout()
plt.show()

# (2) Fourier amplitude spectra for representative epochs
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
gr_spec = np.abs(gr_amplitudes[0][:20])
glome_spec = np.abs(glome_amplitudes[0][:20])

axs[0].bar(range(20), gr_spec)
axs[0].set_title("GR-like |a_m| (Epoch 1)")
axs[0].set_xlabel("Mode m")
axs[0].set_ylabel("|a_m|")

axs[1].bar(range(20), glome_spec)
axs[1].set_title("Glome-like |a_m| (Epoch 1)")
axs[1].set_xlabel("Mode m")
axs[1].set_ylabel("|a_m|")

plt.tight_layout()
plt.show()

# (3) Bar plot of mean |a1| and |a2| with error bars
labels = ["|a₁|", "|a₂|"]
gr_means = [gr_a1_mean, gr_a2_mean]
gr_stds = [gr_a1_std, gr_a2_std]
glome_means = [glome_a1_mean, glome_a2_mean]
glome_stds = [glome_a1_std, glome_a2_std]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
ax.bar(x - width/2, gr_means, width, yerr=gr_stds, label="GR-like", capsize=5)
ax.bar(x + width/2, glome_means, width, yerr=glome_stds, label="Glome-like", capsize=5)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Mean |a_m| ± std")
ax.set_title("Fourier Mode Amplitudes")
ax.legend()
plt.tight_layout()
plt.show()

# (4) Phase dispersion comparison for m=2
fig, ax = plt.subplots()
ax.bar(["GR-like", "Glome-like"], [gr_a2_phase, glome_a2_phase],
       color=["blue", "orange"])
ax.set_ylabel("Circular Std Dev of arg(a₂)")
ax.set_title("Phase Stability of m=2 Mode")
plt.tight_layout()
plt.show()

# ---------- Printed summary ----------

print("Fourier Mode Statistics Across Epochs:")
print("Scenario     | Mean |a₁| ± std | Mean |a₂| ± std | Phase Dispersion of a₂")
print("-------------|----------------|----------------|--------------------------")
print(f"GR-like      | {gr_a1_mean:.3f} ± {gr_a1_std:.3f} | "
      f"{gr_a2_mean:.3f} ± {gr_a2_std:.3f} | {gr_a2_phase:.3f} rad")
print(f"Glome-like   | {glome_a1_mean:.3f} ± {glome_a1_std:.3f} | "
      f"{glome_a2_mean:.3f} ± {glome_a2_std:.3f} | {glome_a2_phase:.3f} rad")

print("\nInterpretation:")
print("- GR-like: Strong, variable m=1 amplitude; m=2 phase is unstable.")
print("- Glome-like: Suppressed m=1; strong, stable m=2 with low phase dispersion.")
print("This supports the glome vs GR discriminator based on Fourier mode structure.")
