import numpy as np
import matplotlib.pyplot as plt

# Parameters (feel free to tweak)
sigma = 1.0
T = 1.0
# Use either time-constant eigenvalues:
lambda_H = 1.0     # horizontal Ricci eigenvalue
lambda_V = 3.0     # vertical Ricci eigenvalue
# Or provide time series lambda_H(t), lambda_V(t) and integrate numerically.

phi = np.linspace(0, np.pi/2, 400)

# Frozen-Ricci (constant in time) formulas
F = sigma*(lambda_H*np.cos(phi)**2 + lambda_V*np.sin(phi)**2)
f_T = np.exp(T*F)

Delta_psi = -2*sigma*T*(lambda_V - lambda_H)*np.sin(phi)*np.cos(phi)

# Plots
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

axs[0].plot(phi, Delta_psi, color='crimson', lw=2)
axs[0].axhline(0, color='#888', lw=1)
axs[0].set_title("Shear displacement Δψ(T, φ)")
axs[0].set_xlabel("φ (tilt from horizontal)")
axs[0].set_ylabel("Δψ")

axs[1].plot(phi, f_T, color='navy', lw=2)
axs[1].set_title("Time-change factor f_T(φ)")
axs[1].set_xlabel("φ (tilt from horizontal)")
axs[1].set_ylabel("f_T")

plt.tight_layout(); plt.show()
