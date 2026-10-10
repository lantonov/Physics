import numpy as np
import matplotlib.pyplot as plt

# Parameters
eps_val = 0.1
m_val = 1

# Grid in Hopf coordinates
theta_vals = np.linspace(0, np.pi, 200)
psi_vals = np.linspace(0, 2*np.pi, 200)
Theta, Psi = np.meshgrid(theta_vals, psi_vals)

# Time-averaged LL pseudotensor
tau_tt_avg = (eps_val**2 * m_val**2 * np.sin(m_val*Psi)**2 + 2*np.sin(Theta)**2 - 1)/(4*np.pi)

# Spatially averaged over psi for comparison with Hamiltonian
tau_tt_avg_psi = (eps_val**2 * m_val**2 / 2 + 2*np.sin(Theta)**2 - 1)/(4*np.pi)

# Hamiltonian along fiber (p_theta = p_phi = 0, p_psi = eps*m)
H_mode = 0.5 * (eps_val*m_val)**2 * np.ones_like(Theta)  # constant over manifold

# Plot
fig, ax = plt.subplots(1,2, figsize=(14,5))

# LL pseudotensor
c1 = ax[0].contourf(Psi, Theta, tau_tt_avg, levels=50, cmap='viridis')
ax[0].set_title("Time-averaged LL pseudotensor τ^{tt}(\psi,θ)")
ax[0].set_xlabel("ψ")
ax[0].set_ylabel("θ")
fig.colorbar(c1, ax=ax[0])

# Comparison: psi-averaged LL vs Hamiltonian
ax[1].plot(theta_vals, tau_tt_avg_psi[:,0], label="LL ψ-averaged")
ax[1].plot(theta_vals, H_mode[0,:]/(4*np.pi), '--', label="H_mode / 4π")
ax[1].set_xlabel("θ")
ax[1].set_ylabel("Energy density")
ax[1].set_title("ψ-averaged LL pseudotensor vs Hamiltonian")
ax[1].legend()
ax[1].grid(True)

plt.show()
