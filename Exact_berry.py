import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Define Pauli matrices as 2x2 numpy arrays.
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)

# Initial state (ground state |0>) as a 2-element column vector.
psi0 = np.array([1, 0], dtype=complex)

# Parameters for a superconducting phase qubit.
Delta = 1.0      # detuning
Omega = 1.0      # drive amplitude
gamma = 0.5      # controls the rotation of the drive (synthetic gauge potential)
T = 10.0         # total evolution time
steps = 1000     # number of time steps
times = np.linspace(0, T, steps)
dt = T / steps

def H_t(t):
    """
    Time-dependent Hamiltonian:
    H(t) = (Delta/2)*σ_z + (Omega/2)[cos(gamma*t)*σ_x + sin(gamma*t)*σ_y]
    """
    return 0.5 * Delta * sz + 0.5 * Omega * (np.cos(gamma * t) * sx + np.sin(gamma * t) * sy)

# Allocate an array to store the state at each time step.
psi_t = np.zeros((steps, 2), dtype=complex)
psi_t[0, :] = psi0

# Evolve the state using a simple time-stepping method.
for idx in range(steps - 1):
    t_current = times[idx]
    H_current = H_t(t_current)
    # Compute the time evolution operator for the small interval dt.
    U = expm(-1j * H_current * dt)
    # Update the state.
    psi_t[idx + 1, :] = U @ psi_t[idx, :]

# Prepare arrays to store Bloch sphere angles.
theta_arr = np.zeros(steps)
phi_arr = np.zeros(steps)

# Compute expectation values at each time step.
for idx in range(steps):
    # Reshape state to a column vector.
    psi = psi_t[idx, :].reshape(2, 1)
    # Compute expectation values: ⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩.
    x = (psi.conj().T @ sx @ psi)[0, 0].real
    y = (psi.conj().T @ sy @ psi)[0, 0].real
    z = (psi.conj().T @ sz @ psi)[0, 0].real

    # Convert these coordinates to Bloch sphere angles.
    theta_arr[idx] = np.arccos(z)
    phi_arr[idx] = np.arctan2(y, x)

# Unwrap φ to remove discontinuities.
phi_unwrapped = np.unwrap(phi_arr)

# Numerically integrate the Berry connection:
# The Berry connection is A(t) = ½ (1-cosθ) dφ. Numerically, we use midpoint estimates.
dphi = np.diff(phi_unwrapped)               # differences between φ values
theta_mid = 0.5 * (theta_arr[:-1] + theta_arr[1:])  # midpoint estimates for θ
berry_integrand = 0.5 * (1 - np.cos(theta_mid)) * dphi
berry_phase = np.sum(berry_integrand)

print(f"Numerically integrated Berry phase: {berry_phase:.4f} radians")

# Plot Bloch sphere coordinates and the accumulated Berry phase.
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(times, theta_arr, label=r'$\theta(t)$')
plt.plot(times, phi_unwrapped, label=r'$\phi(t)$')
plt.xlabel('Time')
plt.ylabel('Angle (radians)')
plt.title('Bloch Sphere Coordinates vs. Time')
plt.legend()

plt.subplot(1, 2, 2)
# Compute cumulative Berry phase
cumulative_phase = np.cumsum(np.r_[0, berry_integrand])
plt.plot(times, cumulative_phase, label="Cumulative Berry phase")
plt.xlabel('Time')
plt.ylabel('Berry Phase (radians)')
plt.title('Accumulated Berry Phase vs. Time')
plt.legend()

plt.tight_layout()
plt.show()
