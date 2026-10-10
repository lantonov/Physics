import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Define Pauli matrices.
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

def H(t, Delta, Omega, gamma):
    """
    Time-dependent Hamiltonian for a driven qubit.
    The phase factor gamma*t plays the role of a synthetic gauge field.
    
    H(t) = (Δ/2)σ_z + (Ω/2)[cos(γt) σ_x + sin(γt) σ_y]
    """
    return 0.5 * Delta * sigma_z + 0.5 * Omega * (np.cos(gamma * t) * sigma_x + np.sin(gamma * t) * sigma_y)

# Simulation parameters.
Delta = 1.0      # detuning (arbitrary units)
Omega = 1.0      # drive amplitude
gamma = 0.5      # synthetic gauge field rate (phase rotation)
T = 5.0          # total evolution time
dt = 0.01        # time step for simulation
times = np.arange(0, T + dt, dt)
N = len(times)

# --- Exact state evolution using Trotter product (Exact Holonomy Method) ---

# Initialize state vector (column vector) -- starting in |0>.
psi = np.array([1, 0], dtype=complex)

# Prepare arrays to store Bloch coordinates and accumulated Berry phase.
theta_arr = np.zeros(N)
phi_arr = np.zeros(N)
berry_phase_arr = np.zeros(N)

def state_to_angles(psi):
    """
    Convert a qubit state psi to Bloch sphere angles.
    For a state written as
        psi = [cos(θ/2), e^(i φ) sin(θ/2)],
    the Bloch angles are:
        θ = arccos⟨σ_z⟩,  φ = atan2(⟨σ_y⟩, ⟨σ_x⟩).
    """
    # Expectation values.
    x = np.real(np.conjugate(psi).T @ sigma_x @ psi)
    y = np.real(np.conjugate(psi).T @ sigma_y @ psi)
    z = np.real(np.conjugate(psi).T @ sigma_z @ psi)
    theta = np.arccos(z)   # since z = cos(θ)
    phi = np.arctan2(y, x)
    return theta, phi

# Compute initial Bloch angles.
theta_arr[0], phi_arr[0] = state_to_angles(psi)
berry_phase_arr[0] = 0.0  # initial Berry phase is zero

# Evolve the state in time.
berry_phase = 0.0
for i in range(1, N):
    t = times[i-1]
    H_current = H(t, Delta, Omega, gamma)
    # Compute the evolution operator for this time step.
    U = expm(-1j * H_current * dt)
    psi = U @ psi
    # Normalize the state (numerical errors can accumulate).
    psi = psi / np.linalg.norm(psi)
    
    # Extract Bloch angles from the state.
    theta, phi = state_to_angles(psi)
    theta_arr[i] = theta
    phi_arr[i] = phi

# Unwrap φ so that discontinuities at the 2π boundary are removed.
phi_unwrapped = np.unwrap(phi_arr)

# Numerically integrate the Berry connection.
# The Berry phase is given by:  γ = ∫ [½ (1 - cosθ)] dφ.
for i in range(1, N):
    theta_mid = 0.5 * (theta_arr[i] + theta_arr[i-1])
    dphi = phi_unwrapped[i] - phi_unwrapped[i-1]
    berry_phase += 0.5 * (1 - np.cos(theta_mid)) * dphi
    berry_phase_arr[i] = berry_phase

print("Accumulated Berry phase (exact holonomy) = {:.4f} radians".format(berry_phase))

# --- Plotting the results ---

plt.figure(figsize=(10, 6))

# Plot Bloch sphere coordinates vs. time.
plt.subplot(2, 1, 1)
plt.plot(times, theta_arr, label=r'$\theta(t)$')
plt.plot(times, phi_unwrapped, label=r'$\phi(t)$')
plt.xlabel("Time")
plt.ylabel("Angle (radians)")
plt.title("Bloch Sphere Coordinates vs. Time")
plt.legend()
plt.grid(True)

# Plot the accumulated Berry phase vs. time.
plt.subplot(2, 1, 2)
plt.plot(times, berry_phase_arr, label="Cumulative Berry Phase", color='magenta', linewidth=2)
plt.xlabel("Time")
plt.ylabel("Berry Phase (radians)")
plt.title("Accumulated Berry Phase via Exact Holonomy Method")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
