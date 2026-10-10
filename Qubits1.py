import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, sigmax, sigmay, sigmaz, mesolve, Options, Qobj

# ------------------------------------------------------------------
# 1. Define the qubit and simulation parameters.
# ------------------------------------------------------------------
sx = sigmax()
sy = sigmay()
sz = sigmaz()
psi0 = basis(2, 0)  # initial state, |0>

# Parameters for a superconducting qubit driven by a synthetic gauge field drive.
Delta = 1.0    # detuning
Omega = 1.0    # drive amplitude
gamma = 0.5    # rate that defines the rotation of the drive (synthetic gauge potential)
T = 10.0       # total evolution time
steps = 1000   # number of time steps
times = np.linspace(0, T, steps)

# ------------------------------------------------------------------
# 2. Define a time-dependent Hamiltonian H(t)
# ------------------------------------------------------------------
def H_t(t, args):
    # H(t) = (Δ/2) σ_z + (Ω/2)[cos(gamma*t) σ_x + sin(gamma*t) σ_y]
    return 0.5 * Delta * sz + 0.5 * Omega * (np.cos(gamma * t) * sx + np.sin(gamma * t) * sy)

# Use mesolve options to force storage of all states.
opts = Options(store_states=True)
result = mesolve(H_t, psi0, times, c_ops=[], e_ops=[], options=opts)

# ------------------------------------------------------------------
# 3. Extract Bloch Sphere Coordinates and Compute Berry Phase.
# ------------------------------------------------------------------
theta_arr = np.zeros(steps)
phi_arr = np.zeros(steps)

# Loop over each stored state.
for idx, psi in enumerate(result.states):
    # If psi is not a proper Qobj (e.g. a plain complex number),
    # then replace it with the previous (good) state.
    if not hasattr(psi, "dag") or not isinstance(psi, Qobj):
        print(f"Warning: State at index {idx} is not a Qobj; using previous good state.")
        psi = result.states[idx-1] if idx > 0 else psi0
    # Compute expectation values for σx, σy, and σz.
    try:
        ex = (psi.dag() * sx * psi).full()[0, 0].real
        ey = (psi.dag() * sy * psi).full()[0, 0].real
        ez = (psi.dag() * sz * psi).full()[0, 0].real
    except Exception as e:
        print(f"Error computing expectation at index {idx}: {e}")
        continue

    # Compute the Bloch sphere angles.
    # Polar angle: θ = arccos(⟨σz⟩)
    theta_arr[idx] = np.arccos(ez)
    # Azimuthal angle: φ = atan2(⟨σy⟩, ⟨σx⟩)
    phi_arr[idx] = np.arctan2(ey, ex)

# Unwrap the φ array so that jumps near 2π are removed.
phi_unwrapped = np.unwrap(phi_arr)

# The Berry phase is given by γ = ∫ [½(1 – cosθ)] dφ.
# In discrete form, compute:
dphi = np.diff(phi_unwrapped)                  # Δφ between successive points
theta_mid = 0.5 * (theta_arr[:-1] + theta_arr[1:])  # use midpoints for θ
berry_integrand = 0.5 * (1 - np.cos(theta_mid)) * dphi
berry_phase = np.sum(berry_integrand)

print(f"Numerically integrated Berry phase: {berry_phase:.4f} radians")

# ------------------------------------------------------------------
# 4. Plot the Bloch Sphere Coordinates and Accumulated Berry Phase.
# ------------------------------------------------------------------
plt.figure(figsize=(12, 5))

# (A) Plot θ(t) and φ(t)
plt.subplot(1, 2, 1)
plt.plot(times, theta_arr, label=r'$\theta(t)$')
plt.plot(times, phi_unwrapped, label=r'$\phi(t)$')
plt.xlabel('Time')
plt.ylabel('Angle (radians)')
plt.title('Bloch Sphere Coordinates vs. Time')
plt.legend()

# (B) Plot the cumulative Berry phase as a function of time.
plt.subplot(1, 2, 2)
cumulative_phase = np.cumsum(np.r_[0, berry_integrand])
plt.plot(times, cumulative_phase, label="Cumulative Berry phase")
plt.xlabel('Time')
plt.ylabel('Berry Phase (radians)')
plt.title('Accumulated Berry Phase vs. Time')
plt.legend()

plt.tight_layout()
plt.show()
