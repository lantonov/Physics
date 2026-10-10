import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, norm

# Define Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

def H(t, Delta, Omega, gamma):
    """
    Time-dependent Hamiltonian for a driven qubit.
    The phase factor gamma*t plays the role of a synthetic gauge potential.
    """
    return 0.5 * Delta * sigma_z + 0.5 * Omega * (np.cos(gamma * t) * sigma_x + np.sin(gamma * t) * sigma_y)

# Simulation parameters
Delta = 1.0      # detuning (in arbitrary units)
Omega = 1.0      # drive amplitude (in the same units)
gamma = 0.5      # synthetic gauge field rate (controls phase rotation)
T = 5.0          # total evolution time
dt = 0.01        # time step for simulation
times = np.arange(0, T + dt, dt)
N = len(times)

# Allocate arrays to store phase evolution (phase of the (0,0) element)
phase_exact = np.zeros(N)
phase_mag1  = np.zeros(N)
phase_mag2  = np.zeros(N)

# Initialize evolution operators (exact and cumulative Magnus terms)
U_exact = np.eye(2, dtype=complex)
Omega1_cum = np.zeros((2, 2), dtype=complex)  # cumulative first order Magnus
# For Omega2, we will compute on-the-fly at each time using a double sum over past steps
# This is acceptable for the moderate number of time steps used here.
# Initialize storage for U_mag1 and U_mag2 at each time.
U_mag1 = np.eye(2, dtype=complex)
U_mag2 = np.eye(2, dtype=complex)

# Also store the Hamiltonians to avoid recomputation in the double loop
H_list = []

for i, t in enumerate(times):
    # Compute H(t) at the current time
    H_current = H(t, Delta, Omega, gamma)
    H_list.append(H_current)
    
    # --- Exact evolution via Trotter product ---
    U_exact = expm(-1j * H_current * dt) @ U_exact

    # --- Update cumulative first-order Magnus term ---
    Omega1_cum += -1j * H_current * dt

    # --- Compute second-order Magnus term up to current time ---
    # Omega2_cum = -1/2 * \int_0^t dt1 \int_0^{t1} dt2 [H(t1), H(t2)]
    Omega2_cum = np.zeros((2, 2), dtype=complex)
    # Loop over previous times (t_j < current t)
    for j in range(i):
        H_j = H_list[j]
        Omega2_cum += -0.5 * (H_current @ H_j - H_j @ H_current) * (dt**2)
    
    # Compute evolution operators from Magnus expansion
    U_mag1 = expm(Omega1_cum)
    U_mag2 = expm(Omega1_cum + Omega2_cum)
    
    # Store the phase of the (0,0) element for each operator as a measure (the overall phase is gauge-invariant)
    phase_exact[i] = np.angle(U_exact[0, 0])
    phase_mag1[i]  = np.angle(U_mag1[0, 0])
    phase_mag2[i]  = np.angle(U_mag2[0, 0])

# Compute final error norms (Frobenius norm) between evolution operators:
err_mag1 = norm(U_exact - U_mag1)
err_mag2 = norm(U_exact - U_mag2)
print(f"Final error using 1st order Magnus: {err_mag1:.2e}")
print(f"Final error using 2nd order Magnus: {err_mag2:.2e}")

# --- Plotting the phase accumulation vs. time ---
plt.figure(figsize=(10, 6))
plt.plot(times, phase_exact, label="Exact evolution", linewidth=2)
plt.plot(times, phase_mag1, '--', label="Magnus (1st order)")
plt.plot(times, phase_mag2, ':', label="Magnus (1st + 2nd order)")
plt.xlabel("Time")
plt.ylabel("Phase (radians) [arg(U[0,0])]")
plt.title("Accumulated Phase in Superconducting Qubit Evolution\nwith Synthetic Gauge Field Drive")
plt.legend()
plt.grid(True)
plt.show()
