# Contact Hamiltonian Flow Simulation for Gravitational Waves

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define contact Hamiltonian parameters
omega = 1.0        # Base gravitational wave frequency
epsilon = 0.01    # Fiber phase coupling strength
omega_f = 1.5     # Fiber phase frequency
delta = 0.0       # Initial phase offset

# Define contact Hamiltonian flow equations
def contact_flow(t, y):
    q_plus, p_plus, q_cross, p_cross, psi = y

    # Base Hamiltonians (standard oscillators)
    H_plus = 0.5 * (p_plus ** 2 + omega ** 2 * q_plus ** 2)
    H_cross = 0.5 * (p_cross ** 2 + omega ** 2 * q_cross ** 2)
    H_c = H_plus + H_cross + epsilon * np.cos(omega_f * t + delta) * psi

    # Contact flow equations
    dq_plus = p_plus + (q_plus * H_c)
    dp_plus = -omega ** 2 * q_plus + (p_plus * H_c)

    dq_cross = p_cross + (q_cross * H_c)
    dp_cross = -omega ** 2 * q_cross + (p_cross * H_c)

    dpsi = H_c

    return [dq_plus, dp_plus, dq_cross, dp_cross, dpsi]

# Initial conditions
y0 = [0.1, 0.0, 0.1, 0.0, 0.0]

# Time span
t_span = (0, 100)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# Solve the system
sol = solve_ivp(contact_flow, t_span, y0, t_eval=t_eval, method='RK45')

# Extract solutions
q_plus = sol.y[0]
p_plus = sol.y[1]
q_cross = sol.y[2]
p_cross = sol.y[3]
psi = sol.y[4]

# Plot q_plus and q_cross (base modes)
plt.figure(figsize=(12, 6))
plt.plot(sol.t, q_plus, label='q_plus (Base-Base)')
plt.plot(sol.t, q_cross, label='q_cross (Base-Fiber)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Contact Hamiltonian Flow: Base Oscillations')
plt.legend()
plt.grid()
plt.show()

# Plot fiber phase evolution
plt.figure(figsize=(12, 6))
plt.plot(sol.t, psi, label='Fiber Phase (psi)', color='purple')
plt.xlabel('Time')
plt.ylabel('Fiber Phase (psi)')
plt.title('Contact Hamiltonian Flow: Fiber Phase Evolution')
plt.legend()
plt.grid()
plt.show()

# Plot phase-space trajectory for q_plus
plt.figure(figsize=(8, 6))
plt.plot(q_plus, p_plus, label='q_plus vs p_plus')
plt.xlabel('q_plus')
plt.ylabel('p_plus')
plt.title('Contact Hamiltonian Phase Space: q_plus')
plt.legend()
plt.grid()
plt.show()

# Plot phase-space trajectory for q_cross
plt.figure(figsize=(8, 6))
plt.plot(q_cross, p_cross, label='q_cross vs p_cross', color='green')
plt.xlabel('q_cross')
plt.ylabel('p_cross')
plt.title('Contact Hamiltonian Phase Space: q_cross')
plt.legend()
plt.grid()
plt.show()
