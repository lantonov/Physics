import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
omega = 1.0       # Gravitational wave frequency
epsilon = 0.1     # Fiber coupling strength
omega_f = 0.5     # Fiber phase oscillation frequency
delta = 0.0       # Phase offset

def contact_flow(t, y):
    q_plus, p_plus, q_cross, p_cross, psi, phi = y
    
    # Hamiltonian components
    H_plus = 0.5 * (p_plus ** 2 + omega ** 2 * q_plus ** 2)
    H_cross = 0.5 * (p_cross ** 2 + omega ** 2 * q_cross ** 2)
    H_c = H_plus + H_cross + epsilon * np.cos(omega_f * t + delta) * psi

    # Equations of motion
    dq_plus = p_plus
    dp_plus = -omega ** 2 * q_plus
    dq_cross = p_cross
    dp_cross = -omega ** 2 * q_cross
    dpsi = - (q_plus * dq_plus + q_cross * dq_cross) + H_c
    dphi = epsilon * np.cos(omega_f * t + delta)

    return [dq_plus, dp_plus, dq_cross, dp_cross, dpsi, dphi]

# Initial conditions
y0 = [0.1, 0.0, 0.1, 0.0, 0.0, 0.0]
t_span = (0, 100)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

sol = solve_ivp(contact_flow, t_span, y0, t_eval=t_eval)

# Plot the fiber phase evolution
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[5])
plt.xlabel('Time')
plt.ylabel('Fiber Phase φ(t)')
plt.title('Reeb Flow: Fiber Phase Evolution Over Time')
plt.grid()
plt.show()
