import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# -------------------------------------------------------
# Parameters for our closed radiation-dominated model:
# -------------------------------------------------------
# Assume a closed universe with radius R; for the chosen mode, k = sqrt(l(l+2))/R.
# For simplicity, we set k = 1.0 (this can be adjusted or computed from l and R).
k = 1.0  
# For a radiation-dominated era, a(τ) ∝ τ so that H = 1/τ.
# We'll integrate in conformal time tau.
tau_start = 0.1  # starting conformal time
tau_end   = 100.0
t_span = (tau_start, tau_end)

# --------------------------
# ODE system definition:
# --------------------------
# y[0] = Phi        (metric potential)
# y[1] = Phi'       (time derivative of metric potential)
# y[2] = F0         (neutrino monopole)
# y[3] = F1         (neutrino dipole)
# y[4] = F2         (neutrino quadrupole, proportional to anisotropic stress)

def odes(tau, y, A):
    # Damping factor: in radiation era, H = a'/a = 1/tau.
    H = 1.0 / tau
    # Equation for metric potential:
    # y0' = y1
    dy0 = y[1]
    # y1' = -2H * y1 - (k^2)/3 * y0 + A * y4
    dy1 = -2 * H * y[1] - (k**2)/3.0 * y[0] + A * y[4]
    
    # Neutrino monopole equation: F0' = - (k/3) F1 - Phi'
    dy2 = - (k/3.0) * y[3] - y[1]
    
    # Neutrino dipole: F1' = (k/3)(F0 - 2F2) + Phi'
    # (Here we approximate Psi' ~ Phi')
    dy3 = (k/3.0) * (y[2] - 2.0 * y[4]) + y[1]
    
    # Neutrino quadrupole: F2' = (2k/5)*F1
    dy4 = (2*k/5.0) * y[3]
    
    return [dy0, dy1, dy2, dy3, dy4]

# --------------------------
# Initial conditions:
# --------------------------
# Assume adiabatic initial conditions:
# Often one sets Phi = constant (say 1e-5),
# Phi' = 0, and for neutrinos one might set F0 ~ -Phi, F1 = 0, F2 = 0.
Phi0 = 1e-5
dPhi0 = 0.0
F0_0 = -1e-5
F1_0 = 0.0
F2_0 = 0.0

y0_initial = [Phi0, dPhi0, F0_0, F1_0, F2_0]

# --------------------------
# Integration and comparison:
# --------------------------
# We'll perform two runs:
#   (a) Without anisotropic stress coupling: A = 0.
#   (b) With anisotropic stress coupling: A nonzero (e.g., 0.5).

sol_no_aniso = solve_ivp(lambda tau, y: odes(tau, y, A=0.0), t_span, y0_initial, method='RK45', dense_output=True)
sol_with_aniso = solve_ivp(lambda tau, y: odes(tau, y, A=0.5), t_span, y0_initial, method='RK45', dense_output=True)

# Evaluate solutions on a common grid for plotting
taus = np.logspace(np.log10(tau_start), np.log10(tau_end), 400)
Phi_no_aniso = sol_no_aniso.sol(taus)[0]
Phi_with_aniso = sol_with_aniso.sol(taus)[0]

# --------------------------
# Plotting:
# --------------------------
plt.figure(figsize=(10, 6))
plt.loglog(taus, np.abs(Phi_no_aniso), label=r'$\Phi$ without neutrino anisotropic stress ($A=0$)')
plt.loglog(taus, np.abs(Phi_with_aniso), label=r'$\Phi$ with neutrino anisotropic stress ($A=0.5$)')
plt.xlabel(r'Conformal time $\tau$')
plt.ylabel(r'$|\Phi|$')
plt.title("Evolution of a Single Mode in a Closed, Radiation-Dominated Universe")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()
