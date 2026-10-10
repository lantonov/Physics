import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# Parameters
m = 1.0          # mass scale (set units so R=1)
k = 1            # angular momentum parameter (l=0)
kappa = 0.2      # torsion coupling strength

# Gamma constants for torsion term (adjusted per gamma conventions)
C1 = 1.0
C2 = 0.0

# Domain
chi = np.linspace(1e-6, np.pi - 1e-6, 200)

# Initial guess energy
E = 1.0

def torsion_sigma(f, g):
    return kappa * (C1 * f * g + C2 * (f**2 - g**2))

def ode_system(chi, y, E, Sigma):
    f, g = y
    sf = - (k / np.sin(chi)) * f + (m + E + Sigma) * g
    sg = (k / np.sin(chi)) * g + (m - E + Sigma) * f
    return np.vstack((sf, sg))

def bc(ya, yb):
    # Regularity boundary conditions
    return np.array([ya[0], yb[1]])

# Iterative solver
def solve_dirac_nonlinear(E_init, tol=1e-5, max_iter=20):
    E = E_init
    Sigma = np.zeros_like(chi)
    
    for _ in range(max_iter):
        def fun(chi, y):
            f, g = y
            sigma = torsion_sigma(f, g)
            return ode_system(chi, y, E, sigma)
        
        y_init = np.vstack((np.sin(chi), np.cos(chi)))
        sol = solve_bvp(fun, bc, chi, y_init)
        
        f_sol = sol.sol(chi)[0]
        g_sol = sol.sol(chi)[1]
        
        Sigma_new = torsion_sigma(f_sol, g_sol)
        
        # Energy correction estimate (simple)
        delta_m = np.trapz(Sigma_new * (f_sol**2 + g_sol**2) * np.sin(chi)**2, chi)
        E_new = E_init + delta_m
        
        # Check convergence
        if np.abs(E_new - E) < tol:
            break
        E = E_new
        Sigma = Sigma_new
    
    return E, f_sol, g_sol, Sigma_new, sol.status

# Run solver
E_final, f_final, g_final, Sigma_final, status = solve_dirac_nonlinear(E)

# Output results
print(f"Converged energy E = {E_final}")
print(f"Solve status = {status}")

# Plot results
plt.figure(figsize=(10,6))
plt.plot(chi, f_final, label='f(chi)')
plt.plot(chi, g_final, label='g(chi)')
plt.plot(chi, Sigma_final, label='Sigma(chi) torsion')
plt.xlabel('chi')
plt.ylabel('Amplitude')
plt.legend()
plt.title('Nonlinear Dirac Spinor Components and Torsion on S^3')
plt.show()
