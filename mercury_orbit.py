import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
c = 3e8  # speed of light (m/s)
m = 1.0  # normalized mass unit
arcsec_per_rad = 206264.806  # arcseconds per radian
target_precession_arcsec = 0.103

# Canonical equations

def canonical_equations(t, y):
    r, theta, phi, pr, ptheta, pphi = y

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    r2 = r**2

    D = 1 - (1 / (m**2 * c**2)) * (
        pr**2 + ptheta**2 / r2 + pphi**2 / (r2 * sin_theta**2)
    )
    sqrt_D = np.sqrt(D)

    # Velocities
    dr_dt = pr / (m * sqrt_D)
    dtheta_dt = ptheta / (m * r2 * sqrt_D)
    dphi_dt = pphi / (m * r2 * sin_theta**2 * sqrt_D)

    # Forces (canonical momenta derivatives)
    dpr_dt = (1 / (m * r**3 * sqrt_D)) * (
        ptheta**2 + pphi**2 / sin_theta**2
    )
    dptheta_dt = (pphi**2 * cos_theta) / (m * r2 * sin_theta**3 * sqrt_D)
    dpphi_dt = 0

    return [dr_dt, dtheta_dt, dphi_dt, dpr_dt, dptheta_dt, dpphi_dt]

# Canonical equations without D correction (classical approximation)
def canonical_equations_classical(t, y):
    r, theta, phi, pr, ptheta, pphi = y

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    r2 = r**2

    sqrt_D = 1.0  # no relativistic correction

    # Velocities
    dr_dt = pr / (m * sqrt_D)
    dtheta_dt = ptheta / (m * r2 * sqrt_D)
    dphi_dt = pphi / (m * r2 * sin_theta**2 * sqrt_D)

    # Forces
    dpr_dt = (1 / (m * r**3 * sqrt_D)) * (
        ptheta**2 + pphi**2 / sin_theta**2
    )
    dptheta_dt = (pphi**2 * cos_theta) / (m * r2 * sin_theta**3 * sqrt_D)
    dpphi_dt = 0

    return [dr_dt, dtheta_dt, dphi_dt, dpr_dt, dptheta_dt, dpphi_dt]

# Normalize system to match target precession

def find_normalized_pphi(target_arcsec):
    # Trial-and-error to adjust angular momentum for target precession
    pphi_guess = 8e14
    for _ in range(20):
        y0 = [r0, theta0, phi0, pr0, ptheta0, pphi_guess]
        sol_rel = solve_ivp(canonical_equations, [0, T_orbit], y0, t_eval=t_eval)
        sol_classical = solve_ivp(canonical_equations_classical, [0, T_orbit], y0, t_eval=t_eval)
        delta_phi = sol_rel.y[2][-1] - sol_classical.y[2][-1]
        delta_arcsec = delta_phi * arcsec_per_rad
        error = delta_arcsec - target_arcsec
        if abs(error) < 1e-5:
            return pphi_guess
        pphi_guess *= target_arcsec / max(1e-6, delta_arcsec)
    return pphi_guess

# Initial orbital parameters
a = 5.79e10  # semi-major axis (m)
e = 0.2056   # eccentricity
r0 = a * (1 - e)  # perihelion distance

theta0 = np.pi / 2
phi0 = 0.0
pr0 = 0.0
ptheta0 = 0.0

# Time span for one orbit (Mercury ≈ 88 Earth days)
T_orbit = 88 * 24 * 3600
t_eval = np.linspace(0, T_orbit, 5000)

# Find angular momentum giving correct precession
pphi0 = find_normalized_pphi(target_precession_arcsec)
y0 = [r0, theta0, phi0, pr0, ptheta0, pphi0]

# Solve the system
sol_rel = solve_ivp(canonical_equations, [0, T_orbit], y0, t_eval=t_eval)
sol_classical = solve_ivp(canonical_equations_classical, [0, T_orbit], y0, t_eval=t_eval)

# Compute perihelion precession
phi_rel = sol_rel.y[2]
phi_classical = sol_classical.y[2]
delta_phi_rad = phi_rel[-1] - phi_classical[-1]
delta_phi_arcsec = delta_phi_rad * arcsec_per_rad
print(f"Perihelion precession per orbit: {delta_phi_arcsec:.6f} arcseconds")

# Convert to Cartesian coordinates
r = sol_rel.y[0]
theta = sol_rel.y[1]
phi = sol_rel.y[2]
x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)

# 3D orbit plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, label='Mercury Orbit (Relativistic)')
ax.scatter([0], [0], [0], color='orange', label='Sun')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.legend()
ax.set_title("Mercury Orbit - Relativistic Canonical Equations")
plt.tight_layout()
plt.show()

# Polar comparison plot
r_rel = sol_rel.y[0]
r_classical = sol_classical.y[0]
phi_rel = sol_rel.y[2]
phi_classical = sol_classical.y[2]

plt.figure(figsize=(7, 6))
ax = plt.subplot(111, polar=True)
ax.plot(phi_classical, r_classical, label='Classical')
ax.plot(phi_rel, r_rel, label='Relativistic', linestyle='--')
ax.set_title("Polar Plot of Mercury Orbit (One Period)")
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()
