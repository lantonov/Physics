import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hyp1f1, sph_harm

# ======================================
# Model Parameters for a 10-solar-mass BH
# ======================================
r_s = 30000.0         # Schwarzschild radius (m)
r_c = 142500.0        # Outer boundary (m) chosen so that L_eff = r_c - r_s = 112500 m
L_eff = r_c - r_s     # Effective domain length (m)
c = 3e8             # Speed of light in m/s

# Splitting constant from Hopf fiber twist
beta = 1.0            # Example value

# ======================================
# Quantization Condition Function:
# ω_{n,ℓ,m} = (1 / (r_c - r_s)) * sqrt[(ℓ+1)(ℓ+2) + 2n(ℓ+1) + n² + β m²]
# =======================================
def omega_calc(n, ell, m):
    value = (ell+1)*(ell+2) + 2*n*(ell+1) + n**2 + beta*(m**2)
    return np.sqrt(value) / L_eff  # frequency in m⁻¹ (geometric units)

def omega_phys(n, ell, m):
    return c * omega_calc(n, ell, m)  # angular frequency in rad/s

def frequency(n, ell, m):
    return omega_phys(n, ell, m) / (2 * np.pi)
# ======================================
# Radial Eigenfunction from the M-solution:
# R_{n,ℓ}(r) = [ (r_c - r) / (r_c - r_s) ]^(ℓ+1) * {}₁F₁(-n; 1; (r - r_s)/(r_c - r_s) )
# ======================================
def R_radial(n, ell, r):
    z = (r - r_s) / L_eff   # dimensionless variable in [0, 1]
    prefactor = (1 - z)**(ell + 1)
    poly = hyp1f1(-n, 1, z)  # for integer n this terminates
    return prefactor * poly

# ======================================
# Angular Eigenfunction (approximate):
# We use the standard spherical harmonic Y_ℓ^m(θ, φ) as an approximation.
# ======================================
def Y_ang(ell, m, theta, phi):
    return sph_harm(m, ell, phi, theta)

# ======================================
# Set up grids for numerical evaluation:
# Radial grid:
r_vals = np.linspace(r_s, r_c, 500)
# Angular grid for visualization:
theta_vals = np.linspace(0, np.pi, 100)
phi_vals = np.linspace(0, 2*np.pi, 100)
THETA, PHI = np.meshgrid(theta_vals, phi_vals)

# ======================================
# Choose mode numbers for testing:
# For ℓ = 0, only m = 0 is available.
# For ℓ = 1, test m = -1, 0, 1.
# Also, take radial modes n = 0 and n = 1.
# ======================================
radial_modes = [(0, 'n = 0'), (1, 'n = 1')]
ell_modes = [0, 1]
m_values_for_ell = {0: [0], 1: [-1, 0, 1]}
print(radial_modes)
# ======================================
# Plot Radial Eigenfunctions for ℓ = 0:
# ======================================
plt.figure(figsize=(8, 5))
for n, label in radial_modes:
    R_vals = R_radial(n, ell=0, r=r_vals)
    plt.plot(r_vals, R_vals, label=f'{label}, ℓ = 0')
plt.xlabel('r (m)')
plt.ylabel(r'$R_{n,\ell}(r)$')
plt.title('Radial Eigenfunctions for ℓ = 0 (10 M₍⊙₎ BH, L_eff = 112500 m)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ======================================
# Plot Angular Eigenfunctions for ℓ = 1
# We'll plot |Y_ℓ^m(θ, φ)| for m = -1, 0, 1.
# ======================================
fig, axs = plt.subplots(1, 3, subplot_kw={'projection':'polar'}, figsize=(14, 4))
for i, m in enumerate(m_values_for_ell[1]):
    Y_vals = np.abs(Y_ang(ell=1, m=m, theta=THETA, phi=PHI))
    c = axs[i].contourf(PHI, THETA, Y_vals, 50, cmap='viridis')
    axs[i].set_title(r'$\ell = 1, \; m = %d$' % m)
    fig.colorbar(c, ax=axs[i])
plt.suptitle('Angular Eigenfunctions |Y$_\ell^m(\theta, \phi)$| for ℓ = 1 (approximation)')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ======================================
# Print Computed Eigenfrequencies Including Splitting:
# ======================================
print("Computed eigenfrequencies for a 10-solar-mass BH with r_s = 30000 m and r_c = 142500 m (L_eff = 112500 m):")
for ell in ell_modes:
    for n, _ in radial_modes:
        for m in m_values_for_ell[ell]:
            om_phys = omega_phys(n, ell, m)
            freq = frequency(n, ell, m)
            print(f"Mode (n={n}, ℓ={ell}, m={m}): ω_phys = {om_phys:.2e} rad/s, Frequency = {freq:.2f} Hz")
