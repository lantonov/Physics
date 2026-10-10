import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hyp1f1

# ==================================================
# Model Parameters
# ==================================================
# Schwarzschild radius (in meters)
r_s = 3000.0
# Outer boundary (antipodal point) in meters
r_c = 10000.0
# Effective domain length
L_eff = r_c - r_s

# Speed of light (for dimensional restoration, if needed)
c = 3.0e8  # m/s

# Splitting parameter due to Hopf fiber twist
beta = 1.0

# ==================================================
# Quantization condition:
# With splitting, the eigenfrequency (in geometrized units) is:
#   ω_{n,ℓ,m} = (1/(r_c-r_s)) * sqrt[(ℓ+1)(ℓ+2) + 2n(ℓ+1) + n^2 + β m^2]
# ==================================================
def omega(n, ell, m):
    """Compute the eigenfrequency (in m^{-1} units) for given quantum numbers."""
    value = (ell+1)*(ell+2) + 2*n*(ell+1) + n**2 + beta * m**2
    return np.sqrt(value) / L_eff

def omega_phys(n, ell, m):
    """Return physical angular frequency (in rad/s) by multiplying with c."""
    return c * omega(n, ell, m)

def frequency(n, ell, m):
    """Return frequency in Hz."""
    return omega_phys(n, ell, m) / (2*np.pi)

# ==================================================
# Eigenfunction from the M-solution form
# Here we use a simplified ansatz for the radial part:
#   Ψₙ,ℓ(r) = [ (r_c - r)/(r_c - r_s) ]^(ℓ+1) * {}₁F₁(-n; 1; (r - r_s)/(r_c - r_s))
# This function is defined on r in [r_s, r_c]
# ==================================================
def Psi(n, ell, r):
    """Compute the radial eigenfunction for mode (n, ell).    
    (This function does not depend on m since splitting appears in the eigenfrequency.)
    """
    z = (r - r_s) / L_eff   # dimensionless coordinate in [0,1]
    prefactor = (1 - z)**(ell + 1)
    poly = hyp1f1(-n, 1, z)  # terminates for nonnegative integer n
    return prefactor * poly

# ==================================================
# Set up a grid for radial coordinate:
# ==================================================
r_vals = np.linspace(r_s, r_c, 500)

# ==================================================
# Choose mode numbers for testing:
# We'll compare several (n, ell, m) combinations.
# For instance, fix ell=0, and vary n and m.
# ==================================================
ell_val = 0
m_values = [0, 1, -1]  # different fiber quantum numbers (split modes)
modes = [(0, 'n=0'), (1, 'n=1'), (2, 'n=2')]

# ==================================================
# Plot the eigenfunctions for a fixed ell (they do not depend on m)
# ==================================================
plt.figure(figsize=(8, 5))
for n, label in modes:
    psi_vals = Psi(n, ell_val, r_vals)
    plt.plot(r_vals, psi_vals, label=f'{label}, ℓ={ell_val}')
plt.xlabel('r (m)')
plt.ylabel(r'$\Psi_{n,\ell}(r)$')
plt.title('Simulated Radial Eigenfunctions from the M-Solution')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==================================================
# Print the computed eigenfrequencies incorporating splitting
# for each (n, ℓ, m) mode.
# ==================================================
print("Computed eigenfrequencies for the compact BH model with Hopf fiber splitting:")
for n, n_label in modes:
    for m in m_values:
        om_phys = omega_phys(n, ell_val, m)
        f = frequency(n, ell_val, m)
        print(f"Mode (n={n}, ℓ={ell_val}, m={m}): ω_phys = {om_phys:.2e} rad/s, frequency = {f:.2f} Hz")
