import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hyp1f1

# -------------------------------
# Set model parameters
# -------------------------------
# Schwarzschild radius (in meters)
r_s = 3000.0   
# Maximum radius for the compact spatial section
r_c = 10000.0  
# Effective length of the domain
L_eff = r_c - r_s  

# Speed of light for conversion (if needed)
c = 3.0e8  # m/s

# -------------------------------
# Define the eigenfunction Psi_{n,l}(r)
# -------------------------------
def Psi(n, ell, r):
    """
    Compute the radial eigenfunction Psi_{n,ell}(r) for a given mode (n,ell)
    using the M-solution form:
    
       Psi(r) = [ (r_c - r) / (r_c - r_s) ]^(ell+1) * hyp1f1(-n, 1, (r - r_s)/(r_c - r_s) )
    
    Parameters:
        n : int
            Radial quantum number (n = 0,1,2,...)
        ell : int
            Angular momentum quantum number.
        r : ndarray
            Array of radial coordinates (in meters), with r in [r_s, r_c].
    
    Returns:
        Psi : ndarray
            The eigenfunction evaluated at the given r.
    """
    # Map r to the dimensionless coordinate: z in [0, 1]
    z = (r - r_s) / L_eff
    # Pre-factor to absorb known endpoint behavior
    prefactor = (1 - z)**(ell + 1)
    # Evaluate the confluent hypergeometric function; when n is non-negative integer,
    # hyp1f1(-n, 1, z) is a polynomial.
    poly = hyp1f1(-n, 1, z)
    return prefactor * poly

# -------------------------------
# Define the eigenfrequency formula:
#   omega_{n,ell} = 1/(r_c - r_s) * sqrt[(ell+1)(ell+2) + 2n(ell+1) + n^2]
# -------------------------------
def omega(n, ell):
    term = (ell + 1) * (ell + 2) + 2 * n * (ell + 1) + n**2
    return np.sqrt(term) / L_eff  # in inverse meters (geometrized units)

# To obtain angular frequency in physical units (rad/s): multiply by c.
def omega_phys(n, ell):
    return c * omega(n, ell)

# And the ordinary frequency in Hz (divide by 2*pi):
def frequency(n, ell):
    return omega_phys(n, ell) / (2 * np.pi)

# -------------------------------
# Set-up a grid in r for evaluation of eigenfunctions
# -------------------------------
r_vals = np.linspace(r_s, r_c, 500)

# -------------------------------
# Choose mode numbers for testing
# -------------------------------
# We choose a couple of radial excitations with a fixed angular index, e.g. ell = 0.
ell_val = 0
modes = [(0, 'n=0'), (1, 'n=1'), (2, 'n=2')]

# -------------------------------
# Plot the eigenfunctions Psi(r)
# -------------------------------
plt.figure(figsize=(8, 5))
for n, label in modes:
    psi_vals = Psi(n, ell_val, r_vals)
    plt.plot(r_vals, psi_vals, label=f'{label}, ell={ell_val}')
plt.xlabel('r (m)')
plt.ylabel(r'$\Psi_{n,\ell}(r)$')
plt.title('Simulated Normal Mode Eigenfunctions from M-Solution')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------
# Print the computed eigenfrequencies for these modes
# -------------------------------
print("Computed eigenfrequencies for a compact BH model:")
for n, label in modes:
    om = omega_phys(n, ell_val)
    f = frequency(n, ell_val)
    print(f"Mode (n={n}, ell={ell_val}): omega_phys = {om:.2e} rad/s,   frequency = {f:.2f} Hz")
