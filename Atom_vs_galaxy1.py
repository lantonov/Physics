import numpy as np
import matplotlib.pyplot as plt
from clifford import Cl

# ------------------------------------------------------
# 1. Build Cl(4,0) for the internal rotor, then rename its basis
# ------------------------------------------------------
layout40, blades40 = Cl(4, 0, firstIdx=1)
ie1 = blades40['e1']
ie2 = blades40['e2']
ie3 = blades40['e3']
ie4 = blades40['e4']

# ------------------------------------------------------
# 2. Build Cl(1,3) for the external observer, then rename its basis
# ------------------------------------------------------
layout13, blades13 = Cl(1, 3, firstIdx=0)
E0 = blades13['e0']
E1 = blades13['e1']
E2 = blades13['e2']
E3 = blades13['e3']

# ------------------------------------------------------
# 3. Define internal rotor generators (Cl(4,0))
# ------------------------------------------------------
B_fiber = ie3 * ie4   # (ie3 ∧ ie4) plane
B_base  = ie1 * ie2   # (ie1 ∧ ie2) plane

def rotor_state(psi, theta):
    """
    Build a unit rotor in Cl(4,0): R = exp(-psi/2 * B_fiber) * exp(-theta/2 * B_base).
    We then normalize via .normal().
    """
    Rf = np.exp(-0.5 * psi * B_fiber)
    Rb = np.exp(-0.5 * theta * B_base)
    return (Rf * Rb).normal()

def map_40_to_13(mv40):
    """
    Map a Cl(4,0) multivector (using ie1..ie4) into Cl(1,3) (using E0..E3):
      ie1 -> E1, ie2 -> E2, ie3 -> E3, ie4 -> E0.
    """
    coeffs    = mv40.value
    bladeList = layout40.bladeTupList
    mv13 = 0
    for coeff, blade in zip(coeffs, bladeList):
        if abs(coeff) < 1e-12:
            continue
        if len(blade) == 0:
            mapped = 1.0
        else:
            mapped = 1.0
            for idx in blade:
                if idx == 1:
                    mapped *= E1
                elif idx == 2:
                    mapped *= E2
                elif idx == 3:
                    mapped *= E3
                elif idx == 4:
                    mapped *= E0
        mv13 += coeff * mapped
    return mv13

# ------------------------------------------------------
# 4. Lorentz boost and gravitational redshift
# ------------------------------------------------------
def lorentz_boost(mv13, eta):
    """
    Boost along E1 in Cl(1,3) with rapidity eta:
      L = exp(-eta/2 * E0 E1)
      return L * mv13 * (~L).
    """
    L = np.exp(-0.5 * eta * (E0 * E1))
    return L * mv13 * ~L

def gravitational_redshift(psi, phi):
    """
    Approximate gravitational redshift of the internal phase:
      psi' = psi * (1 - phi), where phi = GM/(r c^2).
    """
    return psi * (1 - phi)

# ------------------------------------------------------
# 5. Simulate uncertainty in one system (ensemble of trials)
# ------------------------------------------------------
def simulate_uncertainty(N, psi0, theta0, theta_noise_std, eta_boost, phi_redshift):
    """
    Perform N trials. Each trial:
      1) Add a small random fluctuation to the internal angle theta.
      2) Apply gravitational redshift to psi0.
      3) Build the internal rotor in Cl(4,0).
      4) Measure two observables A = R ie1 (~R), B = R ie2 (~R).
      5) Map them to Cl(1,3) (external), extract their E1/E2 components.
      6) Apply a Lorentz boost, extract E1/E2 again.
    Returns a dict with σ_A, σ_B before/after, their product, and raw arrays.
    """
    A_before = []
    B_before = []
    A_after  = []
    B_after  = []
    
    for _ in range(N):
        # (1) Fluctuate theta
        delta_theta = np.random.normal(0, theta_noise_std)
        theta_trial = theta0 + delta_theta
        
        # (2) Redshift psi0
        psi_eff = gravitational_redshift(psi0, phi_redshift)
        
        # (3) Build internal rotor R in Cl(4,0)
        R_int = rotor_state(psi_eff, theta_trial)
        
        # (4) Internal observables in Cl(4,0):
        A_int = R_int * ie1 * ~R_int   # “radial position”
        B_int = R_int * ie2 * ~R_int   # “radial momentum”
        
        # (5) Map to external Cl(1,3) and extract E1/E2
        A_ext = map_40_to_13(A_int)
        B_ext = map_40_to_13(B_int)
        a0 = float(A_ext | E1)
        b0 = float(B_ext | E2)
        A_before.append(a0)
        B_before.append(b0)
        
        # (6) Apply Lorentz boost
        A_bo = lorentz_boost(A_ext, eta_boost)
        B_bo = lorentz_boost(B_ext, eta_boost)
        a1 = float(A_bo | E1)
        b1 = float(B_bo | E2)
        A_after.append(a1)
        B_after.append(b1)
    
    A_before = np.array(A_before)
    B_before = np.array(B_before)
    A_after  = np.array(A_after)
    B_after  = np.array(B_after)
    
    sigma_A_before = np.std(A_before)
    sigma_B_before = np.std(B_before)
    sigma_A_after  = np.std(A_after)
    sigma_B_after  = np.std(B_after)
    
    return {
        'sigma_A_before': sigma_A_before,
        'sigma_B_before': sigma_B_before,
        'uncertainty_before': sigma_A_before * sigma_B_before,
        'sigma_A_after': sigma_A_after,
        'sigma_B_after': sigma_B_after,
        'uncertainty_after': sigma_A_after * sigma_B_after,
        'A_before_vals': A_before,
        'B_before_vals': B_before,
        'A_after_vals': A_after,
        'B_after_vals': B_after
    }

# Example parameters for a single‐system run
N = 2000
psi0 = 0.8
theta0 = np.deg2rad(20)
theta_noise_std = 0.02  # small base‐angle noise
eta_boost = 0.5         # rapidity (v ≈ tanh(0.5)) for the external boost
phi_redshift = 1e-6     # tiny GM/(r c²) factor for redshift

results = simulate_uncertainty(N, psi0, theta0, theta_noise_std, eta_boost, phi_redshift)

print("=== Single System Uncertainties ===")
print(" Before boost & redshift:")
print("  σ_A =", results['sigma_A_before'])
print("  σ_B =", results['sigma_B_before'])
print("  ΔA·ΔB =", results['uncertainty_before'])
print("\n After boost & redshift:")
print("  σ_A =", results['sigma_A_after'])
print("  σ_B =", results['sigma_B_after'])
print("  ΔA·ΔB =", results['uncertainty_after'])

# Plot distributions of A before vs. after
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(results['A_before_vals'], bins=30, color='skyblue', edgecolor='k', alpha=0.7)
plt.xlabel("A (E1) before")
plt.title("Distribution of A before Boost & Redshift")

plt.subplot(1, 2, 2)
plt.hist(results['A_after_vals'], bins=30, color='salmon', edgecolor='k', alpha=0.7)
plt.xlabel("A (E1) after")
plt.title("Distribution of A after Boost & Redshift")
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 5))
plt.scatter(results['A_before_vals'], results['B_before_vals'],
            s=5, alpha=0.6, label="Before (blue)", color='blue')
plt.scatter(results['A_after_vals'], results['B_after_vals'],
            s=5, alpha=0.6, label="After (red)", color='red')
plt.xlabel("A (E1 component)")
plt.ylabel("B (E2 component)")
plt.title("Scatter: A vs. B Before vs. After")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------
# 6. Simulate an “entangled pair” of subsystems
# ------------------------------------------------------
def simulate_entangled_pair(N, psi_mean, psi_corr, theta0, theta_noise_std,
                            eta_boost, phi_redshift):
    """
    Build two subsystems whose internal phases are correlated:
      psi1 = psi_mean + d,  psi2 = psi_mean - d, where d ~ N(0, psi_corr).
    Each subsystem also gets a small θ‐noise.  Then apply the same redshift
    and boost to each.  Return covariances of (A1,A2) and (B1,B2) before/after.
    """
    A1_b, B1_b = [], []
    A2_b, B2_b = [], []
    A1_a, B1_a = [], []
    A2_a, B2_a = [], []
    
    deviations = np.random.normal(0, psi_corr, size=N)
    
    for d in deviations:
        # Correlated internal phases
        psi1 = psi_mean + d
        psi2 = psi_mean - d
        
        # Small θ‐noise per subsystem
        t1 = theta0 + np.random.normal(0, theta_noise_std)
        t2 = theta0 + np.random.normal(0, theta_noise_std)
        
        # Gravitational redshift on each
        psi1_r = gravitational_redshift(psi1, phi_redshift)
        psi2_r = gravitational_redshift(psi2, phi_redshift)
        
        # Build rotors
        R1 = rotor_state(psi1_r, t1)
        R2 = rotor_state(psi2_r, t2)
        
        # Internal observables
        A1_int = R1 * ie1 * ~R1
        B1_int = R1 * ie2 * ~R1
        A2_int = R2 * ie1 * ~R2
        B2_int = R2 * ie2 * ~R2
        
        # Map to external
        A1_ext = map_40_to_13(A1_int)
        B1_ext = map_40_to_13(B1_int)
        A2_ext = map_40_to_13(A2_int)
        B2_ext = map_40_to_13(B2_int)
        
        # Extract “before boost” components
        a1_b = float(A1_ext | E1)
        b1_b = float(B1_ext | E2)
        a2_b = float(A2_ext | E1)
        b2_b = float(B2_ext | E2)
        A1_b.append(a1_b)
        B1_b.append(b1_b)
        A2_b.append(a2_b)
        B2_b.append(b2_b)
        
        # Apply the same boost to each observable
        A1_bo = lorentz_boost(A1_ext, eta_boost)
        B1_bo = lorentz_boost(B1_ext, eta_boost)
        A2_bo = lorentz_boost(A2_ext, eta_boost)
        B2_bo = lorentz_boost(B2_ext, eta_boost)
        
        a1_a = float(A1_bo | E1)
        b1_a = float(B1_bo | E2)
        a2_a = float(A2_bo | E1)
        b2_a = float(B2_bo | E2)
        A1_a.append(a1_a)
        B1_a.append(b1_a)
        A2_a.append(a2_a)
        B2_a.append(b2_a)
    
    # Convert lists to arrays
    A1_b = np.array(A1_b);  B1_b = np.array(B1_b)
    A2_b = np.array(A2_b);  B2_b = np.array(B2_b)
    A1_a = np.array(A1_a);  B1_a = np.array(B1_a)
    A2_a = np.array(A2_a);  B2_a = np.array(B2_a)
    
    # Compute covariances
    cov_A_before = np.cov(A1_b, A2_b)[0, 1]
    cov_B_before = np.cov(B1_b, B2_b)[0, 1]
    cov_A_after  = np.cov(A1_a, A2_a)[0, 1]
    cov_B_after  = np.cov(B1_a, B2_a)[0, 1]
    
    return {
        'A1_before': A1_b, 'A2_before': A2_b,
        'B1_before': B1_b, 'B2_before': B2_b,
        'A1_after':  A1_a, 'A2_after':  A2_a,
        'B1_after':  B1_a, 'B2_after':  B2_a,
        'cov_A_before': cov_A_before,
        'cov_B_before': cov_B_before,
        'cov_A_after':  cov_A_after,
        'cov_B_after':  cov_B_after
    }

# Example parameters for entangled pair
psi_mean = 0.8
psi_corr = 0.05  # standard deviation of correlated fiber-phase deviations
entangled_results = simulate_entangled_pair(
    N, psi_mean, psi_corr, theta0, theta_noise_std, eta_boost, phi_redshift
)

print("\n=== Entangled Pair Covariances ===")
print(" Cov(A1,A2) before boost:", entangled_results['cov_A_before'])
print(" Cov(B1,B2) before boost:", entangled_results['cov_B_before'])
print(" Cov(A1,A2) after boost: ", entangled_results['cov_A_after'])
print(" Cov(B1,B2) after boost: ", entangled_results['cov_B_after'])

# Scatter‐plot of A1 vs. A2 before/after
plt.figure(figsize=(5, 5))
plt.scatter(entangled_results['A1_before'], entangled_results['A2_before'],
            s=5, alpha=0.6, label="Before", color='purple')
plt.scatter(entangled_results['A1_after'], entangled_results['A2_after'],
            s=5, alpha=0.6, label="After", color='orange')
plt.xlabel("A1 (E1) subsystem 1")
plt.ylabel("A2 (E1) subsystem 2")
plt.title("Entangled A: Before vs. After Boost")
plt.legend()
plt.grid(True)
plt.show()
