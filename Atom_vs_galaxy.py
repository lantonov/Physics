import numpy as np
import matplotlib.pyplot as plt
from clifford import Cl

##############################
# 1. Set Up Cl(4,0)
##############################
# We define a Clifford algebra with signature (++++). This has four orthonormal basis vectors:
# e1, e2, e3, e4.
layout, blades = Cl(4, 0, firstIdx=1)
locals().update(blades)
# Our interpretation:
#  - The three “spatial” directions are spanned by e1, e2, e3.
#  - The “time” (or internal phase, as in the Hopf fiber) is encoded in e4.
#
# We choose the fiber generator as:
B_fiber = e3 * e4   # will govern the phase (time evolution)
# And the base generator (for simplicity, one rotation in the plane of e1 and e2):
B_base  = e1 * e2

##############################
# 2. Define a Manual Reverse Function
##############################
def compute_reverse(mv):
    """
    Compute the reverse of a multivector mv.
    For each basis blade of grade g, multiply its coefficient by (-1)^(g*(g-1)/2).
    """
    coeffs = mv.value.copy()  # copy the coefficient array
    for i, blade in enumerate(mv.layout.bladeTupList):
        grade = len(blade)
        sign = (-1) ** ((grade * (grade - 1)) // 2)
        coeffs[i] *= sign
    return mv.layout.MultiVector(coeffs)

##############################
# 3. Define an Ideal Rotor State
##############################
def rotor_state(psi, theta):
    """
    Construct a rotor in Cl(4,0) with:
      - R_fiber = exp(-psi/2 * B_fiber)
      - R_base  = exp(-theta/2 * B_base)
    The overall state R = R_fiber * R_base.
    """
    R_fiber = np.exp(-0.5 * psi * B_fiber)
    R_base = np.exp(-0.5 * theta * B_base)
    return R_fiber * R_base

# Choose an ideal state with parameters psi and theta.
# These are our "central" (unperturbed) values.
psi_ideal = 0.8          # intrinsic phase (radians) in the fiber direction
theta_ideal = np.deg2rad(20)  # base rotation (radians)

# Ideal state:
R0 = rotor_state(psi_ideal, theta_ideal)

##############################
# 4. Introduce Noise to Model Intrinsic Fluctuations
##############################
def simulate_trial(R0, noise_std):
    """
    Simulate one measurement trial by perturbing the ideal rotor R0.
    We introduce a small random perturbation in the fiber angle psi.
    For simplicity, we assume the noise affects only the fiber part.
    """
    # Draw a random noise value delta with standard deviation noise_std.
    delta = np.random.normal(0, noise_std)
    # Perturb psi by delta; keep theta fixed.
    psi_trial = psi_ideal + delta
    R_trial = rotor_state(psi_trial, theta_ideal)
    # Normalize (should be unit, but numerical drifts can occur)
    norm_R = np.sqrt(abs(R_trial)**2)
    if norm_R != 0:
        R_trial = R_trial / norm_R
    return R_trial

##############################
# 5. Define Observables: Radial Position and Radial Momentum
##############################
def measure_observables(R):
    """
    Given a rotor state R (in Cl(4,0)), compute two observables:
      A = R * e1 * (manual reverse of R)  (interpreted as a "radial position" along e1)
      B = R * e2 * (manual reverse of R)  (interpreted as a conjugate "radial momentum" along e2)
    Then extract the scalar component along the same basis vector.
    """
    A_full = R * e1 * compute_reverse(R)
    B_full = R * e2 * compute_reverse(R)
    A_val = float(A_full | e1)   # projection onto e1
    B_val = float(B_full | e2)   # projection onto e2
    return A_val, B_val

##############################
# 6. Set Up the Two Systems: Atom vs. Galaxy
##############################
# We simulate two ensembles with the same ideal state R0 but with different noise levels.
# For the atomic system (tight nucleus), we choose a very low noise_std.
noise_atom = 0.01
# For the galactic system (diffuse central bulge), we choose a higher noise_std.
noise_galaxy = 0.1

N_trials = 1000

# Atomic system simulation
A_atom_vals = []
B_atom_vals = []
for i in range(N_trials):
    R_trial = simulate_trial(R0, noise_atom)
    A_val, B_val = measure_observables(R_trial)
    A_atom_vals.append(A_val)
    B_atom_vals.append(B_val)
A_atom_vals = np.array(A_atom_vals)
B_atom_vals = np.array(B_atom_vals)

# Galactic system simulation
A_galaxy_vals = []
B_galaxy_vals = []
for i in range(N_trials):
    R_trial = simulate_trial(R0, noise_galaxy)
    A_val, B_val = measure_observables(R_trial)
    A_galaxy_vals.append(A_val)
    B_galaxy_vals.append(B_val)
A_galaxy_vals = np.array(A_galaxy_vals)
B_galaxy_vals = np.array(B_galaxy_vals)

##############################
# 7. Compute Uncertainties
##############################
sigma_A_atom = np.std(A_atom_vals)
sigma_B_atom = np.std(B_atom_vals)
uncertainty_atom = sigma_A_atom * sigma_B_atom

sigma_A_galaxy = np.std(A_galaxy_vals)
sigma_B_galaxy = np.std(B_galaxy_vals)
uncertainty_galaxy = sigma_A_galaxy * sigma_B_galaxy

print("Atomic System (Nucleus):")
print("  sigma_A (radial position) =", sigma_A_atom)
print("  sigma_B (radial momentum) =", sigma_B_atom)
print("  Uncertainty product (ΔA·ΔB) =", uncertainty_atom)

print("\nGalactic System (Central Bulge):")
print("  sigma_A (radial position) =", sigma_A_galaxy)
print("  sigma_B (radial momentum) =", sigma_B_galaxy)
print("  Uncertainty product (ΔA·ΔB) =", uncertainty_galaxy)

##############################
# 8. Plotting the Distributions and Scatter Plots
##############################
plt.figure(figsize=(12, 5))
plt.subplot(2, 2, 1)
plt.hist(A_atom_vals, bins=30, color='blue', edgecolor='black')
plt.xlabel("Atom: Position Observable")
plt.title("Atom: Distribution of A")

plt.subplot(2, 2, 2)
plt.hist(B_atom_vals, bins=30, color='green', edgecolor='black')
plt.xlabel("Atom: Momentum Observable")
plt.title("Atom: Distribution of B")

plt.subplot(2, 2, 3)
plt.hist(A_galaxy_vals, bins=30, color='red', edgecolor='black')
plt.xlabel("Galaxy: Position Observable")
plt.title("Galaxy: Distribution of A")

plt.subplot(2, 2, 4)
plt.hist(B_galaxy_vals, bins=30, color='orange', edgecolor='black')
plt.xlabel("Galaxy: Momentum Observable")
plt.title("Galaxy: Distribution of B")

plt.suptitle("Distributions of Two Noncommuting Observables (Atom vs. Galaxy)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(A_atom_vals, B_atom_vals, c='blue', s=10, alpha=0.6, label="Atom")
plt.scatter(A_galaxy_vals, B_galaxy_vals, c='red', s=10, alpha=0.6, label="Galaxy")
plt.xlabel("Radial Position Observable")
plt.ylabel("Radial Momentum Observable")
plt.title("Scatter Plot: Atom (blue) vs. Galaxy (red)")
plt.legend()
plt.grid(True)
plt.show()
