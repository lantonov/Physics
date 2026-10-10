import numpy as np
import matplotlib.pyplot as plt
from clifford import Cl

##############################################
# 1. Setup internal Cl(4,0)
##############################################
# Create Cl(4,0) with signature (++++). Basis vectors: e1, e2, e3, e4.
layout40, blades40 = Cl(4, 0, firstIdx=1)
locals().update(blades40)
# Interpretation: e1, e2, e3 span space; e4 encodes the "fiber"/internal time.

# We choose the fiber generator as:
B_fiber = e3 * e4   # (e3e4)^2 = e3*e4*e3*e4 = -1 due to anticommutation.
# And choose a single base generator for simplicity:
B_base = e1 * e2    # (e1e2) represents a rotation in the e1-e2 plane.

##############################################
# 2. Define an "ideal" rotor in Cl(4,0) using a Hopf fibration
##############################################
def rotor_state(psi, theta):
    """
    Construct a rotor R = R_fiber * R_base,
    with R_fiber = exp(-psi/2 * B_fiber) and R_base = exp(-theta/2 * B_base).
    """
    R_fiber = np.exp(-0.5 * psi * B_fiber)
    R_base = np.exp(-0.5 * theta * B_base)
    return R_fiber * R_base

# Choose internal parameters (these are dimensionless)
psi_internal = 0.8          # fiber phase (radians)
theta_internal = np.deg2rad(20)  # base rotation (radians)

# Ideal (internal) rotor:
R_internal = rotor_state(psi_internal, theta_internal)

# Compute our two noncommuting observables in Cl(4,0):
# We define:
#   A_internal = R_internal * e1 * R_internal.reverse()   -- position observable (along e1)
#   B_internal = R_internal * e2 * R_internal.reverse()   -- momentum observable (along e2)
A_internal = R_internal * e1 * R_internal.inverse()
B_internal = R_internal * e2 * R_internal.inverse()

# For our demonstration, we extract the coefficient along e1 and e2 respectively:
A_coeff = float(A_internal | e1)
B_coeff = float(B_internal | e2)

print("Internal (Cl(4,0)) observables:")
print("  A (along e1) =", A_coeff)
print("  B (along e2) =", B_coeff)

##############################################
# 3. Map from Cl(4,0) to external Cl(1,3)
##############################################
# For the external observer we work with Cl(1,3) (Minkowski space).
# We identify:
#    e1, e2, e3  ---> E1, E2, E3  (space)
#    e4          ---> E0           (time)
layout13, blades13 = Cl(1, 3, firstIdx=0)
locals().update(blades13)
# Define external basis: E0 is time-like, E(1,2,3) are space-like.
E0 = e0  # external time basis vector
E1 = e1
E2 = e2
E3 = e3

def map_40_to_13(mv):
    """
    Map a multivector mv from Cl(4,0) to Cl(1,3) using the identification:
       e1 -> E1, e2 -> E2, e3 -> E3, e4 -> E0.
    """
    coeffs = mv.value
    basis_blades = layout40.bladeTupList
    mapped_mv = 0
    for coeff, blade in zip(coeffs, basis_blades):
        if abs(coeff) < 1e-12:
            continue
        if len(blade)==0:
            mapped_blade = 1.0
        else:
            mapped_blade = 1.0
            for idx in blade:
                if idx == 4:
                    mapped_blade *= E0
                elif idx == 1:
                    mapped_blade *= E1
                elif idx == 2:
                    mapped_blade *= E2
                elif idx == 3:
                    mapped_blade *= E3
        mapped_mv += coeff * mapped_blade
    return mapped_mv

# Map the two observables
A_ext = map_40_to_13(A_internal)
B_ext = map_40_to_13(B_internal)

# For convenience, extract the components:
A_ext_coeff = float(A_ext | E1)   # corresponds to the spatial (radial) component along E1.
B_ext_coeff = float(B_ext | E2)   # the other conjugate observable.
print("\nMapped to external Cl(1,3)):")
print("  A_ext (component along E1) =", A_ext_coeff)
print("  B_ext (component along E2) =", B_ext_coeff)

##############################################
# 4. Apply a Lorentz boost in Cl(1,3)
##############################################
# We now apply a boost along the E1 direction.
# In Cl(1,3) (with signature (+,-,-,-)), the boost rotor is given by:
#   L = exp(-eta/2 * (E0 E1))
# For a Lorentz boost with rapidity eta, note that:
#   (E0 E1)^2 = +1  because E0^2 = +1 and E1^2 = -1.
eta = 0.4  # rapidity
L_boost = np.exp(-0.5 * eta * (E0 * E1))
# Apply the Lorentz boost to our external observables:
A_boosted = L_boost * A_ext * L_boost.reverse()
B_boosted = L_boost * B_ext * L_boost.reverse()

# Extract the (boosted) components along the same directions.
A_boosted_coeff = float(A_boosted | E1)
B_boosted_coeff = float(B_boosted | E2)
print("\nAfter Lorentz Boost:")
print("  Boosted A (component along E1) =", A_boosted_coeff)
print("  Boosted B (component along E2) =", B_boosted_coeff)

##############################################
# 5. Comments on the Transformation
##############################################
# In the external representation, the boost rotor L_boost transforms the observables via
#   O' = L_boost * O * L_boost.reverse()
# For an observable like A (originally aligned along E1), a boost along E1 will in principle mix E0 and E1 components.
# In our simulation we explicitly extract the coefficient along E1 (similarly for B).
#
# Because the boost does not change the underlying uncertainty relations (it acts as a similarity transformation),
# the noncommutativity of the pair A and B persists. However, the numerical components change.
#
# In a detailed analysis one would follow how the intrinsic uncertainties ΔA and ΔB transform,
# but here we simply illustrate that the measured components (the "projections") are altered by the boost.
#
# For instance, a boost along E1 tends to "lengthen" the temporal component as seen in E0 and reduce the spatial E1 component
# by a factor of cosh(eta/2) (or mix them accordingly).
#
# One can extend this simulation to run an ensemble of measurements (by introducing noise) and then extract
# the uncertainties (e.g., standard deviations) before and after the boost. The product ΔA ΔB is invariant
# under similarity transformations; however, the individual distributions and their mean values do transform.

##############################################
# (Optional) Plot the Transformation Effect
##############################################
# For a visual demonstration, we will simulate a small ensemble of measurements,
# first in the external Cl(1,3) before and after boost.

N_trials = 2000
A_vals_before = []
B_vals_before = []
A_vals_after = []
B_vals_after = []

for i in range(N_trials):
    # We simulate a small fluctuation in the internal fiber angle.
    delta = np.random.normal(0, 0.01)
    R_trial_internal = rotor_state(psi_internal + delta, theta_internal)
    # Compute internal observables:
    A_trial = R_trial_internal * e1 * R_trial_internal.reverse()
    B_trial = R_trial_internal * e2 * R_trial_internal.reverse()
    # Map to external:
    A_trial_ext = map_40_to_13(A_trial)
    B_trial_ext = map_40_to_13(B_trial)
    # Extract the components along E1 and E2 (before boost)
    A_before = float(A_trial_ext | E1)
    B_before = float(B_trial_ext | E2)
    A_vals_before.append(A_before)
    B_vals_before.append(B_before)
    
    # Now apply boost:
    A_trial_boosted = L_boost * A_trial_ext * L_boost.reverse()
    B_trial_boosted = L_boost * B_trial_ext * L_boost.reverse()
    A_after = float(A_trial_boosted | E1)
    B_after = float(B_trial_boosted | E2)
    A_vals_after.append(A_after)
    B_vals_after.append(B_after)

# Plot histograms for the observable A (E1 component) before and after boost.
plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)
plt.hist(A_vals_before, bins=30, color='cyan', edgecolor='k', alpha=0.7)
plt.xlabel("A (component along E1) before boost")
plt.title("Distribution Before Boost")

plt.subplot(1,2,2)
plt.hist(A_vals_after, bins=30, color='magenta', edgecolor='k', alpha=0.7)
plt.xlabel("A (component along E1) after boost")
plt.title("Distribution After Boost")

plt.suptitle("Effect of Lorentz Boost on Observable A")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Similarly, plot scatter plots for both observables before and after boost.
plt.figure(figsize=(6, 6))
plt.scatter(A_vals_before, B_vals_before, s=5, label="Before Boost", alpha=0.6, color='blue')
plt.scatter(A_vals_after, B_vals_after, s=5, label="After Boost", alpha=0.6, color='red')
plt.xlabel("Observable A (E1 component)")
plt.ylabel("Observable B (E2 component)")
plt.title("Scatter Plot: Before (blue) vs. After Boost (red)")
plt.legend()
plt.grid(True)
plt.show()
