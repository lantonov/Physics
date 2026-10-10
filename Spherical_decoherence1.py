import numpy as np
import matplotlib.pyplot as plt
from clifford import Cl

##############################################
# 1. Setup internal Cl(4,0)
##############################################
layout40, blades40 = Cl(4, 0, firstIdx=1)
locals().update(blades40)
# Save internal basis vectors to avoid later conflicts with external Cl(1,3)
e1_int, e2_int, e3_int, e4_int = e1, e2, e3, e4

# Interpretation: e1, e2, e3 span space; e4 encodes the "fiber"/internal time.
B_fiber = e3_int * e4_int   # (e3e4)^2 = -1
B_base  = e1_int * e2_int   # represents a rotation in the e1-e2 plane

##############################################
# 2. Define helper function: compute_reverse
##############################################
def compute_reverse(mv):
    """
    Compute the reverse of a multivector mv.
    For each term, multiply its coefficient by (-1)^(grade*(grade-1)/2).
    """
    coeffs = mv.value.copy()  # Copy coefficients
    for i, blade in enumerate(mv.layout.bladeTupList):
        grade = len(blade)
        sign = (-1) ** ((grade * (grade - 1)) // 2)
        coeffs[i] *= sign
    return mv.layout.MultiVector(coeffs)

##############################################
# 3. Define an "ideal" rotor in Cl(4,0) using a Hopf fibration
##############################################
def rotor_state(psi, theta):
    """
    Construct a rotor R = R_fiber * R_base,
    where R_fiber = exp(-psi/2 * B_fiber) and R_base = exp(-theta/2 * B_base).
    """
    R_fiber = np.exp(-0.5 * psi * B_fiber)
    R_base  = np.exp(-0.5 * theta * B_base)
    return R_fiber * R_base

# Choose internal parameters in radians.
psi_internal   = 0.8
theta_internal = np.deg2rad(20)

# Ideal (internal) rotor in Cl(4,0)
R_internal = rotor_state(psi_internal, theta_internal)

# Use the internal basis (e1_int, e2_int) for observables.
A_internal = R_internal * e1_int * compute_reverse(R_internal)
B_internal = R_internal * e2_int * compute_reverse(R_internal)

# For demonstration, extract the coefficient along e1_int and e2_int.
A_coeff = float(A_internal | e1_int)
B_coeff = float(B_internal | e2_int)

print("Internal (Cl(4,0)) observables:")
print("  A (along e1_int) =", A_coeff)
print("  B (along e2_int) =", B_coeff)

##############################################
# 4. Map from Cl(4,0) to external Cl(1,3)
##############################################
layout13, blades13 = Cl(1, 3, firstIdx=0)
locals().update(blades13)
# Define external basis: E0 is time-like, E1, E2, E3 are space-like.
E0 = e0  # external time basis vector
E1 = e1
E2 = e2
E3 = e3

def map_40_to_13(mv):
    """
    Map a multivector mv from Cl(4,0) to Cl(1,3) using:
       e1 -> E1, e2 -> E2, e3 -> E3, e4 -> E0.
    """
    coeffs = mv.value
    basis_blades = layout40.bladeTupList
    mapped_mv = 0
    for coeff, blade in zip(coeffs, basis_blades):
        if abs(coeff) < 1e-12:
            continue
        if len(blade) == 0:
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

# Map the two observables to the external algebra Cl(1,3)
A_ext = map_40_to_13(A_internal)
B_ext = map_40_to_13(B_internal)

A_ext_coeff = float(A_ext | E1)
B_ext_coeff = float(B_ext | E2)
print("\nMapped to external Cl(1,3)):")
print("  A_ext (component along E1) =", A_ext_coeff)
print("  B_ext (component along E2) =", B_ext_coeff)

##############################################
# 5. Apply a Lorentz boost in Cl(1,3)
##############################################
eta = 0.4  # rapidity
L_boost = np.exp(-0.5 * eta * (E0 * E1))
# Use compute_reverse for the boost rotor as well.
A_boosted = L_boost * A_ext * compute_reverse(L_boost)
B_boosted = L_boost * B_ext * compute_reverse(L_boost)

A_boosted_coeff = float(A_boosted | E1)
B_boosted_coeff = float(B_boosted | E2)
print("\nAfter Lorentz Boost:")
print("  Boosted A (component along E1) =", A_boosted_coeff)
print("  Boosted B (component along E2) =", B_boosted_coeff)

##############################################
# 6. Simulation Loop and Plotting
##############################################
N_trials = 2000
A_vals_before = []
B_vals_before = []
A_vals_after = []
B_vals_after = []

for i in range(N_trials):
    # Simulate a small fluctuation in the internal fiber angle.
    delta = np.random.normal(0, 0.01)
    R_trial_internal = rotor_state(psi_internal + delta, theta_internal)
    # Compute internal observables using the internal basis (e1_int/e2_int)
    A_trial = R_trial_internal * e1_int * compute_reverse(R_trial_internal)
    B_trial = R_trial_internal * e2_int * compute_reverse(R_trial_internal)
    # Map to external Cl(1,3)
    A_trial_ext = map_40_to_13(A_trial)
    B_trial_ext = map_40_to_13(B_trial)
    # Extract the components along E1 and E2 before the boost.
    A_before = float(A_trial_ext | E1)
    B_before = float(B_trial_ext | E2)
    A_vals_before.append(A_before)
    B_vals_before.append(B_before)
    
    # Apply the Lorentz boost.
    A_trial_boosted = L_boost * A_trial_ext * compute_reverse(L_boost)
    B_trial_boosted = L_boost * B_trial_ext * compute_reverse(L_boost)
    A_after = float(A_trial_boosted | E1)
    B_after = float(B_trial_boosted | E2)
    A_vals_after.append(A_after)
    B_vals_after.append(B_after)

# Plot histograms for observable A (component along E1) before and after boost.
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(A_vals_before, bins=30, color='cyan', edgecolor='k', alpha=0.7)
plt.xlabel("A (component along E1) before boost")
plt.title("Distribution Before Boost")

plt.subplot(1, 2, 2)
plt.hist(A_vals_after, bins=30, color='magenta', edgecolor='k', alpha=0.7)
plt.xlabel("A (component along E1) after boost")
plt.title("Distribution After Boost")
plt.suptitle("Effect of Lorentz Boost on Observable A")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Scatter plot of both observables before and after boost.
plt.figure(figsize=(6, 6))
plt.scatter(A_vals_before, B_vals_before, s=5, label="Before Boost", alpha=0.6, color='blue')
plt.scatter(A_vals_after, B_vals_after, s=5, label="After Boost", alpha=0.6, color='red')
plt.xlabel("Observable A (E1 component)")
plt.ylabel("Observable B (E2 component)")
plt.title("Scatter Plot: Before (blue) vs. After Boost (red)")
plt.legend()
plt.grid(True)
plt.show()
