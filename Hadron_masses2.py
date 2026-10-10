import numpy as np
import pandas as pd
from scipy.optimize import minimize

# --------------------------------------------------------------------------
# Part 1. Optimization to fit multiple hadron masses with flavor-dependent potential
# --------------------------------------------------------------------------

# Constants (in natural units with GeV and fm)
hbarc = 0.1973269804  # GeV·fm

# Constituent quark masses (in GeV)
# These masses are used to calculate the kinetic energy of the quarks.
quark_masses = {
    'u': 0.33,   # Up quark mass
    'd': 0.33,   # Down quark mass
    's': 0.50,   # Strange quark mass
    'c': 1.60,   # Charm quark mass
    'b': 4.80    # Bottom quark mass
}

# Define a list of hadrons with their quark content, experimental mass,
# and a *relative* spin-hyperfine coupling factor.
# The 'spin_factor' is a dimensionless value that scales the spin-hyperfine
# interaction for each hadron. These factors are derived from the relative
# magnitudes of the spin corrections in the original problem description.
hadrons_data = [
    {"name": "Neutron", "quarks": ["u", "d", "d"], "exp_mass": 0.939, "spin_factor": -1.0}, # Normalized to original -0.065
    {"name": "Proton", "quarks": ["u", "u", "d"], "exp_mass": 0.938, "spin_factor": -1.0},  # Normalized to original -0.065
    {"name": "Lambda (Λ)", "quarks": ["u", "d", "s"], "exp_mass": 1.116, "spin_factor": -1.0}, # Normalized to original -0.065
    {"name": "Sigma (Σ)", "quarks": ["u", "u", "s"], "exp_mass": 1.189, "spin_factor": -0.050 / 0.065}, # Original -0.050 relative to -0.065
    {"name": "Xi (Ξ)", "quarks": ["s", "s", "d"], "exp_mass": 1.315, "spin_factor": -0.050 / 0.065}, # Original -0.050 relative to -0.065
    {"name": "Omega (Ω)", "quarks": ["s", "s", "s"], "exp_mass": 1.672, "spin_factor": 0.0}, # Original 0.0
    {"name": "J/ψ (charmonium)", "quarks": ["c", "c"], "exp_mass": 3.097, "spin_factor": -0.100 / 0.065}, # Original -0.100 relative to -0.065
    {"name": "Υ (bottomonium)", "quarks": ["b", "b"], "exp_mass": 9.460, "spin_factor": -0.150 / 0.065}  # Original -0.150 relative to -0.065
]

def calculate_hadron_mass(quark_list, spin_factor, R, Z0, k_CM, A_hf,
                          kappa_base, sigma_base, C_base,
                          kappa_SS_factor, C_SS_factor,
                          kappa_HH_factor, C_HH_factor):
    """
    Calculates the estimated mass of a hadron given its quark content and model parameters.
    This version includes flavor-dependent potential parameters for strange-strange
    and heavy-heavy quark pairs.

    Parameters:
    - quark_list (list): List of quark types composing the hadron (e.g., ["u", "d", "d"]).
    - spin_factor (float): Dimensionless relative strength of the spin-hyperfine
                           interaction for this specific hadron.
    - R (float): Bag radius in femtometers (fm).
    - Z0 (float): Dimensionless zero-point energy constant.
    - k_CM (float): Coefficient for center-of-mass correction (units: GeV·fm).
    - A_hf (float): Coefficient for hyperfine interaction (units: GeV·fm^3).
    - kappa_base (float): Base effective coupling factor for the Coulomb-like potential.
    - sigma_base (float): Base string tension parameter for the linear confinement.
    - C_base (float): Base constant term in the quark-quark potential.
    - kappa_SS_factor (float): Multiplier for kappa for strange-strange pairs.
    - C_SS_factor (float): Multiplier for C for strange-strange pairs.
    - kappa_HH_factor (float): Multiplier for kappa for heavy-heavy pairs.
    - C_HH_factor (float): Multiplier for C for heavy-heavy pairs.

    Returns:
    - float: Estimated hadron mass in GeV (Giga-electronvolts).
    """
    n_quarks = len(quark_list)
    
    # 1. Kinetic Energy (Quark Energy):
    KE_total = 0.0
    # Quark momentum within the bag, inversely proportional to the bag radius.
    p_GeV = (np.pi / R) * hbarc 
    for q_type in quark_list:
        m_qi = quark_masses[q_type]
        # Relativistic energy calculation.
        E_quark = np.sqrt(p_GeV**2 + m_qi**2)
        KE_total += E_quark

    # 2. Potential Energy (Sum over pairs with flavor-dependent parameters):
    V_total = 0.0
    # Iterate over all unique pairs of quarks in the hadron.
    for i in range(n_quarks):
        for j in range(i + 1, n_quarks):
            q1_type = quark_list[i]
            q2_type = quark_list[j]

            # Initialize with base potential parameters.
            kappa_eff = kappa_base
            sigma_eff = sigma_base # Sigma is kept universal for simplicity in this iteration
            C_eff = C_base

            # Check quark types for flavor-dependent adjustments.
            is_q1_strange = (q1_type == 's')
            is_q2_strange = (q2_type == 's')
            is_q1_heavy = (q1_type in ['c', 'b'])
            is_q2_heavy = (q2_type in ['c', 'b'])

            # Apply specific factors for strange-strange (SS) pairs.
            if is_q1_strange and is_q2_strange:
                kappa_eff *= kappa_SS_factor
                C_eff *= C_SS_factor
            # Apply specific factors for heavy-heavy (HH) pairs.
            elif is_q1_heavy and is_q2_heavy:
                kappa_eff *= kappa_HH_factor
                C_eff *= C_HH_factor
            # Light-light (u-u, u-d, d-d) and light-strange (u-s, d-s) pairs
            # will use the base kappa, sigma, and C values.

            # Calculate the potential energy for the current quark pair.
            V_pair_term = (-4/3 * kappa_eff * (hbarc / R) + sigma_eff * (R / hbarc) + C_eff)
            V_total += V_pair_term

    # 3. Zero-Point Energy:
    V_zero = -Z0 * (hbarc / R)

    # 4. Center-of-Mass (CM) Correction:
    CM_correction = k_CM / R

    # 5. Spin-Hyperfine Interaction:
    # Hyperfine correction is applied if R is positive; otherwise, it's zero.
    hyperfine_correction = A_hf * spin_factor / (R**3) if R > 0 else 0 

    # Total Hadron Mass: Summation of all energy contributions.
    total_mass = KE_total + V_total + V_zero - CM_correction + hyperfine_correction
    return total_mass

def objective_global(x):
    """
    Objective function for the global minimization.
    Calculates the sum of squared differences (residuals) between estimated
    hadron masses (from the model) and their known experimental masses.
    """
    # Unpack the 11 parameters from the optimization vector 'x'
    R, Z0, k_CM, A_hf, \
    kappa_base, sigma_base, C_base, \
    kappa_SS_factor, C_SS_factor, \
    kappa_HH_factor, C_HH_factor = x

    # Penalize nonphysical parameter values.
    # Radius must be positive. Factors should typically be positive.
    if R <= 0 or Z0 < 0 or k_CM < 0 or kappa_base < 0 or sigma_base < 0 or \
       kappa_SS_factor <= 0 or C_SS_factor <= 0 or kappa_HH_factor <= 0 or C_HH_factor <= 0:
        return np.inf

    total_squared_error = 0.0
    for hadron in hadrons_data:
        estimated_mass = calculate_hadron_mass(
            hadron["quarks"], hadron["spin_factor"],
            R, Z0, k_CM, A_hf,
            kappa_base, sigma_base, C_base,
            kappa_SS_factor, C_SS_factor,
            kappa_HH_factor, C_HH_factor
        )
        squared_difference = (estimated_mass - hadron["exp_mass"])**2
        total_squared_error += squared_difference
    return total_squared_error

# Initial guesses for the 11 parameters:
# [R, Z0, k_CM, A_hf, kappa_base, sigma_base, C_base,
#  kappa_SS_factor, C_SS_factor, kappa_HH_factor, C_HH_factor]
initial_guess_global = [
    0.9,    # R (fm)
    1.8,    # Z0 (dimensionless)
    0.1,    # k_CM (GeV·fm) - Adjusted to a small non-zero value
    -0.005, # A_hf (GeV·fm^3)
    0.5,    # kappa_base (dimensionless)
    0.18,   # sigma_base (GeV^2)
    -0.3,   # C_base (GeV)
    1.0,    # kappa_SS_factor (multiplier) - Initial guess as no change
    1.0,    # C_SS_factor (multiplier) - Initial guess as no change
    1.0,    # kappa_HH_factor (multiplier) - Initial guess as no change
    1.0     # C_HH_factor (multiplier) - Initial guess as no change
]

# Bounds for the 11 parameters:
bounds_global = [
    (0.5, 2.0),   # R (fm)
    (0.5, 3.0),   # Z0
    (0.0, 2.0),   # k_CM
    (-0.1, 0.1),  # A_hf
    (0.1, 1.0),   # kappa_base
    (0.01, 0.5),  # sigma_base
    (-1.0, 0.5),  # C_base
    (0.1, 2.0),   # kappa_SS_factor: Allow for significant change
    (0.1, 2.0),   # C_SS_factor: Allow for significant change
    (0.1, 2.0),   # kappa_HH_factor: Allow for significant change
    (0.1, 2.0)    # C_HH_factor: Allow for significant change
]

# Perform global minimization.
result_global = minimize(objective_global, initial_guess_global, bounds=bounds_global, method='L-BFGS-B') 

# Unpack the optimized parameters.
R_opt_global, Z0_opt_global, k_CM_opt, A_hf_opt, \
kappa_base_opt, sigma_base_opt, C_base_opt, \
kappa_SS_factor_opt, C_SS_factor_opt, \
kappa_HH_factor_opt, C_HH_factor_opt = result_global.x

# Display the globally optimized parameters.
df_opt_global = pd.DataFrame({
    "Fitted Parameter": [
        "Bag radius R", "Zero-point constant Z₀", "CM correction coefficient k_CM",
        "Hyperfine coefficient A_hf", 
        "Base Potential kappa", "Base Potential sigma", "Base Potential C",
        "Kappa SS Factor", "C SS Factor",
        "Kappa HH Factor", "C HH Factor"
    ],
    "Value": [
        f"{R_opt_global:.3f} fm",
        f"{Z0_opt_global:.3f}",
        f"{k_CM_opt:.3f} GeV·fm",
        f"{A_hf_opt:.4f} GeV·fm³",
        f"{kappa_base_opt:.3f}",
        f"{sigma_base_opt:.3f} GeV²",
        f"{C_base_opt:.3f} GeV",
        f"{kappa_SS_factor_opt:.3f}",
        f"{C_SS_factor_opt:.3f}",
        f"{kappa_HH_factor_opt:.3f}",
        f"{C_HH_factor_opt:.3f}"
    ]
})
print("Optimized Parameters for Global Hadron Fit (Flavor-Dependent Potential):")
print(df_opt_global)
print(f"\nOptimization success: {result_global.success}")
print(f"Total squared error: {result_global.fun:.6f}\n")


# --------------------------------------------------------------------------
# Part 2. Compute Hadron Mass Estimates Using Globally Fitted Parameters
# --------------------------------------------------------------------------

# Calculate estimated masses for each hadron using the new set of optimized parameters.
results_refined = []
for hadron in hadrons_data:
    m_est_refined = calculate_hadron_mass(
        hadron["quarks"], hadron["spin_factor"],
        R_opt_global, Z0_opt_global, k_CM_opt, A_hf_opt,
        kappa_base_opt, sigma_base_opt, C_base_opt,
        kappa_SS_factor_opt, C_SS_factor_opt,
        kappa_HH_factor_opt, C_HH_factor_opt
    )
    results_refined.append({
        "Hadron": hadron["name"],
        "Quark Content": "-".join(hadron["quarks"]),
        "Experimental Mass (GeV)": round(hadron["exp_mass"], 3),
        "Estimated Mass (GeV)": round(m_est_refined, 3),
        "Difference (GeV)": round(m_est_refined - hadron["exp_mass"], 3)
    })

# Display the comparison between experimental and estimated masses.
df_hadrons_refined = pd.DataFrame(results_refined)
print("Hadron Mass Estimates (Flavor-Dependent Potential):")
print(df_hadrons_refined)
