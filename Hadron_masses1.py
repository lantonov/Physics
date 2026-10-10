import numpy as np
import pandas as pd
from scipy.optimize import minimize

# --------------------------------------------------------------------------
# Part 1. Optimization to fit multiple hadron masses
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

def calculate_hadron_mass(quark_list, spin_factor, R, Z0, k_CM, A_hf, kappa, sigma, C):
    """
    Calculates the estimated mass of a hadron given its quark content and model parameters.

    This function incorporates kinetic energy, potential energy (from quark-quark
    interactions), zero-point energy, center-of-mass corrections, and spin-hyperfine
    interactions.

    Parameters:
    - quark_list (list): List of quark types composing the hadron (e.g., ["u", "d", "d"]).
    - spin_factor (float): Dimensionless relative strength of the spin-hyperfine
                           interaction for this specific hadron.
    - R (float): Bag radius in femtometers (fm).
    - Z0 (float): Dimensionless zero-point energy constant.
    - k_CM (float): Coefficient for center-of-mass correction (units: GeV·fm).
                    This term accounts for the spurious kinetic energy of the CM.
    - A_hf (float): Coefficient for hyperfine interaction (units: GeV·fm^3).
                    This term describes spin-dependent mass splittings.
    - kappa (float): Dimensionless effective coupling factor for the Coulomb-like
                     part of the quark-quark potential.
    - sigma (float): String tension parameter for the linear confinement part
                     of the potential (units: GeV^2).
    - C (float): Constant term in the quark-quark potential (units: GeV).

    Returns:
    - float: Estimated hadron mass in GeVe (Giga-electronvolts).
    """
    n_quarks = len(quark_list)
    
    # 1. Kinetic Energy (Quark Energy):
    # Assumes a common momentum scale for all quarks inside the bag, inversely
    # proportional to the bag radius R. This is a common approximation.
    KE_total = 0.0
    p_GeV = (np.pi / R) * hbarc 
    for q_type in quark_list:
        m_qi = quark_masses[q_type]
        # Quark energy is calculated relativistically (E = sqrt(p^2 + m^2)).
        E_quark = np.sqrt(p_GeV**2 + m_qi**2)
        KE_total += E_quark

    # 2. Potential Energy (Sum over pairs):
    # This model uses an additive pair potential, consisting of a Coulomb-like term
    # (-4/3 * kappa * hbarc / R), a linear confinement term (sigma * R / hbarc),
    # and a constant offset C.
    n_pairs = n_quarks * (n_quarks - 1) // 2 if n_quarks > 1 else 0 
    V_pair_term = (-4/3 * kappa * (hbarc / R) + sigma * (R / hbarc) + C)
    V_total = n_pairs * V_pair_term
    
    # 3. Zero-Point Energy:
    # A negative energy term that accounts for the zero-point fluctuations
    # of the quark fields within the bag.
    V_zero = -Z0 * (hbarc / R)

    # 4. Center-of-Mass (CM) Correction:
    # Subtracts a phenomenological term proportional to 1/R to remove the spurious
    # kinetic energy associated with the unphysical center-of-mass motion of the bag.
    CM_correction = k_CM / R

    # 5. Spin-Hyperfine Interaction:
    # Accounts for mass splittings due to the spin-spin interaction between quarks.
    # It scales with 1/R^3 (due to dependence on the wavefunction at the origin)
    # and a hadron-specific 'spin_factor' to capture relative interaction strengths.
    hyperfine_correction = A_hf * spin_factor / (R**3) if R > 0 else 0 

    # Total Hadron Mass: Summation of all energy contributions.
    total_mass = KE_total + V_total + V_zero - CM_correction + hyperfine_correction
    return total_mass

def objective_global(x):
    """
    Objective function for the global minimization.
    Calculates the sum of squared differences (residuals) between estimated
    hadron masses (from the model) and their known experimental masses.
    This is the function that the optimizer tries to minimize.
    """
    # Unpack the parameters from the optimization vector 'x'
    R, Z0, k_CM, A_hf, kappa, sigma, C = x

    # Penalize nonphysical parameter values by returning a very large error.
    # Radius (R) must be positive. Energy coefficients (Z0, k_CM, kappa, sigma)
    # are typically non-negative based on their physical interpretations.
    if R <= 0 or Z0 < 0 or k_CM < 0 or kappa < 0 or sigma < 0:
        return np.inf

    total_squared_error = 0.0
    # Loop through each hadron in the dataset to calculate its estimated mass
    # and contribute to the total squared error.
    for hadron in hadrons_data:
        estimated_mass = calculate_hadron_mass(
            hadron["quarks"], hadron["spin_factor"],
            R, Z0, k_CM, A_hf, kappa, sigma, C
        )
        squared_difference = (estimated_mass - hadron["exp_mass"])**2
        total_squared_error += squared_difference
    return total_squared_error

# Initial guesses for the parameters: [R, Z0, k_CM, A_hf, kappa, sigma, C]
# These values are chosen based on typical ranges found in hadron physics
# models and the initial parameters provided in the problem description.
initial_guess_global = [
    0.9,    # R (fm): Starting near the suggested 0.8-1.0 fm range for light hadrons.
    1.8,    # Z0 (dimensionless): Starting near the suggested 1.5-2.0 range.
    0.5,    # k_CM (GeV·fm): Initial guess for the CM correction coefficient.
    -0.005, # A_hf (GeV·fm^3): Initial guess for the hyperfine interaction coefficient.
    0.52,   # kappa (dimensionless): Carried over from original problem.
    0.18,   # sigma (GeV^2): Carried over from original problem (assuming GeV^2 units).
    -0.3    # C (GeV): Carried over from original problem.
]

# Bounds for the parameters: Defines the permissible range for each parameter
# during the optimization process. This helps keep the parameters physically realistic.
bounds_global = [
    (0.5, 2.0),   # R (fm): Allows for a realistic range for hadron radii.
    (0.5, 3.0),   # Z0: Expanded range for better optimization.
    (0.0, 2.0),   # k_CM: Must be non-negative as it represents a magnitude to be subtracted.
    (-0.1, 0.1),  # A_hf: Can be negative or positive depending on the nature of interaction.
    (0.1, 1.0),   # kappa: Typical range for coupling constants.
    (0.01, 0.5),  # sigma: String tension, set a small positive lower bound.
    (-1.0, 0.5)   # C: Constant potential term.
]

# Perform global minimization using the 'L-BFGS-B' method, which is suitable
# for bounded optimization problems.
result_global = minimize(objective_global, initial_guess_global, bounds=bounds_global, method='L-BFGS-B') 

# Unpack the optimized parameter values from the optimization result.
R_opt_global, Z0_opt_global, k_CM_opt, A_hf_opt, kappa_opt, sigma_opt, C_opt = result_global.x

# Display the globally optimized parameters in a structured DataFrame for clarity.
df_opt_global = pd.DataFrame({
    "Fitted Parameter": [
        "Bag radius R", "Zero-point constant Z₀", "CM correction coefficient k_CM",
        "Hyperfine coefficient A_hf", "Potential kappa", "Potential sigma", "Potential C"
    ],
    "Value": [
        f"{R_opt_global:.3f} fm",
        f"{Z0_opt_global:.3f}",
        f"{k_CM_opt:.3f} GeV·fm",
        f"{A_hf_opt:.4f} GeV·fm³",
        f"{kappa_opt:.3f}",
        f"{sigma_opt:.3f} GeV²", # Corrected unit display for sigma
        f"{C_opt:.3f} GeV"
    ]
})
print("Optimized Parameters for Global Hadron Fit:")
print(df_opt_global)
print(f"\nOptimization success: {result_global.success}")
print(f"Total squared error: {result_global.fun:.6f}\n")


# --------------------------------------------------------------------------
# Part 2. Compute Hadron Mass Estimates Using Globally Fitted Parameters
# --------------------------------------------------------------------------

# Calculate estimated masses for each hadron using the newly optimized parameters.
results_refined = []
for hadron in hadrons_data:
    m_est_refined = calculate_hadron_mass(
        hadron["quarks"], hadron["spin_factor"],
        R_opt_global, Z0_opt_global, k_CM_opt, A_hf_opt,
        kappa_opt, sigma_opt, C_opt
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
print("Hadron Mass Estimates (using globally optimized R, Z₀, k_CM, A_hf, κ, σ, C):")
print(df_hadrons_refined)
