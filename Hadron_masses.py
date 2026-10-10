import numpy as np
import pandas as pd
from scipy.optimize import minimize

# --------------------------------------------------------------------------
# Part 1. Optimization to fit the neutron mass
# --------------------------------------------------------------------------

# Constants (in natural units with GeV and fm)
hbarc = 0.1973269804  # GeV·fm
kappa = 0.52          # dimensionless (effective coupling factor)
sigma = 0.18          # effective string tension constant (units such that sigma * (R/hbarc) is in GeV)
C = -0.3              # constant term in potential (GeV)
m_q = 0.33            # quark mass for light quarks (GeV)
E_spin = -0.065       # spin correction for neutron (GeV)
target_mass = 0.939   # neutron mass in GeV

# Define the objective function:
def objective(x):
    R, Z0 = x
    if R <= 0 or Z0 < 0:
        return np.inf  # Penalize nonphysical values
    # For the neutron (3 quarks → 3 pairs)
    V_pair = (-4/3 * kappa * (hbarc / R) + sigma * (R / hbarc) + C)
    V_total = 3 * V_pair  # there are 3 quark pairs in a baryon
    # Assume an identical momentum scale for each quark:
    p_GeV = (np.pi / R) * hbarc
    E_quark = np.sqrt(p_GeV**2 + m_q**2)
    KE_total = 3 * E_quark
    V_zero = -Z0 * (hbarc / R)
    total_mass = KE_total + V_total + V_zero + E_spin
    return (total_mass - target_mass)**2

# Initial guess and bounds: R (in fm) and Z0 (dimensionless)
initial_guess = [1.0, 1.5]
bounds = [(0.1, 3.0), (0.0, 5.0)]

# Perform minimization
result = minimize(objective, initial_guess, bounds=bounds)
R_opt, Z0_opt = result.x

# Prepare and display the optimized parameters
df_opt = pd.DataFrame({
    "Fitted Parameter": ["Bag radius R", "Zero-point constant Z₀", "Target neutron mass"],
    "Value": [f"{R_opt:.3f} fm", f"{Z0_opt:.3f}", f"{target_mass:.3f} GeV"]
})
print("Optimized Parameters for the Neutron Fit:")
print(df_opt)
print("\n")

# --------------------------------------------------------------------------
# Part 2. Compute Hadron Mass Estimates Using Fitted Parameters
# --------------------------------------------------------------------------

# Revised constituent quark masses (in GeV)
quark_masses = {
    'u': 0.33,   # GeV
    'd': 0.33,   # GeV
    's': 0.50,   # GeV
    'c': 1.60,   # GeV
    'b': 4.80    # GeV
}

# Define a list of hadrons with their quark content and a spin-correction
hadrons = [
    {"name": "Neutron", "quarks": ["u", "d", "d"], "spin": -0.065},
    {"name": "Proton", "quarks": ["u", "u", "d"], "spin": -0.065},
    {"name": "Lambda (Λ)", "quarks": ["u", "d", "s"], "spin": -0.065},
    {"name": "Sigma (Σ)", "quarks": ["u", "u", "s"], "spin": -0.050},
    {"name": "Xi (Ξ)", "quarks": ["s", "s", "d"], "spin": -0.050},
    {"name": "Omega (Ω)", "quarks": ["s", "s", "s"], "spin": 0.000},
    {"name": "J/ψ (charmonium)", "quarks": ["c", "c"], "spin": -0.100},
    {"name": "Υ (bottomonium)", "quarks": ["b", "b"], "spin": -0.150}
]

# Define the function that estimates hadron mass from its quark content,
# including a spin correction.
def mass_with_flavor(quark_list, spin_corr, R, Z0):
    n_quarks = len(quark_list)
    n_pairs = n_quarks * (n_quarks - 1) // 2
    V_pair = (-4/3 * kappa * (hbarc / R) + sigma * (R / hbarc) + C)
    V_total = n_pairs * V_pair
    KE_total = 0.0
    # The same momentum scale is assumed for each quark
    p_GeV = (np.pi / R) * hbarc
    for q in quark_list:
        m_qi = quark_masses[q]
        E_quark = np.sqrt(p_GeV**2 + m_qi**2)
        KE_total += E_quark
    V_zero = -Z0 * (hbarc / R)
    return KE_total + V_total + V_zero + spin_corr

# Compute estimated masses for each hadron
results = []
for hadron in hadrons:
    m_est = mass_with_flavor(hadron["quarks"], hadron["spin"], R_opt, Z0_opt)
    results.append({
        "Hadron": hadron["name"],
        "Quark Content": "-".join(hadron["quarks"]),
        "Spin Correction (GeV)": hadron["spin"],
        "Estimated Mass (GeV)": round(m_est, 3)
    })

df_hadrons = pd.DataFrame(results)
print("Hadron Mass Estimates (using optimized R and Z₀):")
print(df_hadrons)
