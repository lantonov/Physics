import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ----------------------------------------------------------------------
# Step 1. Define Fundamental Constants and BTFR Model
# ----------------------------------------------------------------------
G = 6.67430e-11            # gravitational constant in m^3/(kg*s^2)
c = 3.0e8                  # speed of light, m/s
a0 = 1.2e-10               # characteristic acceleration [m/s^2]
M_sun = 1.989e30           # solar mass in kg

# BTFR model: V^4 = G * M_bar * a0   ==> V = (G*M_bar*a0)^(1/4), with V in m/s.
def V_from_BTFR(M_bar):
    return (G * M_bar * a0)**0.25

# ----------------------------------------------------------------------
# Step 2. Read the SPARC BTFR Data from the attached file
# ----------------------------------------------------------------------
# File "BTFR_Lelli2019.txt" is in fixed-width format. Based on its description,
# we define the column specifications (0-indexed; end index exclusive):
# - Name: columns 0-12
# - log(Mb): columns 12-18  (i.e. characters 13-18 in the description)
# - e_log(Mb): columns 18-24 (we don’t necessarily need errors for this simple test)
# - Vf: columns 36-42 (flat rotation velocity in km/s)
# - e_Vf: columns 42-48 (error in km/s)
colspecs = [(0, 12), (12, 18), (18, 24), (36, 42), (42, 48)]
col_names = ['Name', 'logMb', 'elogMb', 'Vf', 'eVf']

# Read the fixed-width file.
# Depending on the file, you might need to skip header lines.
try:
    data = pd.read_fwf("BTFR_Lelli2019.txt", colspecs=colspecs, names=col_names, skiprows=1)
except Exception as e:
    print("Error reading BTFR_Lelli2019.txt:", e)
    exit()

# Convert the columns to numeric (set errors='coerce' to turn non-numeric into NaN)
data['logMb'] = pd.to_numeric(data['logMb'], errors='coerce')
data['Vf'] = pd.to_numeric(data['Vf'], errors='coerce')

# Drop rows with missing or zero values in key columns.
data = data.dropna(subset=['logMb', 'Vf'])
data = data[(data['logMb'] > 0) & (data['Vf'] > 0)]
data.reset_index(drop=True, inplace=True)

# ----------------------------------------------------------------------
# Step 3. Prepare BTFR Data: convert logMb to baryonic mass in kg.
# logMb is defined as log10(Mbar) where Mbar is given in solar masses.
data['M_bar_kg'] = (10 ** data['logMb']) * M_sun

# For our subsequent regression, also convert Vf from km/s to m/s.
data['Vf_mps'] = data['Vf'] * 1e3

# ----------------------------------------------------------------------
# Step 4. Perform Regression in Log-Log Space
# ----------------------------------------------------------------------
# We take the model as log10(V) = (1/4)[log10(G) + log10(M_bar_kg) + log10(a0)]
# Let’s define:
#    X = log10(M_bar_kg)
#    Y = log10(Vf_mps)
X = np.log10(data['M_bar_kg'])
Y = np.log10(data['Vf_mps'])

# Fit a simple linear model: Y = m * X + b
# Using np.polyfit
params, cov = np.polyfit(X, Y, 1, cov=True)
slope, intercept = params
slope_err = np.sqrt(cov[0,0])
print(f"Fitted BTFR slope = {slope:.3f} +/- {slope_err:.3f} (the model predicts 0.25)")

# ----------------------------------------------------------------------
# Step 5. Plot BTFR: Data and Model Prediction
# ----------------------------------------------------------------------
# Create a smooth range for M_bar (in kg)
M_bar_range = np.logspace(np.log10(data['M_bar_kg'].min()), 
                          np.log10(data['M_bar_kg'].max()), 200)
V_model_mps = V_from_BTFR(M_bar_range)
V_model_kmps = V_model_mps / 1e3

plt.figure(figsize=(8,6))
plt.loglog(data['M_bar_kg']/M_sun, data['Vf'], 'bo', label="Observed Vf (km/s)")
plt.loglog(M_bar_range/M_sun, V_model_kmps, 'r-', lw=2,
           label=r"Model: $V=(GM_{bar}a_0)^{1/4}$")
plt.xlabel("Baryonic Mass [$M_\\odot$]", fontsize=14)
plt.ylabel("Rotation Velocity (km/s)", fontsize=14)
plt.title("Baryonic Tully--Fisher Relation", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="--")
plt.show()

# ----------------------------------------------------------------------
# Step 6. Comparison: Radial Acceleration Relation (RAR) – using mock RAR data
# ----------------------------------------------------------------------
# Typically, the RAR is expressed as:
#    g_obs = g_bar / (1 - exp(-sqrt(g_bar/a0)))
# where g_bar = G * M_bar / R^2.
# Without individual galaxy R_last values in this BTFR file, we assume an average R.
R_avg_kpc = 10   # average last measured radius in kpc (for example)
R_avg = R_avg_kpc * 3.086e19  # convert to m
data['g_bar'] = G * data['M_bar_kg'] / (R_avg**2)  # in m/s^2

def predicted_g_obs(g_bar, a0=a0):
    return g_bar / (1 - np.exp(-np.sqrt(g_bar/a0)))

data['g_obs_pred'] = predicted_g_obs(data['g_bar'], a0)

# To simulate observational scatter, add small random scatter (since the file doesn't contain this)
np.random.seed(42)
data['g_obs_mock'] = data['g_obs_pred'] * np.random.uniform(0.9, 1.1, size=len(data))

plt.figure(figsize=(8,6))
plt.loglog(data['g_bar'], data['g_obs_pred'], 'r-', lw=2, label="Predicted RAR")
plt.scatter(data['g_bar'], data['g_obs_mock'], color='blue', s=20, alpha=0.7,
            label="Mock Observational Data")
plt.xlabel("Baryonic Acceleration $g_{bar}$ (m/s$^2$)", fontsize=14)
plt.ylabel("Observed Acceleration $g_{obs}$ (m/s$^2$)", fontsize=14)
plt.title("Redial Acceleration Relation (RAR)", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="--")
plt.show()
