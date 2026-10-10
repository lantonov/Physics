import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -----------------------------
# Constants
# -----------------------------
G = 6.67430e-11          # gravitational constant, m^3/(kg*s^2)
c = 3.0e8                # speed of light, m/s
a0 = 1.2e-10             # characteristic acceleration, m/s^2
M_sun = 1.989e30         # solar mass, kg

# -----------------------------
# Read in SPARC BTFR Data
# -----------------------------
# The attached file "BTFR_Lelli2019.txt" is fixed-width format.
# We know the columns start at fixed positions:
#   - Name: columns 1-12
#   - log(Mb): columns 13-18  (in solMass units, i.e. log(M_baryonic/M_sun))
#   - e_log(Mb): columns 19-24 (error, but we can ignore errors for our simple fit)
#   - Vf: columns 37-42 (rotation velocity in km/s)
#   - e_Vf: columns 43-48 (error in km/s, optional)

# We can use pandas.read_fwf. Specify colspecs (0-indexed, end-exclusive).
colspecs = [(0, 12), (12, 18), (18, 24), (36, 42), (42, 48)]
column_names = ['Name', 'logMb', 'elogMb', 'Vf', 'eVf']

# Read the file. Change the file name if needed.
# For example, if the file is in your working directory:
data = pd.read_fwf("BTFR_Lelli2019.txt", colspecs=colspecs, names=column_names, skiprows=1)

# Remove any rows that don't have numeric values in logMb or Vf (like header info)
data = data[pd.to_numeric(data['logMb'], errors='coerce').notnull()]
data.reset_index(drop=True, inplace=True)

# Convert columns to numeric types
data['logMb'] = pd.to_numeric(data['logMb'])
data['Vf'] = pd.to_numeric(data['Vf'])

# -----------------------------
# Prepare Data for BTFR Analysis
# -----------------------------
# Calculate the baryonic mass in kg: M_bar = 10^(logMb) * M_sun
data['M_bar_kg'] = (10 ** data['logMb']) * M_sun

# Model prediction: V_pred = (G * M_bar * a0)^(1/4) in m/s, then convert to km/s.
def V_predicted(M_bar):
    return (G * M_bar * a0) ** 0.25

data['V_pred_mps'] = V_predicted(data['M_bar_kg'])
data['V_pred_kmps'] = data['V_pred_mps'] / 1e3

# -----------------------------
# Fit a Line in Log-Log Space to the Data (BTFR)
# -----------------------------
# Using the model: log(V) = 0.25 * [log(G) + log(M_bar) + log(a0)]
# We now perform a linear fit of log(V_obs) vs. log(M_bar) and see the slope.
log_M_bar = np.log10(data['M_bar_kg'])
log_V_obs = np.log10(data['Vf'])
# Fit a line: log(V) = m * log(M_bar) + b
params, cov = np.polyfit(log_M_bar, log_V_obs, 1, cov=True)
slope, intercept = params
slope_err = np.sqrt(cov[0, 0])
print(f"Fitted BTFR slope = {slope:.3f} +/- {slope_err:.3f} (model expects 0.25)")

# -----------------------------
# Plot BTFR: Observed Data and Model Prediction
# -----------------------------
plt.figure(figsize=(8,6))
plt.loglog(data['M_bar_kg']/M_sun, data['Vf'], 'bo', label="Observed Vf (km/s)")
# Create a smooth range of masses for the model curve:
M_bar_range = np.logspace(np.log10(data['M_bar_kg'].min()), 
                          np.log10(data['M_bar_kg'].max()), 200)
V_model = V_predicted(M_bar_range) / 1e3  # in km/s
plt.loglog(M_bar_range/M_sun, V_model, 'r-', lw=2, label="Model Prediction: $V=(G M a_0)^{1/4}$")
plt.xlabel("Baryonic Mass $M_{bar}$ ($M_\\odot$)", fontsize=14)
plt.ylabel("Rotation Velocity $V$ (km/s)", fontsize=14)
plt.title("Baryonic Tully--Fisher Relation (BTFR)", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="--")
plt.show()

# -----------------------------
# Comparison with the Radial Acceleration Relation (RAR)
# -----------------------------
# For the RAR, the standard form is:
#    g_obs = (g_bar) / (1 - exp(-sqrt(g_bar / a0)))
# g_bar is computed from the baryonic mass at the last measured point, e.g.,
#    g_bar = G M_bar / R_last^2.
# The SPARC file contains R_last? In our file, this short BTFR file might not have it.
# Instead, we simulate g_bar using the BTFR data and assume R_last ~ a typical scale.
# For example, assume an average R_last = 10 kpc (1 kpc = 3.086e19 m).

R_avg = 10 * 3.086e19  # 10 kpc in meters
data['g_bar'] = G * data['M_bar_kg'] / (R_avg ** 2)   # in m/s^2

# We assume the model takes the form:
def predicted_g_obs(g_bar, a0=a0):
    return g_bar / (1 - np.exp(-np.sqrt(g_bar/a0)))

data['g_obs_pred'] = predicted_g_obs(data['g_bar'], a0)

plt.figure(figsize=(8,6))
plt.loglog(data['g_bar'], data['g_obs_pred'], 'r-', lw=2, label="Predicted RAR")
# For illustration, add the observed (or mock) RAR using the SPARC file's g_obs if available.
# Since the BTFR file provided here does not include g_obs we simulate small scatter.
g_obs_mock = data['g_obs_pred'] * np.random.uniform(0.9, 1.1, size=len(data))
plt.scatter(data['g_bar'], g_obs_mock, color='blue', s=20, alpha=0.7, label="Mock Observational Data")
plt.xlabel("Baryonic Acceleration $g_{bar}$ (m/s$^2$)", fontsize=14)
plt.ylabel("Observed Acceleration $g_{obs}$ (m/s$^2$)", fontsize=14)
plt.title("Radial Acceleration Relation (RAR)", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="--")
plt.show()
