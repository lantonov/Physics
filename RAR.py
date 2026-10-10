import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ----------------------------------------------------------------------------
# Step 1. Read the Fixed-Width RAR Data
# ----------------------------------------------------------------------------
# According to the file description:
#   Bytes 1-6: log10(g_bar) [m/s^2]
#   Bytes 8-11: error on log10(g_bar)
#   Bytes 13-18: log10(g_obs) [m/s^2]
#   Bytes 20-23: error on log10(g_obs)
# (Indices here are zero-indexed and end-exclusive.)
colspecs = [(0, 6), (7, 11), (12, 18), (19, 23)]
col_names = ['log_gbar', 'err_log_gbar', 'log_gobs', 'err_log_gobs']

try:
    rar_data = pd.read_fwf("RAR.txt", colspecs=colspecs, names=col_names, skiprows=1)
except Exception as e:
    print("Error reading RAR.txt:", e)
    exit()

# Drop rows with missing or non-numeric entries
rar_data = rar_data.dropna(subset=['log_gbar', 'log_gobs'])
rar_data.reset_index(drop=True, inplace=True)

# Convert to numeric (in case some entries are read as strings)
rar_data['log_gbar'] = pd.to_numeric(rar_data['log_gbar'], errors='coerce')
rar_data['log_gobs'] = pd.to_numeric(rar_data['log_gobs'], errors='coerce')
rar_data['err_log_gbar'] = pd.to_numeric(rar_data['err_log_gbar'], errors='coerce')
rar_data['err_log_gobs'] = pd.to_numeric(rar_data['err_log_gobs'], errors='coerce')

# ----------------------------------------------------------------------------
# Step 2. Convert Log Accelerations to Linear Accelerations
# ----------------------------------------------------------------------------
# The data give log10 of acceleration in m/s².
# Compute g_bar = 10^(log_gbar) and g_obs = 10^(log_gobs)
rar_data['g_bar'] = 10 ** rar_data['log_gbar']  # in m/s²
rar_data['g_obs'] = 10 ** rar_data['log_gobs']  # in m/s²

# ----------------------------------------------------------------------------
# Step 3. Define the Analytical Model for the RAR
# ----------------------------------------------------------------------------
# A commonly used interpolation is:
#   g_obs = g_bar / (1 - exp(-sqrt(g_bar / a0)))
# where a0 is the characteristic acceleration.
def model_g_obs(g_bar, a0):
    # Ensure no division by zero: if g_bar is very small, this returns g_bar (limit → Newtonian)
    return g_bar / (1 - np.exp(-np.sqrt(g_bar / a0)))

# Here we initially set a0 = 1.2e-10 m/s².
a0_initial = 1.2e-10

# Compute the predicted g_obs from the model for each g_bar value.
rar_data['g_obs_pred'] = model_g_obs(rar_data['g_bar'], a0_initial)

# ----------------------------------------------------------------------------
# Step 4. Plot the RAR Data and Model Prediction
# ----------------------------------------------------------------------------
plt.figure(figsize=(8,6))
plt.loglog(rar_data['g_bar'], rar_data['g_obs'], 'bo', markersize=4, alpha=0.6, label='Observed SPARC Data')
# For a smooth curve, generate a range of g_bar values.
g_bar_range = np.logspace(np.log10(rar_data['g_bar'].min()), np.log10(rar_data['g_bar'].max()), 200)
g_obs_model = model_g_obs(g_bar_range, a0_initial)
plt.loglog(g_bar_range, g_obs_model, 'r-', lw=2, label=r'Model: $g_{obs}=g_{bar}/[1-e^{-\sqrt{g_{bar}/a_0}}]$')
plt.xlabel(r'$g_{bar}$ (m/s$^2$)', fontsize=14)
plt.ylabel(r'$g_{obs}$ (m/s$^2$)', fontsize=14)
plt.title('Radial Acceleration Relation (RAR)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which='both', ls='--')
plt.show()

# ----------------------------------------------------------------------------
# Step 5. Fit the Model Parameter (a0) to the RAR Data
# ----------------------------------------------------------------------------
# We fit the model function to the data. Because the data scatter is in log10 space,
# one option is to perform the fit in linear space using the uncertainties if needed.
# For simplicity, we fit without weighting here.
# We'll use only the points where g_bar > some small threshold to avoid numerical issues.
mask = rar_data['g_bar'] > 1e-12
popt, pcov = curve_fit(model_g_obs, rar_data['g_bar'][mask], rar_data['g_obs'][mask],
                        p0=[a0_initial])
a0_fit = popt[0]
a0_fit_err = np.sqrt(np.diag(pcov))[0]
print(f"Fitted a0 = {a0_fit:.2e} ± {a0_fit_err:.2e} m/s² (Model expects ~1.2e-10 m/s²)")

# For display, compute the model using the fitted a0.
g_obs_model_fit = model_g_obs(g_bar_range, a0_fit)

plt.figure(figsize=(8,6))
plt.loglog(rar_data['g_bar'], rar_data['g_obs'], 'bo', markersize=4, alpha=0.6, label='Observed SPARC Data')
plt.loglog(g_bar_range, g_obs_model_fit, 'g--', lw=2, label=r'Fitted Model: $a_0$ = %.2e m/s$^2$' % a0_fit)
plt.xlabel(r'$g_{bar}$ (m/s$^2$)', fontsize=14)
plt.ylabel(r'$g_{obs}$ (m/s$^2$)', fontsize=14)
plt.title('Radial Acceleration Relation (RAR)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which='both', ls='--')
plt.show()
