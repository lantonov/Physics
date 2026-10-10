import numpy as np # Import numpy
import matplotlib.pyplot as plt # Import matplotlib for plotting

# Time array (short segment for clarity)
t = np.linspace(0, 0.1, 2000)  # 0.1s, 2000 points

# Initialize full waveforms
h_GR_full = np.zeros_like(t)
h_Theta_full = np.zeros_like(t)

# Assuming n_values, l_values, f_nl, f_nl_pn are defined elsewhere or need to be defined
# For demonstration purposes, I'll define dummy values if they are not present.
# If these variables are meant to be defined in previous cells, please ensure they are executed.

# Dummy values for n_values, l_values, f_nl, f_nl_pn to allow execution
# You should replace these with your actual data if they are not from previous cells.
if 'n_values' not in locals() and 'n_values' not in globals():
    n_values = np.array([1, 2, 3])
if 'l_values' not in locals() and 'l_values' not in globals():
    l_values = np.array([2, 3, 4])
if 'f_nl' not in locals() and 'f_nl' not in globals():
    f_nl = np.random.rand(len(n_values), len(l_values)) * 100 # Example frequencies
if 'f_nl_pn' not in locals() and 'f_nl_pn' not in globals():
    f_nl_pn = np.random.rand(len(n_values), len(l_values)) * 110 # Example frequencies

import numpy as np

# Constants
c = 3e8  # speed of light in m/s
G = 6.67430e-11  # gravitational constant in m^3/kg/s^2
M_sun = 1.98847e30  # solar mass in kg

# LIGO-like BH parameters
M_BH = 30 * M_sun  # 30 solar masses
R = G*M_BH/c**2  # characteristic S^3 radius ~ Schwarzschild radius

# Tensor harmonic eigenvalues on S^3 for l=2-4, n=1-3
# Simple approximation: k_n = (l+1+n)^2 / R^2
l_values = np.array([2, 3, 4])
n_values = np.array([1, 2, 3])
k_nl = np.array([[ (l+1+n)**2 / R**2 for l in l_values] for n in n_values])

# Frequencies omega = c * sqrt(k_n)
omega_nl = c * np.sqrt(k_nl)  # rad/s
f_nl = omega_nl / (2*np.pi)  # Hz

# Post-Newtonian phase correction (first order)
# Simple toy model: Delta f/f ~ v/c ~ sqrt(R_s/R) ~ 1 for Schwarzschild radius
# Here we just give a fractional shift ~ 0.1 (for demonstration)
pn_correction_factor = 0.1
f_nl_pn = f_nl * (1 + pn_correction_factor)

# Display results as table
import pandas as pd

table = []
for i, n in enumerate(n_values):
    for j, l in enumerate(l_values):
        table.append({
            "n": n,
            "l": l,
            "k_nl": k_nl[i,j],
            "f_nl_Hz": f_nl[i,j],
            "f_nl_PN_Hz": f_nl_pn[i,j]
        })

df_qnm = pd.DataFrame(table)
df_qnm


# Sum over modes n=1-3, l=2-4 (dominant m ignored for simplicity)
for i, n in enumerate(n_values):
    for j, l in enumerate(l_values):
        f_GR = f_nl[i,j]
        f_Theta = f_nl_pn[i,j]
        # Small amplitude scaling by mode number (toy model)
        # Changed (n+j-1) to (n+l-1) to avoid division by zero (min value 1+2-1=2)
        amplitude = 1e-21 / (n+l-1)
        h_GR_full += amplitude * np.sin(2*np.pi*f_GR*t)
        h_Theta_full += amplitude * np.sin(2*np.pi*f_Theta*t)

# Plotting
plt.figure(figsize=(12,6))
plt.plot(t, h_GR_full, label='GR waveform (sum of n=1-3, l=2-4)')
plt.plot(t, h_Theta_full, '--', label='Theta+ PN-corrected (sum of modes)')
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.title('Full waveform: GR vs Theta+ with multiple modes')
plt.legend()
plt.grid(True)
plt.show()