import numpy as np
import matplotlib.pyplot as plt

# Define fundamental constants in SI units
hbar = 1.054571817e-34   # J*s
c = 3.0e8                # m/s
G = 6.67430e-11          # m^3/(kg*s^2)

# Compute the threshold mass m such that Schwarzschild radius equals reduced Compton wavelength.
m_threshold = np.sqrt(hbar * c / (2 * G))
print("Threshold mass m =", m_threshold, "kg")

# Compute Schwarzschild radius r_s = 2Gm/c^2 for this mass
r_s = 2 * G * m_threshold / c**2
print("Schwarzschild radius r_s =", r_s, "m")

# Compute reduced Compton wavelength lambda_C = hbar/(m*c)
lambda_C = hbar / (m_threshold * c)
print("Reduced Compton wavelength lambda_C =", lambda_C, "m")

# They should be equal:
print("r_s / lambda_C =", r_s / lambda_C)

# Compute Planck length l_p = sqrt(hbar*G/c^3)
l_p = np.sqrt(hbar * G / c**3)
print("Planck length l_p =", l_p, "m")

# Show the simple relation: r_s = lambda_C = sqrt(2)*l_p
print("sqrt(2)*l_p =", np.sqrt(2) * l_p, "m")

# Now compute the threshold energy E = m*c^2
E_thresh = m_threshold * c**2
print("Threshold energy E =", E_thresh, "J")

# Compute a characteristic time scale t_char = hbar/E
t_char = hbar / E_thresh
print("Characteristic time t_char =", t_char, "s")

# Verification: the Planck time is t_p = sqrt(hbar*G/c^5)
t_p = np.sqrt(hbar * G / c**5)
print("Planck time t_p =", t_p, "s")

# Plot the scaling: let's compare r_s, lambda_C, and sqrt(2)*l_p
labels = ['Schwarzschild radius r_s', 'Reduced Compton wavelength lambda_C', 'sqrt(2) * l_p']
values = [r_s, lambda_C, np.sqrt(2) * l_p]

plt.figure(figsize=(8,6))
plt.bar(labels, values, color=['blue','green','red'])
plt.ylabel("Length (m)", fontsize=14)
plt.title("Comparison of Fundamental Length Scales", fontsize=16)
plt.show()

# Plot the relative time scales
print("\nRelative time scales:")
print("t_char =", t_char, "s")
print("Planck time t_p =", t_p, "s")
