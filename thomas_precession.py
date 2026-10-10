import numpy as np 
import math
import matplotlib.pyplot as plt # Constants c = 3e8 # Speed of light in m/s a = 1e10 # Constant acceleration in m/s^2 # Velocity range from 0 to 0.99c v = np.linspace(0, 0.99 * c, 1000) # Lorentz factor gamma = 1 / np.sqrt(1 - (v / c)**2) # Thomas precession frequency omega_T = (gamma**2 / (gamma + 1)) * (a * v / c**2) # Plotting plt.figure(figsize=(10, 6)) plt.plot(v / c, omega_T, label=r'$\omega_T$ vs $v$') plt.xlabel(r'Velocity $v$ (as a fraction of $c$)') plt.ylabel(r'Thomas Precession Frequency $\omega_T$') plt.title(r'$\omega_T$ vs $v$') plt.legend() plt.grid(True) plt.show()import numpy as np
import matplotlib.pyplot as plt

# Constants
c = 3e8  # Speed of light in m/s
a = (math.pi)*10**7  # Constant acceleration in m/s^2

# Velocity range from 0 to 0.99c
v = np.linspace(0, 0.99 * c, 1000)

# Lorentz factor
gamma = 1 / np.sqrt(1 - (v / c)**2)

# Thomas precession frequency
omega_T = (gamma**2 / (gamma + 1)) * (a * v / c**2)
t = v*(1 + (v / c)**2) / (2 + (v / c)**2) / c
factor = (gamma**2 / (gamma + 1)) * v
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(v / c, t, label=r'$t$ vs $v$')
#plt.plot(v / c, factor, label=r'$factor$ vs $v$')
plt.plot(v / c, omega_T, label=r'$\omega_T$ vs $v$')
plt.xlabel(r'Velocity $v$ (as a fraction of $c$)')
plt.ylabel(r'Thomas Precession Frequency $\omega_T$')
plt.title(r'$t$ and $\omega_T$ vs $v$')
plt.legend()
plt.grid(True)
plt.show()
