import numpy as np
import matplotlib.pyplot as plt

# Parameters
r1, r2 = 1.0, 0.4       # radii (toroidal decomposition)
w1, w2 = 1.0, 1.0       # angular frequencies
tau = np.linspace(-20, 20, 4000)

# Toroidal helix (Hestenes rotor orbit)
x = (r1*np.cos(w1*tau) + r2*np.cos(w2*tau))
y = (r1*np.sin(w1*tau) + r2*np.sin(w2*tau))

# Polar angle = fiber parameter
psi = np.arctan2(y, x)
r = np.sqrt(x**2 + y**2)

# Analytic Poinsot spirals
k = 0.3
psi_grid = np.linspace(-10, 10, 2000)
r_sech = 1/np.cosh(k*psi_grid)
r_csch = 1/np.sinh(k*psi_grid)
r_csch_mask = np.ma.masked_where(np.abs(psi_grid)<0.05, r_csch)

# Plot
fig, ax = plt.subplots(subplot_kw={'projection':'polar'}, figsize=(8,8))
ax.plot(psi, r, '.', ms=0.5, alpha=0.5, label="Hestenes rotor orbit (proj.)")
ax.plot(psi_grid, r_sech, 'navy', lw=2, label="sech spiral")
ax.plot(psi_grid, r_csch_mask, 'darkorange', lw=2, label="csch spiral")
ax.set_title("Hestenes rotor → Villarceau circle → Poinsot spiral")
ax.legend(loc='upper right')
plt.show()