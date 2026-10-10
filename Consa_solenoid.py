import numpy as np
import matplotlib.pyplot as plt

# Parameters
R  = 1.0     # major radius
r  = 0.25    # minor radius
N  = 3       # poloidal windings per toroidal turn
ω  = 1.0     # base frequency
v  = 0.0     # drift (set nonzero to see axial drift)
T  = 40*np.pi
t  = np.linspace(0, T, 5000)

# Helical solenoid (Consa parametrization)
x = (R + r*np.cos(N*ω*t)) * np.cos(ω*t)
y = (R + r*np.cos(N*ω*t)) * np.sin(ω*t)
z = r*np.sin(N*ω*t) + v*t

# Torus phase coordinates (u: toroidal, vph: poloidal)
u   = ω*t
vph = N*ω*t

# Villarceau-like direction (constant slope on torus)
# On the torus parameter space (u,vph), Villarecau circles are oblique lines.
# We'll show the (x,y) projection colored by vph to emphasize oblique wrapping.
fig = plt.figure(figsize=(14,6))
ax1 = fig.add_subplot(1,2,1)
sc = ax1.scatter(x, y, c=np.mod(vph, 2*np.pi), s=0.3, cmap='twilight')
ax1.set_aspect('equal')
ax1.set_title("Helical solenoid projected to (x,y): oblique (Villarceau-like) wraps")
plt.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04, label='poloidal phase v')

# Polar Poinsot spirals using fiber angle as polar angle
psi = np.linspace(-12, 12, 4000)
k   = 0.35
r_sech = 1/np.cosh(k*psi)
r_csch = 1/np.sinh(k*psi)
r_csch_mask = np.ma.masked_where(np.abs(psi)<0.02, r_csch)

ax2 = fig.add_subplot(1,2,2, projection='polar')
ax2.plot(psi, r_sech, color='navy', label='sech spiral (bounded)')
ax2.plot(psi, r_csch_mask, color='darkorange', label='csch spiral (open)')
ax2.set_title("Poinsot spirals in polar coordinates")
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()


# Physical scales (set to 1 in geom. units; scale later if desired)
e = 1.0
m = 1.0
hbar = 1.0

# Estimate a mean period for the toroidal circulation
# For ω=1, toroidal period ≈ 2π; current I = e f ≈ e / (2π)
I = e/(2*np.pi)

# Geometric areas
A_loop = np.pi*R**2
mu_geom = I * A_loop * (1 + 0.5*(r/R)**2)  # Consa's correction

# Bohr magneton in these units μ_B = e*hbar/(2m) → 0.5 if e=m=hbar=1
mu_B = e*hbar/(2*m)

print("I ≈", I)
print("μ_geom / μ_B ≈", mu_geom / mu_B)
print("Expected g_helical ≈ sqrt(1 + (r*N/R)^2) =", np.sqrt(1 + (r*N/R)**2))
