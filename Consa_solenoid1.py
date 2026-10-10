

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
