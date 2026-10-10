import numpy as np
import matplotlib.pyplot as plt

# ---------------- Parameters tuned for visibility ----------------
g0 = 60.0               # base gradient [1/m]
eps = 0.25              # 25% modulation for clarity
T = 1.0                 # modulation period (arb. units)
Omega = 2*np.pi / T

# Choose length near the first static focus z_f0 = pi/(2 g0)
z_f0 = np.pi / (2*g0)   # ≈ 0.0262 m
L = 0.035               # a bit longer to show the waist
Nz = 5000
dz = L / Nz

# Time along the rod: t(z) = t0 + z (normalized units)
def g_of_t(t):
    return g0 * (1.0 + eps*np.sin(Omega * t))

# Integrate the first column of the fundamental matrix (r,u) with IC r=1,u=0
def first_zero_of_m11(t0, zmax=L, dz=dz):
    r, u = 1.0, 0.0
    r_prev = r
    z_prev = 0.0
    z = 0.0
    zf = None
    while z < zmax:
        t = t0 + z
        w2 = g_of_t(t)**2
        # Stormer-Verlet (time-dependent omega)
        u_half = u - 0.5*dz*(w2*r)
        r_new = r + dz*u_half
        w2_end = g_of_t(t + dz)**2
        u_new = u_half - 0.5*dz*(w2_end*r_new)
        if r_prev * r_new < 0 and zf is None:
            alpha = abs(r_prev) / (abs(r_prev) + abs(r_new))
            zf = z_prev + alpha*dz
        r_prev, z_prev = r_new, z + dz
        r, u = r_new, u_new
        z += dz
    return zf

# Trace a symmetric paraxial bundle with zero initial slope
def trace_bundle(t0, r0_span=0.5e-3, Nr=61, zmax=L, dz=dz):
    r0s = np.linspace(-r0_span, r0_span, Nr)
    zs = np.linspace(0, zmax, int(zmax/dz)+1)
    R = np.zeros((Nr, zs.size))
    r = r0s.copy()
    u = np.zeros_like(r)
    R[:,0] = r
    for k in range(1, zs.size):
        z = zs[k-1]
        t = t0 + z
        w2 = g_of_t(t)**2
        u_half = u - 0.5*dz*(w2*r)
        r_new = r + dz*u_half
        w2_end = g_of_t(t + dz)**2
        u_new = u_half - 0.5*dz*(w2_end*r_new)
        r, u = r_new, u_new
        R[:,k] = r
    return zs, R

# Pick phases that maximize contrast
tA = 0.25*T   # sin = +1 at z=0 ⇒ g ≈ g0(1+eps)
tB = 0.75*T   # sin = -1 at z=0 ⇒ g ≈ g0(1-eps)

zA = first_zero_of_m11(tA)
zB = first_zero_of_m11(tB)

print(f"Static focus z_f0 = {z_f0*1e3:.2f} mm")
print(f"Focus at t0=T/4   z_f = {zA*1e3:.2f} mm")
print(f"Focus at t0=3T/4  z_f = {zB*1e3:.2f} mm")
print(f"Shift Δz ≈ {(zA - zB)*1e3:.2f} mm")

# Plot two bundles side-by-side with focus markers
zs, RA = trace_bundle(tA)
_,  RB = trace_bundle(tB)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, R, title, zf in [(axes[0], RA, 't0 = T/4 (g larger)', zA),
                         (axes[1], RB, 't0 = 3T/4 (g smaller)', zB)]:
    for i in range(R.shape[0]):
        ax.plot(zs*1e3, R[i]*1e3, color='navy', lw=0.6, alpha=0.6)
    if zf is not None:
        ax.axvline(zf*1e3, color='crimson', ls='--', lw=1.5, label='first focus')
    ax.axvline(z_f0*1e3, color='gray', ls=':', lw=1.0, label='static focus z_f0')
    ax.set_xlabel('z (mm)')
    ax.set_title(title)
    ax.grid(True)
axes[0].set_ylabel('r (mm)')
axes[0].legend(loc='upper right')
plt.suptitle('Breathing caustic in a time‑modulated GRIN lens')
plt.tight_layout()
plt.show()
