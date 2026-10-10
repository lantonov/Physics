import numpy as np
import matplotlib.pyplot as plt

# ---------------- Parameters (normalized units) ----------------
n0 = 1.0               # base refractive index (drops out in normalized paraxial form)
g0 = 40.0              # base gradient [1/length]
eps = 0.10             # modulation amplitude (10%)
Omega = 2*np.pi/20.0   # modulation angular frequency (slow compared to 1/L)

L = 0.08               # rod length (e.g., 8 cm)
Nz = 4000              # axial steps
dz = L / Nz

# time coupling along each ray: t(z) = t0 + z (normalized)
def g_of_t(t):
    return g0 * (1.0 + eps*np.sin(Omega * t))

# ---------------- Integrate fundamental matrix columns ----------------
def focus_position_vs_t0(t0, zmax=L, dz=dz):
    """
    Return (z_f, found) where z_f is the first axial position where m11 crosses zero
    for initial time t0. If no zero is found up to zmax, found=False.
    """
    # First column: IC r=1, u=0 => m11, m21
    r1, u1 = 1.0, 0.0
    zf = None
    r1_prev = r1
    z_prev = 0.0

    # Stormer-Verlet (time-dependent frequency) for stability
    z = 0.0
    while z < zmax:
        t = t0 + z
        w2 = g_of_t(t)**2
        # half-step for slope
        u1_half = u1 - 0.5*dz * (w2 * r1)
        # full step for radius
        r1_new = r1 + dz * u1_half
        # update frequency at end (optional midpoint; here evaluate again at z+dz)
        t_end = t0 + (z + dz)
        w2_end = g_of_t(t_end)**2
        # complete slope step
        u1_new = u1_half - 0.5*dz * (w2_end * r1_new)

        # zero crossing detection for m11 = r1
        if r1_prev * r1_new < 0.0 and zf is None:
            # linear interpolate the zero
            alpha = abs(r1_prev) / (abs(r1_prev) + abs(r1_new))
            zf = z_prev + alpha * dz

        # advance
        r1_prev = r1_new
        z_prev = z + dz
        r1, u1 = r1_new, u1_new
        z += dz

    if zf is None:
        return (np.nan, False)
    return (zf, True)

# ---------------- Sweep initial time and collect focus positions --------
T = 2*np.pi / Omega
Nt = 120
t0_grid = np.linspace(0.0, T, Nt, endpoint=False)

z_focus = np.full(Nt, np.nan)
ok = np.zeros(Nt, dtype=bool)

for k, t0 in enumerate(t0_grid):
    zf, found = focus_position_vs_t0(t0)
    z_focus[k] = zf
    ok[k] = found

# ---------------- Plot: breathing of the axial focus --------------------
fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
ax.plot(t0_grid[ok], z_focus[ok]*1e2, 'o-', lw=1.5, ms=4, color='crimson')
ax.set_xlabel('modulation phase time t0 (arb. units)')
ax.set_ylabel('first focus position z_f (cm)')
ax.set_title('Breathing caustic: axial focus vs. modulation phase')
ax.grid(True)
plt.tight_layout()
plt.show()

# ---------------- Optional: ray bundle snapshots at two phases ----------
def trace_bundle(t0, zmax=L, dz=dz, r0_span=0.5e-3, Nr=51):
    # small symmetric initial bundle with zero slope
    r0s = np.linspace(-r0_span, r0_span, Nr)
    rs = np.zeros((Nr, int(zmax/dz)+1))
    zs = np.linspace(0, zmax, rs.shape[1])

    # initialize
    r = r0s.copy()
    u = np.zeros_like(r)

    idx = 0
    rs[:, idx] = r
    for z in zs[1:]:
        t = t0 + (z - dz)
        w2 = g_of_t(t)**2
        u_half = u - 0.5*dz * (w2 * r)
        r_new = r + dz * u_half
        w2_end = g_of_t(t + dz)**2
        u_new = u_half - 0.5*dz * (w2_end * r_new)
        r, u = r_new, u_new
        idx += 1
        rs[:, idx] = r
    return zs, rs

# pick two opposite phases
tA, tB = 0.0, 0.5*T
zs, rsA = trace_bundle(tA)
_,  rsB = trace_bundle(tB)

fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4))
for k in (0, 1):
    ax = ax2[k]
    rs = rsA if k==0 else rsB
    for i in range(rs.shape[0]):
        ax.plot(zs*1e2, rs[i]*1e3, color='navy', lw=0.6, alpha=0.6)
    ax.set_xlabel('z (cm)')
    ax.set_ylabel('r (mm)')
    phase = 't0 = 0' if k==0 else 't0 = T/2'
    ax.set_title(f'Ray bundle at {phase}')
    ax.grid(True)
plt.tight_layout()
plt.show()
