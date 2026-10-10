import numpy as np
import matplotlib.pyplot as plt

# ---------------- Medium: Gaussian index bump ----------------
n0 = 1.0
eps = 0.06
sigma = 0.02
lam = 532e-9   # 532 nm (green)
k0 = 2*np.pi/lam

def n_and_grads(x, z):
    r2 = x*x + z*z
    bump = np.exp(-r2/(sigma*sigma))
    n = n0 + eps*bump
    # grads and Hessian of ln n
    nx = eps * (-2*x/(sigma**2)) * bump
    nz = eps * (-2*z/(sigma**2)) * bump
    ln_x = nx / n
    ln_z = nz / n
    nxx = eps * ((4*x*x/(sigma**4)) - (2/(sigma**2))) * bump
    nzz = eps * ((4*z*z/(sigma**4)) - (2/(sigma**2))) * bump
    nxz = eps * (4*x*z/(sigma**4)) * bump
    ln_xx = (nxx*n - nx*nx) / (n*n)
    ln_zz = (nzz*n - nz*nz) / (n*n)
    ln_xz = (nxz*n - nx*nz) / (n*n)
    return n, ln_x, ln_z, ln_xx, ln_xz, ln_zz

# ---------------- Hamiltonian + variational + action ----------------
# State y = [x, z, px, pz,  v1_x, v1_z, v1_px, v1_pz,  v2_x, v2_z, v2_px, v2_pz,  S]
def rhs(y):
    x, z, px, pz = y[0:4]
    v1 = y[4:8]; v2 = y[8:12]
    S = y[12]
    n, ln_x, ln_z, ln_xx, ln_xz, ln_zz = n_and_grads(x, z)
    n2_inv = 1.0/(n*n)
    # rays
    xd = n2_inv * px
    zd = n2_inv * pz
    pxd = ln_x
    pzd = ln_z
    # variational matrix A (linearized flow)
    # order: (dx, dz, dpx, dpz)
    A = np.array([[0, 0, n2_inv, 0],
                  [0, 0, 0,      n2_inv],
                  [ln_xx, ln_xz, 0,      0],
                  [ln_xz, ln_zz, 0,      0]], dtype=float)
    v1d = A @ v1
    v2d = A @ v2
    # action S' = k0 * n * dl/ds; with |q'| = |(xd,zd)| = |p|/n^2; so n*|q'| = |p|/n
    speed = np.sqrt(xd*xd + zd*zd)
    Sd = k0 * n * speed
    return np.array([xd,zd,pxd,pzd, *v1d, *v2d, Sd])

def rk4_step(y, h):
    k1 = rhs(y)
    k2 = rhs(y + 0.5*h*k1)
    k3 = rhs(y + 0.5*h*k2)
    k4 = rhs(y + h*k3)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# ---------------- Propagate central ray + monodromy to screen z=z_s -------
z_s = 0.20     # 20 cm screen
L   = z_s
Ns  = 4000
ds  = L / Ns

# Initial conditions: plane wave at z=0, central ray x0=0
x0, z0 = 0.0, 0.0
n00, *_ = n_and_grads(x0, z0)
px0, pz0 = 0.0, n00
# variational seeds: column1 position, column2 momentum
v1_0 = np.array([1.0, 0.0, 0.0, 0.0])
v2_0 = np.array([0.0, 0.0, 1.0, 0.0])
S0 = 0.0

y = np.array([x0,z0,px0,pz0, *v1_0, *v2_0, S0])
A_hist=[]; zeros_A=0; prevA=1.0

for k in range(Ns):
    y = rk4_step(y, ds)
    # count Maslov crossings when A passes through zero
    A = y[4]   # v1_x = ∂x/∂x0 = A block (1,1)
    A_hist.append(A)
    if prevA * A < 0:
        zeros_A += 1
    prevA = A
    if y[1] >= z_s:  # z reached screen; linear interp for accuracy
        break

x, z, px, pz = y[0:4]
A = y[4]; B = y[8]      # v2_x = ∂x/∂px0
C = y[6]; D = y[10]     # v1_px, v2_px
S = y[12]
mu = zeros_A            # Maslov index up to first crossing

# ---------------- Gaussian beam prediction at screen ----------------
# Use paraxial (x, θ) ABCD: px ≈ n θ, so B, C rescale: Bθ = B / n00, Cθ = n00 * C
Aθ, Bθ, Cθ, Dθ = A, B/n00, n00*C, D

w0 = 0.5e-3    # 0.5 mm initial beam waist at z=0
R0 = np.inf    # planar wavefront at source
# q parameter: 1/q = 1/R - i λ/(π w^2)
q0 = 1.0 / (1.0/R0 - 1j * lam/(np.pi*w0*w0))  # ≈ i z_R
q  = (Aθ*q0 + Bθ) / (Cθ*q0 + Dθ)

# Extract width and radius at screen
inv_q = 1.0/q
R = 1.0 / np.real(inv_q) if np.real(inv_q)!=0 else np.inf
w = np.sqrt( - lam / (np.pi * np.imag(inv_q)) )

# On-screen transverse coordinate and Gaussian intensity
x_axis = np.linspace(-3*w, 3*w, 801)
# Geometric center at the central ray x
I_gb = (1.0 / np.abs(Cθ*q0 + Dθ)) * np.exp( -2.0 * ((x_axis - x)**2) / (w*w) )
# Phase (up to global): S + Gouy = S + (μ + arg(Cθ q0 + Dθ)/π)*π/2
phi_gouy = np.angle(Cθ*q0 + Dθ) / 2.0 + (mu * np.pi/2.0)

# ---------------- Uniform Airy check near a fold on the screen -------------
# Find a fold on z=z_s by tracing a small fan and locating ∂x/∂u=0
umin, umax = -0.01, 0.01
Nu = 401
U = np.linspace(umin, umax, Nu)

def propagate_to_screen(u):
    # one ray from x0=u on z=0, zero slope
    n00, *_ = n_and_grads(u, 0.0)
    y = np.array([u, 0.0, 0.0, n00,  1.0,0,0,0,  0,0,0,0,  0.0])
    z = 0.0
    while z < z_s:
        y = rk4_step(y, ds)
        z = y[1]
    return y  # contains v1_x = ∂x/∂u as "A(u)"

Xs = np.zeros(Nu)
As = np.zeros(Nu)
for i,u in enumerate(U):
    y_u = propagate_to_screen(u)
    Xs[i] = y_u[0]      # x(u; z=z_s)
    As[i] = y_u[4]      # ∂x/∂u

# locate fold u_c: zero of As with sign change
idx = np.where(As[:-1]*As[1:] < 0)[0]
airy_curve = None
if idx.size>0:
    j = idx[0]
    # interpolate u_c and x_c
    wlin = abs(As[j]) / (abs(As[j]) + abs(As[j+1]) + 1e-16)
    u_c = (1-wlin)*U[j] + wlin*U[j+1]
    x_c = (1-wlin)*Xs[j] + wlin*Xs[j+1]
    # local curvature a = x_uu at u_c
    # finite differences around j
    u3 = U[j-1:j+2]
    x3 = Xs[j-1:j+2]
    a = np.gradient(np.gradient(x3, u3), u3)[1]
    # Airy scaling (practical proxy):
    L_ai = (2.0/(k0*abs(a)))**(1.0/3.0)    # length scale
    xi = (x_axis - x_c)/L_ai
    # normalized Airy intensity ~ Ai(xi)^2 (up to a scale)
    try:
        from mpmath import airy
        Ai = np.array([airy(xx)[0] for xx in xi], dtype=float)
        I_ai = (Ai**2) / np.max(Ai**2)
        airy_curve = (x_c + xi*L_ai, I_ai)
    except Exception:
        airy_curve = None

# ---------------- Plots ----------------
fig, ax = plt.subplots(1, 2, figsize=(12,4.5))

# Left: central ray and Gaussian beam width
ax[0].plot([0, z_s*1e2], [0, x*1e3], 'k-', lw=1.2, label='central ray')
dz_vis = 0.2 # cm
ax[0].fill_betweenx([(-3*w)*1e3, (3*w)*1e3], (z_s*1e2 - dz_vis), (z_s*1e2 + dz_vis), color='crimson', alpha=0.15, label='GB width at screen')
ax[0].axvline(z_s*1e2, color='gray', ls='--', lw=1.0)
ax[0].set_xlabel('z (cm)'); ax[0].set_ylabel('x (mm)')
ax[0].set_title('Central ray and Gaussian width at screen')
ax[0].legend(); ax[0].grid(True)

# Right: on-screen intensity profiles
ax[1].plot((x_axis)*1e3, I_gb/np.max(I_gb), color='crimson', lw=1.8, label='Gaussian beam (ABCD)')
if airy_curve is not None:
    ax[1].plot((airy_curve[0])*1e3, airy_curve[1], color='navy', lw=1.4, label='Uniform Airy (fold)')
ax[1].set_xlabel('x at screen (mm)'); ax[1].set_ylabel('normalized intensity')
ax[1].set_title('Wave-optics checks at z = %.1f cm' % (z_s*1e2))
ax[1].grid(True); ax[1].legend(loc='upper right')

plt.tight_layout()
plt.show()

print(f"Monodromy blocks at screen (x, θ): A={Aθ:.4g}, B={Bθ:.4g} m, C={Cθ:.4g} 1/m, D={Dθ:.4g}")
print(f"Gaussian beam width at screen w = {w*1e3:.3f} mm, R = {R:.3g} m")
print(f"Maslov index μ along central ray to screen: {mu} (phase jump {mu}·π/2)")
