import numpy as np
import matplotlib.pyplot as plt

# --------- Medium: Gaussian bump in n(x,z) ----------
n0 = 1.0
eps = 0.06         # index contrast
sigma = 0.02       # 2 cm correlation length

def n_and_grads(x, z):
    r2 = x*x + z*z
    bump = np.exp(-r2/(sigma*sigma))
    n = n0 + eps*bump
    # grads of ln n
    nx = eps * (-2*x/(sigma*sigma)) * bump
    nz = eps * (-2*z/(sigma*sigma)) * bump
    ln_n_x = nx / n
    ln_n_z = nz / n
    # Hessian entries of ln n
    # Compute second derivatives: d/dx (nx/n) etc.
    # First, helpers:
    nxx = eps * ((4*x*x/(sigma**4)) - (2/(sigma**2))) * bump
    nzz = eps * ((4*z*z/(sigma**4)) - (2/(sigma**2))) * bump
    nxz = eps * (4*x*z/(sigma**4)) * bump
    ln_n_xx = (nxx*n - nx*nx)/ (n*n)
    ln_n_zz = (nzz*n - nz*nz)/ (n*n)
    ln_n_xz = (nxz*n - nx*nz)/ (n*n)
    return n, ln_n_x, ln_n_z, ln_n_xx, ln_n_xz, ln_n_zz

# --------- Ray + variational ODE (RK4 for clarity) ----------
def rhs(y):
    x, z, px, pz, dxdu, dzdu, dpxdu, dpzdu = y
    n, ln_x, ln_z, ln_xx, ln_xz, ln_zz = n_and_grads(x, z)
    n2_inv = 1.0/(n*n)
    # Ray equations
    xd = n2_inv * px
    zd = n2_inv * pz
    pxd = ln_x
    pzd = ln_z
    # Variational equations (linearized)
    dxdud = n2_inv * dpxdu
    dzdud = n2_inv * dpzdu
    dpxdud = ln_xx * dxdu + ln_xz * dzdu
    dpzdud = ln_xz * dxdu + ln_zz * dzdu
    return np.array([xd, zd, pxd, pzd, dxdud, dzdud, dpxdud, dpzdud])

def rk4_step(y, h):
    k1 = rhs(y)
    k2 = rhs(y + 0.5*h*k1)
    k3 = rhs(y + 0.5*h*k2)
    k4 = rhs(y + h*k3)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# --------- Initialize Λ0 on z=0 and propagate ----------
L = 0.25            # propagation distance (m)
Ns = 2000
ds = L / Ns
umin, umax = -0.06, 0.06
Nu = 401
U = np.linspace(umin, umax, Nu)

# Storage
X = np.zeros((Nu, Ns+1))
Z = np.zeros((Nu, Ns+1))
dXdu = np.zeros((Nu, Ns+1))

for i, u in enumerate(U):
    # initial data on z=0
    n00, ln_x0, ln_z0, *_ = n_and_grads(u, 0.0)
    x0, z0 = u, 0.0
    px0, pz0 = 0.0, n00            # H=1/2
    dxdu0, dzdu0 = 1.0, 0.0
    dpxdu0, dpzdu0 = 0.0, (eps*(-2*u/(sigma**2))*np.exp(-u*u/(sigma*sigma)))/n00  # ∂x n / n at z=0

    y = np.array([x0, z0, px0, pz0, dxdu0, dzdu0, dpxdu0, dpzdu0])

    X[i,0], Z[i,0], dXdu[i,0] = x0, z0, dxdu0
    for k in range(1, Ns+1):
        y = rk4_step(y, ds)
        X[i,k], Z[i,k], dXdu[i,k] = y[0], y[1], y[4]

# --------- Caustic detection and intensity ----------
# Fold: ∂x/∂u = 0 -> detect sign changes across u for each s
caustic_points = []
for k in range(1, Ns+1):
    sgn = np.sign(dXdu[:,k])
    flips = np.where(sgn[:-1]*sgn[1:] < 0)[0]
    for j in flips:
        # interpolate between U[j], U[j+1]
        a = abs(dXdu[j,k]); b = abs(dXdu[j+1,k])
        w = a/(a+b + 1e-12)
        x_c = (1-w)*X[j,k] + w*X[j+1,k]
        z_c = (1-w)*Z[j,k] + w*Z[j+1,k]
        caustic_points.append((x_c, z_c))

# Intensity proxy ~ 1/|∂x/∂u|
I = 1.0/(np.abs(dXdu) + 1e-6)
I = np.clip(I, 0, np.percentile(I, 99))

# --------- Plots ----------
fig, ax = plt.subplots(1, 2, figsize=(12,5), sharey=True)

# Rays
for i in range(0, Nu, 6):
    ax[0].plot(Z[i]*1e2, X[i]*1e3, color='steelblue', lw=0.6, alpha=0.7)
ax[0].set_xlabel('z (cm)'); ax[0].set_ylabel('x (mm)')
ax[0].set_title('Baseline rays from Λ0 (plane wave, z=0)')
ax[0].grid(True)

# Intensity map + caustic overlay
Zg, Ug = np.meshgrid(np.linspace(0, L, Ns+1), U)
pc = ax[1].pcolormesh(Zg*1e2, X*1e3, I, shading='auto', cmap='inferno')
cb = plt.colorbar(pc, ax=ax[1], label='~ 1/|∂x/∂u| (clipped)')
if caustic_points:
    C = np.array(caustic_points)
    ax[1].scatter(C[:,1]*1e2, C[:,0]*1e3, s=6, c='cyan', label='caustic (fold)')
    ax[1].legend(loc='upper right')
ax[1].set_xlabel('z (cm)'); ax[1].set_title('Intensity and fold caustic')
ax[1].grid(True)

plt.tight_layout()
plt.show()
