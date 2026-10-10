import numpy as np
import matplotlib.pyplot as plt

# ---------- Curvature field K(x,y,t) and its gradient ∇K ----------
# A moving, anisotropic Gaussian bump (proxy for curvature drive)
kappa0 = 50.0       # curvature scale
sigma_x = 0.6
sigma_y = 0.4
a0 = 1.0            # horizontal excursion of the bump
Omega = 2*np.pi     # period T=1

def K_and_grad(x, y, t):
    a = a0 * np.sin(Omega * t)
    X = (x - a) / sigma_x
    Y =  y       / sigma_y
    K = kappa0 * np.exp(-(X*X + Y*Y))
    # ∇K = K * (-2 X/σx, -2 Y/σy) in physical units:
    dKdx = K * (-2.0 * X / sigma_x)
    dKdy = K * (-2.0 * Y / sigma_y)
    return K, dKdx, dKdy

# ---------- Hamiltonian transport on ray space (s, φ) ----------
def step_rayspace(s, phi, t, dt):
    # p = s n, with n=(cosφ, sinφ); t_vec = (-sinφ, cosφ)
    c, sφ = np.cos(phi), np.sin(phi)
    nvec = np.stack([c, sφ], axis=-1)
    tvec = np.stack([-sφ, c], axis=-1)
    px, py = s * nvec[...,0], s * nvec[...,1]

    # curvature gradient at closest point p
    _, dKdx, dKdy = K_and_grad(px, py, t)
    gdotn = dKdx * nvec[...,0] + dKdy * nvec[...,1]
    gdott = dKdx * tvec[...,0] + dKdy * tvec[...,1]

    # ODE: sdot = s * (∇K·t), phidot = -(∇K·n)
    sdot   = s * gdott
    phidot = - gdotn

    # RK2 (midpoint) for stability
    s_mid   = s   + 0.5*dt*sdot
    phi_mid = phi + 0.5*dt*phidot
    c2, s2 = np.cos(phi_mid), np.sin(phi_mid)
    n2 = np.stack([c2, s2], axis=-1)
    t2 = np.stack([-s2, c2], axis=-1)
    px2, py2 = s_mid * n2[...,0], s_mid * n2[...,1]
    _, dKdx2, dKdy2 = K_and_grad(px2, py2, t + 0.5*dt)
    gdotn2 = dKdx2 * n2[...,0] + dKdy2 * n2[...,1]
    gdott2 = dKdx2 * t2[...,0] + dKdy2 * t2[...,1]
    sdot2   = s_mid * gdott2
    phidot2 = - gdotn2

    s_new   = s   + dt * sdot2
    phi_new = phi + dt * phidot2
    # wrap φ
    phi_new = (phi_new + np.pi) % (2*np.pi) - np.pi
    return s_new, phi_new

# ---------- Initialize a baseline fold (circle envelope) ----------
R0 = 2.0
Nλ = 1024
lam = np.linspace(-np.pi, np.pi, Nλ, endpoint=False)
s0   = R0 * np.ones_like(lam)
phi0 = lam.copy()  # one full turn

# Transport over t in [0, t1]
t0, t1 = 0.0, 0.50  # half period
Nt = 2000
dt = (t1 - t0) / Nt
s, phi = s0.copy(), phi0.copy()
t = t0
for _ in range(Nt):
    s, phi = step_rayspace(s, phi, t, dt)
    t += dt

# ---------- Reconstruct envelope at t1 ----------
# Finite differences with circular wrap in λ (using λ as the parameter)
dphi_dlam = np.gradient(phi, lam, edge_order=2)
ds_dlam   = np.gradient(s,   lam, edge_order=2)

# Avoid division by ~0 (near parallel parameterization); add a small floor
eps = 1e-6
q = ds_dlam / (dphi_dlam + np.sign(dphi_dlam)*eps)

x_env = s*np.cos(phi) - q*np.sin(phi)
y_env = s*np.sin(phi) + q*np.cos(phi)

# Initial envelope (circle)
x0_env = R0*np.cos(phi0)
y0_env = R0*np.sin(phi0)

# ---------- Plot ----------
fig, ax = plt.subplots(1,1, figsize=(6,6))
ax.plot(x0_env, y0_env, color='gray', lw=1.0, ls='--', label='baseline caustic (t=0)')
ax.plot(x_env,  y_env,  color='crimson', lw=1.8, label=f'transported caustic (t={t1:.2f})')

# Also plot the curvature bump center path
ts = np.linspace(0, 1, 200)
ax.plot(a0*np.sin(Omega*ts), np.zeros_like(ts), 'k:', lw=1.0, label='K-center path')

ax.set_aspect('equal', 'box')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('Transport of fold caustic by Hamiltonian on ray space')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
