import numpy as np
from scipy.linalg import expm
import pandas as pd
import matplotlib.pyplot as plt  # For convergence plot

# Wedge product for bivector
def wedge(u, v):
    W = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            W[i, j] = u[i] * v[j] - u[j] * v[i]
    return W

# Second-order Magnus expansion
def magnus_expansion(A_list, ds):
    N = len(A_list)
    Omega1 = np.zeros_like(A_list[0])
    for A in A_list:
        Omega1 += A * ds
    Omega2 = np.zeros_like(A_list[0])
    for i in range(N):
        for j in range(i):
            comm = np.dot(A_list[i], A_list[j]) - np.dot(A_list[j], A_list[i])
            Omega2 += 0.5 * ds**2 * comm
    return Omega1 + Omega2

# Path parameterization (corrected normalization: ||~γ||^2 = 1 + sin^2 s)
def gamma4(s, nu=0.5):
    kx = s
    ky = np.sin(nu * s)
    w = s
    raw = np.array([np.cos(kx), np.sin(kx) * np.cos(ky), np.sin(kx) * np.sin(ky), np.sin(w)])
    norm = np.linalg.norm(raw)  # sqrt(1 + sin^2 s)
    return raw / norm if norm > 1e-10 else raw

# Numerical derivative
def gamma4_dot(s, nu=0.5):
    eps = 1e-6
    return (gamma4(s + eps, nu) - gamma4(s - eps, nu)) / (2 * eps)

# Hopf coordinates with robust extraction
def hopf_project(gamma):
    x0, x1, x2, x3 = gamma
    z1 = x0 + 1j * x1
    z2 = x2 + 1j * x3
    sin_theta_cos_psi = 2 * np.real(z1 * np.conj(z2))
    sin_theta_sin_psi = 2 * np.imag(z1 * np.conj(z2))
    cos_theta = np.abs(z1)**2 - np.abs(z2)**2
    r2 = np.sqrt(sin_theta_cos_psi**2 + sin_theta_sin_psi**2 + cos_theta**2)
    if r2 > 1e-10:
        sin_theta_cos_psi /= r2
        sin_theta_sin_psi /= r2
        cos_theta /= r2
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Stability clip
    if np.sin(theta) < 1e-10:  # Pole degeneracy: set psi=0
        psi = 0.0
    else:
        psi = np.arctan2(sin_theta_sin_psi, sin_theta_cos_psi)
    return theta, psi

# Extract phi = arg(z1) - arg(z2)
def get_phi(gamma):
    x0, x1, x2, x3 = gamma
    z1 = x0 + 1j * x1
    z2 = x2 + 1j * x3
    arg_z1 = np.angle(z1)
    arg_z2 = np.angle(z2)
    return arg_z1 - arg_z2

# Signed spherical excess (orientation via mean normal)
def signed_spherical_area(points):
    normal = np.mean(points, axis=0)  # Reference orientation
    normal /= np.linalg.norm(normal) + 1e-12  # Avoid div0
    n = len(points)
    area = 0.0
    for i in range(n):
        p1 = points[i]
        p2 = points[(i + 1) % n]
        cross = np.cross(p1, p2)
        signed_mag = np.dot(cross, normal)
        area += np.arctan2(signed_mag, np.clip(np.dot(p1, p2), -1, 1))
    return area

# Line integral with clipped unwrapping and closure
def line_integral_berry(thetas, psis, phis):
    # Clipped diffs for unwrapping
    dphi = np.diff(phis)
    dphi = np.where(np.abs(dphi) > np.pi, dphi - 2 * np.pi * np.sign(dphi), dphi)
    dpsi = np.diff(psis)
    dpsi = np.where(np.abs(dpsi) > np.pi, dpsi - 2 * np.pi * np.sign(dpsi), dpsi)

    # Midpoint cos theta
    cos_theta_mid = np.cos((thetas[:-1] + thetas[1:]) / 2)

    # Integral over open path
    integrand = 0.5 * (dphi + cos_theta_mid * dpsi)
    gamma_open = np.sum(integrand)

    # Closure: ensure total Delta phi = 2 pi w (integer multiple)
    total_delta_phi = phis[-1] - phis[0] + 2 * np.pi * np.round((phis[0] - phis[-1]) / (2 * np.pi))
    closure_dphi = total_delta_phi - np.sum(dphi)
    closure_dpsi = (psis[0] - psis[-1]) + 2 * np.pi * np.round((psis[-1] - psis[0]) / (2 * np.pi)) - np.sum(dpsi)
    closure_theta_mid = np.cos((thetas[-1] + thetas[0]) / 2)
    gamma_closure = 0.5 * (closure_dphi + closure_theta_mid * closure_dpsi)

    return gamma_open + gamma_closure

# Path-ordered holonomy (forward)
def path_ordered_holonomy(A_list, ds):
    U = np.eye(4)
    for A in A_list:
        dU = expm(A * ds)
        U = np.dot(dU, U)
    return U

# SU(2) phase via quaternion first-column embedding
def extract_berry_phase(U):
    q = U[:, 0]
    u11 = q[0] + 1j * q[1]
    return np.angle(u11) % (2 * np.pi)

# Core computation
def compute_berry_phases(N, nu=0.5):
    s_vals = np.linspace(0, 2 * np.pi, N)
    ds = s_vals[1] - s_vals[0]
    gamma_func = lambda s: gamma4(s, nu)

    A_list = [-0.5 * wedge(gamma_func(s), gamma4_dot(s, nu)) for s in s_vals]

    # Magnus
    Omega = magnus_expansion(A_list, ds)
    U_magnus = expm(Omega)
    gamma_magnus = extract_berry_phase(U_magnus)

    # Path-ordered
    U_path = path_ordered_holonomy(A_list, ds)
    gamma_path = extract_berry_phase(U_path)

    # Hopf
    gammas = [gamma_func(s) for s in s_vals]
    thetas, psis = np.array([hopf_project(g) for g in gammas]).T
    phis = np.array([get_phi(g) for g in gammas])

    psis_unw = np.unwrap(psis)
    phis_unw = np.unwrap(phis)

    points = np.array([[np.sin(th) * np.cos(ps), np.sin(th) * np.sin(ps), np.cos(th)] 
                       for th, ps in zip(thetas, psis_unw)])
    area_signed = signed_spherical_area(points)

    delta_phi = phis_unw[-1] - phis_unw[0]
    w = delta_phi / (2 * np.pi)

    gamma_hopf = (np.pi * w - 0.5 * area_signed) % (2 * np.pi)

    gamma_line = line_integral_berry(thetas, psis_unw, phis_unw)

    return {
        'N': N,
        'Signed A (sr)': round(area_signed, 4),
        'w': round(w, 3),
        'Hopf Signed (rad)': round(gamma_hopf, 4),
        'Line Integral (rad)': round(gamma_line, 4),
        'Magnus SU(2) (rad)': round(gamma_magnus, 4),
        'Path-Ordered (rad)': round(gamma_path, 4)
    }

# Generate table
Ns = [100, 200, 500, 1000, 2000]
results = [compute_berry_phases(N) for N in Ns]
df = pd.DataFrame(results)
df = df.round(4)  # Precision as in table

# Print console table
print(df.to_string(index=False))

# LaTeX table (raw string to avoid escapes)
latex_caption = r'Convergence of Berry phase methods for $\nu=0.5$. All values modulo $2\pi$; target $\pi \approx 3.1416$.'
latex_table = df.to_latex(index=False, escape=False, column_format='c|c|c|c|c|c|c', 
                          caption=latex_caption)
print("\nLaTeX Table:\n", latex_table)

# Convergence plot
fig, ax = plt.subplots(figsize=(8, 6))
pi_true = np.pi
Ns_arr = np.array(Ns)
methods = ['Hopf', 'Line', 'Magnus', 'Path']
colors = ['blue', 'green', 'red', 'orange']
for i, method in enumerate(methods):
    vals = df[[f for f in df.columns if method.lower() in f.lower()]][0].values
    ax.semilogy(Ns_arr, np.abs(vals - pi_true), 'o-', label=method, color=colors[i], linewidth=2)

ax.set_xlabel('Discretization $N$', fontsize=12)
ax.set_ylabel('Residual $|\\gamma - \\pi|$ (rad)', fontsize=12)
ax.set_title('Convergence to Topological $\\pi$-Phase', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('convergence.pdf', dpi=300, bbox_inches='tight')  # Archival plot
plt.show()

# Residual diagnostics
max_res = np.max(np.abs(df.iloc[:, 3:].values - pi_true))
print(f"\nMax residual at N=2000: {max_res:.2e} rad")