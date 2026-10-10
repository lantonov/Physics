import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm, pro_ang1
from scipy.ndimage import gaussian_filter1d

# Parameters
N = 100_000
l_bulge, m_bulge = 4, 2
m_sph, n_sph, c_sph = 2, 4, 3.0
m_ell, n_ell, c_ell_a, c_ell_b = 2, 4, 2.0, 5.0
sigma_smooth = 5.0

# Sampling functions
def sample_spherical(l, m, N):
    samples = []
    thetas = np.linspace(0, np.pi, 200)
    phis = np.linspace(0, 2*np.pi, 200)
    TH, PH = np.meshgrid(thetas, phis, indexing='ij')
    Yvals = sph_harm(m, l, PH, TH)
    dens = (np.abs(Yvals)**2) * np.sin(TH)
    max_d = dens.max()
    while len(samples) < N:
        theta = np.arccos(1 - 2*np.random.rand())
        phi = 2*np.pi * np.random.rand()
        y = (np.abs(sph_harm(m, l, phi, theta))**2) * np.sin(theta)
        if np.random.rand() * max_d < y:
            samples.append(theta)
    return np.array(samples)

def sample_spheroidal(m, n, c, N):
    samples = []
    thetas = np.linspace(0, np.pi, 400)
    Svals = pro_ang1(m, n, c, np.cos(thetas))[0]
    dens = (Svals**2) * np.sin(thetas)
    max_d = dens.max()
    while len(samples) < N:
        theta = np.arccos(1 - 2*np.random.rand())
        Sval = pro_ang1(m, n, c, np.cos(theta))[0]
        y = (Sval**2) * np.sin(theta)
        if np.random.rand() * max_d < y:
            samples.append(theta)
    return np.array(samples)

def sample_ellipsoidal_proxy(m, n, c_a, c_b, N):
    N2 = N // 2
    part_a = sample_spheroidal(m, n, c_a, N2)
    part_b = sample_spheroidal(m, n, c_b, N - N2)
    return np.concatenate([part_a, part_b])

def sample_hyperbolic(N):
    u = np.random.rand(N) * 2 - 1
    chi = 0.5 * np.log((1 + u)/(1 - u))
    return chi

# Generate samples
theta_bulge = sample_spherical(l_bulge, m_bulge, N)
theta_sph = sample_spheroidal(m_sph, n_sph, c_sph, N)
theta_ell = sample_ellipsoidal_proxy(m_ell, n_ell, c_ell_a, c_ell_b, N)
chi_bulge = np.arctanh(np.cos(theta_bulge))
chi_sph = np.arctanh(np.cos(theta_sph))
chi_ell = np.arctanh(np.cos(theta_ell))
chi_far = sample_hyperbolic(N)

# Histograms & smoothing
bins = 200
hist_bulge, edges = np.histogram(chi_bulge, bins=bins, density=True)
centers = 0.5*(edges[:-1] + edges[1:])
smooth_bulge = gaussian_filter1d(hist_bulge, sigma=sigma_smooth)
hist_sph, _ = np.histogram(chi_sph, bins=bins, density=True)
smooth_sph = gaussian_filter1d(hist_sph, sigma=sigma_smooth)
hist_ell, _ = np.histogram(chi_ell, bins=bins, density=True)
smooth_ell = gaussian_filter1d(hist_ell, sigma=sigma_smooth)
hist_far, _ = np.histogram(chi_far, bins=bins, density=True)
smooth_far = gaussian_filter1d(hist_far, sigma=sigma_smooth)

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].hist(theta_bulge, bins=100, density=True, color='skyblue', edgecolor='k')
axes[0, 0].set_xlabel(r'$\theta$')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title(f'Bulge (spherical ℓ={l_bulge},m={m_bulge})')
axes[0, 0].set_xlim(0, np.pi)
axes[0, 1].plot(centers, smooth_bulge, color='blue', lw=1)
axes[0, 1].set_xlabel(r'$\chi$')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('Bulge (χ‐space, smoothed)')
axes[1, 0].hist(theta_sph, bins=100, density=True, color='lightcoral', edgecolor='k')
axes[1, 0].set_xlabel(r'$\theta$')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_title(f'Near‐bulge (spheroidal m={m_sph},n={n_sph},c={c_sph})')
axes[1, 0].set_xlim(0, np.pi)
axes[1, 1].plot(centers, smooth_sph, color='red', lw=1)
axes[1, 1].set_xlabel(r'$\chi$')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title('Near‐bulge (χ‐space, smoothed)')
plt.tight_layout()
plt.show()

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.hist(theta_ell, bins=100, density=True, color='plum', edgecolor='k')
ax1.set_xlabel(r'$\theta$')
ax1.set_ylabel('Density')
ax1.set_title(f'Halo (ellipsoidal proxy m={m_ell},n={n_ell})')
ax1.set_xlim(0, np.pi)
ax2.plot(centers, smooth_far, color='darkgreen', lw=1)
ax2.set_xlabel(r'$\chi$')
ax2.set_ylabel('Density')
ax2.set_title('Distant observer (hyperbolic kernel)')
plt.tight_layout()
plt.show()