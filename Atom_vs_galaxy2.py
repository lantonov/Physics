import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm

# --------------------------------------------------------
# Parameters: which spherical harmonic to use (ℓ, m)
# --------------------------------------------------------
l = 3
m = 1

# Number of Monte Carlo samples
N = 100_000

# --------------------------------------------------------
# 1. Sample (theta, phi) from |Y_l^m|^2 on the sphere
# --------------------------------------------------------
#
# We do rejection sampling on (theta, phi) with density 
#    p(θ,φ) ∝ |Y_l^m(θ,φ)|^2 * sin(θ),
# which is exactly the usual probability measure for a spherical harmonic intensity.
#

# First find the approximate maximum of |Y_l^m|^2 * sin(theta) on a fine grid,
# so that our rejection sampler will be properly bounded.
thetas_grid = np.linspace(0, np.pi, 300)
phis_grid   = np.linspace(0, 2*np.pi, 300)
TH, PH = np.meshgrid(thetas_grid, phis_grid, indexing='ij')
Y_vals = sph_harm(m, l, PH, TH)
density_grid = (np.abs(Y_vals)**2) * np.sin(TH)
max_density = density_grid.max()

def sample_theta_phi(l, m, N, max_density):
    """
    Rejection‐sample N points (θ,φ) from the |Y_l^m(θ,φ)|^2 * sin(θ) distribution
    on [0,π]×[0,2π].
    Returns an array of shape (N,2) with columns [θ, φ].
    """
    samples = []
    tries = 0
    while len(samples) < N:
        # 1) Propose theta uniformly by drawing u in [0,1], then theta = arccos(1-2u)
        #    ensures uniform sampling in cos(θ).  Equivalently, just do:
        theta = np.arccos(1 - 2*np.random.rand())
        phi   = 2 * np.pi * np.random.rand()
        # 2) Compute the “target density” at that point:
        y = np.abs(sph_harm(m, l, phi, theta))**2 * np.sin(theta)
        # 3) Accept/reject
        if np.random.rand() * max_density < y:
            samples.append((theta, phi))
        tries += 1
        if tries > 20*N:
            # just in case it is converging poorly
            break
    return np.array(samples)

# Draw samples
theta_phi_samples = sample_theta_phi(l, m, N, max_density)
thetas = theta_phi_samples[:, 0]   # shape (N,)
phis   = theta_phi_samples[:, 1]   # shape (N,)

# --------------------------------------------------------
# 2. Map θ → χ = arctanh(cos θ), then plot histograms
# --------------------------------------------------------
cos_thetas = np.cos(thetas)
# Note: cos(θ) ∈ [-1, +1], so arctanh(cos(θ)) ∈ (-∞, +∞)
chi = np.arctanh(cos_thetas)

# Plot: 1) histogram of θ,  2) histogram of χ
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(thetas, bins=100, density=True, color='skyblue', edgecolor='k')
plt.xlabel(r'$\theta$')
plt.title(f'Distribution of $\\theta$ for $Y_{{{l}}}^{{{m}}}$')

plt.subplot(1, 2, 2)
plt.hist(chi, bins=100, density=True, color='salmon', edgecolor='k')
plt.xlabel(r'$\chi = \tanh^{-1}(\cos\theta)$')
plt.title('Mapped Distribution on $\chi$ (Hyperbolic)')

plt.tight_layout()
plt.show()
