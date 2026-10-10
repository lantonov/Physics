import numpy as np

# Define the parameters of the problem
# Max l and q for demonstration
l_max = 2
q_max = 3

# Time array (s)
t = np.linspace(0, 1, 500)

# Observation direction (unit vector)
n_hat = np.array([0, 0, 1])

# Define TT projector for an arbitrary direction n_hat
def tt_projector(n):
    P = np.eye(3) - np.outer(n, n)
    Lambda = np.einsum('ik,jl->ijkl', P, P) - 0.5 * np.einsum('ij,kl->ijkl', P, P)
    return Lambda

Lambda = tt_projector(n_hat)

# Generate mock c_qℓm(t) coefficients for demonstration
# Shape: (q_max+1, l_max+1, 2*l_max+1, len(t))
# Note: m in [-l, l] -> index m + l
c_q_l_m = np.random.randn(q_max+1, l_max+1, 2*l_max+1, len(t)) + 1j*np.random.randn(q_max+1, l_max+1, 2*l_max+1, len(t))

# Mock radial integration factor (for simplicity)
r4_integral = 1.0  # In practice, would integrate r^4 dr times radial density

# Precompute spherical harmonics Y_{ℓm} at a grid (mock: set to 1 for simplicity)
# For actual evaluation, use scipy.special.sph_harm
Y_l_m = np.ones((l_max+1, 2*l_max+1), dtype=complex)  # placeholder

# Rank-2 spherical harmonics matrices (mock: identity)
# In practice, compute Cartesian representation of Y_{2m}
Y2m_ij = np.ones((5,3,3))  # 5 for m=-2..2, 3x3 matrix

# Function to compute Q_ij^(q)(t) (TT projected)
def compute_Q_TT(q_index):
    Q_tt = np.zeros((3,3,len(t)), dtype=complex)
    for l in range(0, l_max+1):
        for m_index, m in enumerate(range(-l,l+1)):
            # Mock gradient term and q^2/R^2 term: for simplicity, include as scalar
            term = r4_integral * (np.abs(c_q_l_m[q_index,l,m_index,:])**2)  # |c|^2 term only
            # Map to rank-2 tensor via Y2m_ij (mock: use m_index modulo 5)
            # Corrected: Y_ij should be a (3,3) matrix for the projector contraction
            Y_ij = Y2m_ij[m_index%5,:,:] # Shape (3,3)

            # Corrected: Apply the TT projector using einsum for correct tensor contraction
            # The result of einsum('ijkl,kl->ij', Lambda, Y_ij) is (3,3)
            # This (3,3) tensor is then multiplied by 'term' (500,), broadcasting to (3,3,500)
            Q_tt += np.einsum('ijkl,kl->ij', Lambda, Y_ij)[:,:,np.newaxis] * term
    return Q_tt

# Compute Q_ij^(q) TT for each q-sector
Q_TT_all_q = [compute_Q_TT(q) for q in range(q_max+1)]

# Compute second derivative w.r.t. time to get h_ij(t)
dt = t[1] - t[0]
h_ij_q = []
for Q_tt in Q_TT_all_q:
    d2Q_dt2 = np.gradient(np.gradient(Q_tt, dt, axis=2), dt, axis=2)
    h_ij = 2 / 1.0 * d2Q_dt2  # 2G/(c^4 r), set constants=1 for demo
    h_ij_q.append(h_ij)

# Extract h_+ and h_x from TT tensor for n_hat along z
# h_+ = h_xx - h_yy, h_x = 2 h_xy in TT frame
def extract_h_plus_cross(h_ij):
    h_plus = h_ij[0,0,:] - h_ij[1,1,:]
    h_cross = 2*h_ij[0,1,:]
    return h_plus.real, h_cross.real

# Collect h_+, h_x for each q-sector
h_plus_cross_q = [extract_h_plus_cross(h) for h in h_ij_q]

# Display results for each q-sector
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
for q, (h_plus, h_cross) in enumerate(h_plus_cross_q):
    plt.plot(t, h_plus, label=f'h_+ (q={q})')
    plt.plot(t, h_cross, '--', label=f'h_x (q={q})')
plt.xlabel('Time [s]')
plt.ylabel('Strain (arb units)')
plt.title('GW signal from Hopf q-sectors (mock data)')
plt.legend()
plt.show()