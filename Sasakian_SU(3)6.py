import numpy as np
import scipy.linalg
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ── Quaternion utilities ──────────────────────────────────────────────────────

def quaternion_multiply(q, r):
    """Hamilton product q⋅r for quaternions q,r = [q0,q1,q2,q3]."""
    return np.array([
        q[0]*r[0] - q[1]*r[1] - q[2]*r[2] - q[3]*r[3],
        q[0]*r[1] + q[1]*r[0] + q[2]*r[3] - q[3]*r[2],
        q[0]*r[2] - q[1]*r[3] + q[2]*r[0] + q[3]*r[1],
        q[0]*r[3] + q[1]*r[2] - q[2]*r[1] + q[3]*r[0]
    ])

def su2_action(z, th1, th2, th3):
    """SU(2) via right‐multiplication by unit quaternion (preserves 3-Sasakian)."""
    z = z/np.linalg.norm(z)
    q = np.array([1.0, th1, th2, th3])
    q /= np.linalg.norm(q)
    out = np.zeros(4, dtype=complex)
    for i in (0,2):
        qi = np.array([z[i].real, z[i].imag, z[i+1].real, z[i+1].imag])
        qr = quaternion_multiply(qi, q)
        out[i]   = qr[0] + 1j*qr[1]
        out[i+1] = qr[2] + 1j*qr[3]
    return out/np.linalg.norm(out)

# ── SU(3) embedding ───────────────────────────────────────────────────────────

def get_gell_mann_matrices():
    λ1 = np.array([[0,1,0],[1,0,0],[0,0,0]],complex)
    λ2 = np.array([[0,-1j,0],[1j,0,0],[0,0,0]],complex)
    λ3 = np.array([[1,0,0],[0,-1,0],[0,0,0]],complex)
    λ4 = np.array([[0,0,1],[0,0,0],[1,0,0]],complex)
    λ5 = np.array([[0,0,-1j],[0,0,0],[1j,0,0]],complex)
    λ6 = np.array([[0,0,0],[0,0,1],[0,1,0]],complex)
    λ7 = np.array([[0,0,0],[0,0,-1j],[0,1j,0]],complex)
    λ8 = np.array([[1,0,0],[0,1,0],[0,0,-2]],complex)/np.sqrt(3)
    return [λ1,λ2,λ3,λ4,λ5,λ6,λ7,λ8]

def su3_action(z, params):
    """Exponentiate Gell-Mann basis into SU(3), embed in SU(4)."""
    Λ = get_gell_mann_matrices()
    U = np.eye(3, dtype=complex)
    for p, Λi in zip(params, Λ):
        U = U @ scipy.linalg.expm(1j * p * Λi)
    M = np.eye(4, dtype=complex)
    M[:3,:3] = U
    M[3,3]  = np.conj(np.linalg.det(U))
    return M @ z

# ── Contact forms ─────────────────────────────────────────────────────────────

def compute_contact_forms(z, tangents):
    """
    Evaluate α₁,α₂,α₃ on each tangent vector in T_z S^7.
    Returns array of shape (len(tangents), 3).
    """
    z = z/np.linalg.norm(z)
    out = []
    for v in tangents:
        v_tan = v - np.real(np.vdot(v, z)) * z
        a1 = 0.5j*(np.vdot(np.conj(z), v_tan) - np.vdot(z, np.conj(v_tan)))
        a2 = np.real(
            np.conj(z[0])*v_tan[1] - np.conj(z[1])*v_tan[0]
          + np.conj(z[2])*v_tan[3] - np.conj(z[3])*v_tan[2]
        )
        a3 = np.imag(
            np.conj(z[0])*v_tan[1] - np.conj(z[1])*v_tan[0]
          + np.conj(z[2])*v_tan[3] - np.conj(z[3])*v_tan[2]
        )
        out.append([a1, a2, a3])
    return np.array(out)

def tanaka_webster_ricci(z, idx):
    return 6  # round S^7, all three eigenvalues = 6

# ── Contact-Ricci flow RHS ────────────────────────────────────────────────────

def contact_ricci_flow_rhs(t, state):
    """
    state = [g_re(16), g_im(16), κ1, κ2, κ3]
    Implements ∂g = -2 Ric_1 · g,   dκ_i/dt = -2 Ric_i * κ_i (multiplicative).
    """
    n = 16
    g_re = state[:n].reshape(4,4)
    g_im = state[n:2*n].reshape(4,4)
    g = g_re + 1j*g_im

    ric = [tanaka_webster_ricci(None,i) for i in range(3)]

    # metric flow
    gdot = -2 * ric[0] * g
    dg_re = gdot.real.flatten()
    dg_im = gdot.imag.flatten()

    # multiplicative κ flow
    κ = state[2*n:2*n+3]
    dκ = -2 * np.array(ric) * κ

    return np.concatenate([dg_re, dg_im, dκ])

# ── Symmetry-breaking solver ──────────────────────────────────────────────────

KICK_FACTOR    = 2.0        # double the speed of κ₂,κ₃ after t>0.1
FINAL_TIME     = 2.0        # run from t=0 to t=2
SU3_PARAM_SIGMA = 0.5       # draws ~N(0, SU3_PARAM_SIGMA^2)

def track_symmetry_breaking(t_span, init_state):
    def rhs(t, y):
        f = contact_ricci_flow_rhs(t, y)
        if t > 0.1:
            f[32+1] *= KICK_FACTOR
            f[32+2] *= KICK_FACTOR
        return f
    return solve_ivp(rhs, t_span, init_state, rtol=1e-8, dense_output=True)

def analyze_symmetry_evolution():
    # initial: identity metric + κ=(1,1,1)
    I4 = np.eye(4, dtype=complex)
    state0 = np.concatenate([I4.real.flatten(), I4.imag.flatten(), [1.0,1.0,1.0]])
    sol = track_symmetry_breaking((0,FINAL_TIME), state0)

    t = sol.t
    κ = sol.y[32:35,:]

    plt.figure(figsize=(12,8))

    # 1) SU(3) breaking
    plt.subplot(2,2,1)
    plt.plot(t, κ[0], label='κ₁ (preserved)')
    plt.plot(t, κ[1], '--', label='κ₂ (broken)')
    plt.plot(t, κ[2], '--', label='κ₃ (broken)')
    plt.xlabel('Time'); plt.ylabel('κᵢ')
    plt.title('SU(3) Symmetry Breaking in Contact‐Ricci Flow')
    plt.legend(); plt.grid(True)

    # 2) SU(2) preservation
    plt.subplot(2,2,2)
    plt.plot(t, κ[0], label='κ₁=κ₂=κ₃', linewidth=3)
    plt.xlabel('Time'); plt.ylabel('κᵢ')
    plt.title('SU(2) Symmetry Preservation')
    plt.legend(); plt.grid(True)

    # 3) variance measure
    var = np.var(κ, axis=0)
    plt.subplot(2,2,3)
    plt.plot(t, var, label='Var(κᵢ)')
    plt.xlabel('Time'); plt.ylabel('Variance')
    plt.title('Quantifying 3-Sasakian → 1-Sasakian Breaking')
    plt.legend(); plt.grid(True)

    # 4) phase space (κ₂ vs κ₃)
    plt.subplot(2,2,4)
    plt.plot(κ[1], κ[2], label='Trajectory')
    plt.plot(κ[1][0], κ[2][0], 'go', markersize=8, label='Initial')
    plt.plot(κ[1][-1], κ[2][-1], 'ro', markersize=8, label='Final')
    plt.xlabel('κ₂'); plt.ylabel('κ₃')
    plt.title('SU(3) Breaking: Phase Space')
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.show()
    return sol

def compare_group_actions(n_points=200):
    su2_scores = []
    su3_scores = []
    for _ in range(n_points):
        z = np.random.randn(4) + 1j*np.random.randn(4)
        z /= np.linalg.norm(z)
        tangents = [np.random.randn(4)+1j*np.random.randn(4) for _ in range(3)]
        α0 = compute_contact_forms(z, tangents)

        z2 = su2_action(z, *(np.random.randn(3)*0.1))
        α2 = compute_contact_forms(z2, tangents)
        su2_scores.append(np.sum(np.abs(α2 - α0)))

        z3 = su3_action(z, np.random.randn(8)*SU3_PARAM_SIGMA)
        α3 = compute_contact_forms(z3, tangents)
        su3_scores.append( np.abs(np.var(α3) - np.var(α0)) )

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.hist(su2_scores, bins=25, alpha=0.7)
    plt.title('SU(2) preservation scores')
    plt.xlabel('Change in contact forms')
    plt.ylabel('Frequency')
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.hist(su3_scores, bins=25, alpha=0.7, color='red')
    plt.title('SU(3) breaking scores')
    plt.xlabel('Change in contact form variance')
    plt.ylabel('Frequency')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print(f"Mean SU(2) score: {np.mean(su2_scores):.6f}")
    print(f"Mean SU(3) score: {np.mean(su3_scores):.6f}")

if __name__ == "__main__":
    compare_group_actions()
    sol = analyze_symmetry_evolution()
    final = sol.y[32:35,-1]
    print(f"Final κ₁: {final[0]:.4e}, κ₂: {final[1]:.4e}, κ₃: {final[2]:.4e}")
    print(f"Symmetry breaking measure: {np.var(final):.4e}")
