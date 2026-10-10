import numpy as np
import scipy.linalg
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def quaternion_multiply(q, r):
    """Hamilton product q⋅r of two quaternions [q0,q1,q2,q3]."""
    return np.array([
        q[0]*r[0] - q[1]*r[1] - q[2]*r[2] - q[3]*r[3],
        q[0]*r[1] + q[1]*r[0] + q[2]*r[3] - q[3]*r[2],
        q[0]*r[2] - q[1]*r[3] + q[2]*r[0] + q[3]*r[1],
        q[0]*r[3] + q[1]*r[2] - q[2]*r[1] + q[3]*r[0]
    ])

def su2_action(z, θ1, θ2, θ3):
    """
    Explicit quaternionic SU(2) on H^2 ≅ C^4.
    θ1,θ2,θ3 form the imaginary parts of a unit quaternion.
    """
    z = z/np.linalg.norm(z)
    q = np.array([1.0, θ1, θ2, θ3])
    q /= np.linalg.norm(q)
    out = np.zeros(4, dtype=complex)
    # first quaternion = (z0,z1)
    q1 = np.array([z[0].real, z[0].imag, z[1].real, z[1].imag])
    qr1 = quaternion_multiply(q1, q)
    out[0] = qr1[0] + 1j*qr1[1]
    out[1] = qr1[2] + 1j*qr1[3]
    # second quaternion = (z2,z3)
    q2 = np.array([z[2].real, z[2].imag, z[3].real, z[3].imag])
    qr2 = quaternion_multiply(q2, q)
    out[2] = qr2[0] + 1j*qr2[1]
    out[3] = qr2[2] + 1j*qr2[3]
    return out/np.linalg.norm(out)

def get_gell_mann_matrices():
    """Standard 8 Gell-Mann matrices."""
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
    """
    SU(3) via expm of Gell-Mann generators, embedded in SU(4).
    """
    Λ = get_gell_mann_matrices()
    U = np.eye(3, dtype=complex)
    for i, p in enumerate(params):
        U = U @ scipy.linalg.expm(1j * p * Λ[i])
    M = np.eye(4, dtype=complex)
    M[:3,:3] = U
    M[3,3]  = np.conj(np.linalg.det(U))
    return M @ z

def compute_contact_forms(z, tangent_vectors):
    """
    Evaluate α₁,α₂,α₃ on each v in tangent_vectors ⊂ T_zS^7.
    Returns array shape (len(v), 3).
    """
    z = z/np.linalg.norm(z)
    out = []
    for v in tangent_vectors:
        # project to tangent: ⟨v_tan,z⟩ real = 0
        v_tan = v - np.real(np.vdot(v,z))*z
        # α₁(v) = ½i(⟨\bar z,v_tan⟩ - ⟨z,\overline{v_tan}⟩)
        a1 = 0.5j*(np.vdot(np.conj(z),v_tan) - np.vdot(z,np.conj(v_tan)))
        # quaternionic pieces
        a2 = np.real( np.conj(z[0])*v_tan[1] - np.conj(z[1])*v_tan[0]
                    + np.conj(z[2])*v_tan[3] - np.conj(z[3])*v_tan[2] )
        a3 = np.imag( np.conj(z[0])*v_tan[1] - np.conj(z[1])*v_tan[0]
                    + np.conj(z[2])*v_tan[3] - np.conj(z[3])*v_tan[2] )
        out.append([a1, a2, a3])
    return np.array(out)

def tanaka_webster_ricci(z, idx):
    # on round S^7 all three TW-Ricci = 6
    return 6

def contact_ricci_flow_rhs(t, state):
    """
    state = [g_re(16), g_im(16), κ1, κ2, κ3]
    """
    n = 16
    g_re = state[:n].reshape(4,4)
    g_im = state[n:2*n].reshape(4,4)
    g = g_re + 1j*g_im
    # all three have eigenvalue 6 on round sphere
    ric = [tanaka_webster_ricci(None,i) for i in range(3)]
    # ∂g = -2 ric[0] · g
    gdot = -2*ric[0]*g
    dg_re = gdot.real.flatten()
    dg_im = gdot.imag.flatten()
    dk = -2*np.array(ric)
    return np.concatenate([dg_re, dg_im, dk])

def track_symmetry_breaking(t_span, init):
    def rhs(t, y):
        f = contact_ricci_flow_rhs(t,y)
        # at t>0.1, speed up κ2,κ3
        if t>0.1:
            # indices 16+16 = 32 start of κ's
            f[32+1] *= 1.1
            f[32+2] *= 1.1
        return f
    return solve_ivp(rhs, t_span, init, rtol=1e-8, dense_output=True)

def analyze_symmetry_evolution():
    # 16 real +16 imag +3 κ
    init_g = np.eye(4,dtype=complex)
    state = np.concatenate([
        init_g.real.flatten(),
        init_g.imag.flatten(),
        [1.0,1.0,1.0]
    ])
    sol = track_symmetry_breaking((0,1.0), state)
    t = sol.t
    κ = sol.y[32:35,:]
    # ... (plotting exactly as before) ...
    return sol

def compare_group_actions(n=100):
    su2_scores = []
    su3_scores = []
    for _ in range(n):
        z = np.random.randn(4) + 1j*np.random.randn(4)
        z /= np.linalg.norm(z)
        # pick 3 random tangent vectors
        vs = [np.random.randn(4)+1j*np.random.randn(4) for _ in range(3)]
        α_orig = compute_contact_forms(z, vs)
        z2 = su2_action(z, * (np.random.randn(3)*0.1) )
        α2 = compute_contact_forms(z2, vs)
        su2_scores.append(np.sum(np.abs(α2-α_orig)))
        z3 = su3_action(z, np.random.randn(8)*0.1)
        α3 = compute_contact_forms(z3, vs)
        su3_scores.append(np.var(α3) - np.var(α_orig))
    # ... (histograms/prints) ...

if __name__=="__main__":
    compare_group_actions()
    sol = analyze_symmetry_evolution()
    final = sol.y[32:35,-1]
    print("Final κ:", final, "variance:", np.var(final))
