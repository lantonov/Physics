# honest_magnus_hopf_comparison.py
# Honest comparison of Hopf fibration vs Magnus expansion for Berry phases
# No result overriding or unfair subsampling

import numpy as np
from scipy.linalg import expm
import time

def hopf_to_R4(theta, psi, phi):
    """Convert Hopf coordinates to R^4 point on S^3."""
    c_th2, s_th2 = np.cos(theta/2), np.sin(theta/2)
    exp_avg = np.exp(1j * (psi + phi) / 2)
    exp_diff = np.exp(1j * (psi - phi) / 2)
    
    z1 = c_th2 * exp_avg
    z2 = s_th2 * exp_diff
    
    return np.array([z1.real, z1.imag, z2.real, z2.imag])

def compute_omega(gamma, dgamma):
    """Bivector spin connection Ω = γ ∧ dγ (4×4 skew-symmetric)."""
    return np.outer(gamma, dgamma) - np.outer(dgamma, gamma)

def geodesic_rotor(gamma_T, gamma_0):
    """Rotor for parallel transport along geodesic from T to 0."""
    cos_alpha = np.dot(gamma_T, gamma_0)
    if abs(cos_alpha - 1.0) < 1e-12:
        return np.eye(4)
    
    sin_alpha = np.sqrt(max(0.0, 1 - cos_alpha**2))
    v = (gamma_0 - cos_alpha * gamma_T) / sin_alpha
    Omega = compute_omega(gamma_T, v)
    alpha = np.arccos(np.clip(cos_alpha, -1, 1))
    
    # Handle normalization carefully
    omega_norm = np.sqrt(np.abs(np.tensordot(Omega, Omega)) / 2)
    if omega_norm > 1e-12:
        Omega_normalized = Omega / omega_norm
    else:
        Omega_normalized = Omega
        
    Omega_M = -0.5 * alpha * Omega_normalized
    return expm(Omega_M)

def compute_R_exact(gammas, dgammas, ds):
    """Exact parallel transport R(T) by product of infinitesimal rotors."""
    R = np.eye(4)
    for k in range(len(gammas)-1):
        Omega_k = compute_omega(gammas[k], dgammas[k])
        A_k = -0.5 * Omega_k * ds
        R_k = expm(A_k)
        R = R_k @ R  # Correct order: R(s+ds) = R_k(s) @ R(s)
    return R

def magnus_order3_full(Omegas, ds):
    """3rd-order Magnus expansion using ALL points (no subsampling)."""
    N = len(Omegas)
    M_list = [-0.5 * O for O in Omegas]
    
    # Ω1 term
    Omega1 = np.sum(M_list, axis=0) * ds
    
    # Ω2 term - full computation
    Omega2 = np.zeros((4, 4))
    for i in range(N):
        for j in range(i):
            comm = M_list[i] @ M_list[j] - M_list[j] @ M_list[i]
            Omega2 += comm * (ds ** 2)
    
    # Ω3 term - full computation  
    Omega3 = np.zeros((4, 4))
    for i in range(N):
        for j in range(i):
            for k in range(j):
                comm12 = M_list[i] @ M_list[j] - M_list[j] @ M_list[i]
                comm = comm12 @ M_list[k] - M_list[k] @ comm12
                Omega3 += comm * (ds ** 3)
    
    Omega_M = Omega1 + Omega2 + Omega3
    return Omega_M

def berry_phase(R, gamma0, gamma_T):
    """Compute Berry phase using geodesic closure."""
    U = geodesic_rotor(gamma_T, gamma0)
    H = U @ R
    
    # Project to spinor and compute overlap
    z1_0 = gamma0[0] + 1j * gamma0[1]
    z2_0 = gamma0[2] + 1j * gamma0[3]
    
    vec = H @ gamma0
    vec_norm = np.linalg.norm(vec)
    if vec_norm > 0:
        vec /= vec_norm
        
    z1_h = vec[0] + 1j * vec[1]
    z2_h = vec[2] + 1j * vec[3]
    
    overlap = np.conj(z1_0) * z1_h + np.conj(z2_0) * z2_h
    return np.angle(overlap)

def generate_test_paths():
    """Generate three physically motivated test paths."""
    
    def figure_eight(s):
        """Figure-eight pattern on S²"""
        theta = np.pi/2 + 0.3 * np.sin(2*s)
        psi = s + 0.2 * np.sin(3*s) 
        phi = 0.5 * s
        return theta, psi, phi
    
    def spiral(s):
        """Spiral from pole to pole"""
        theta = np.pi * s / (2*np.pi)
        psi = 2 * s
        phi = 0.8 * s
        return theta, psi, phi
    
    def random_walk(s):
        """Random walk-like path"""
        theta = np.pi/2 + 0.4 * np.sin(1.5*s) * np.cos(2.1*s)
        psi = 1.2 * s + 0.3 * np.sin(2.7*s) * np.cos(1.8*s)
        phi = s + 0.2 * np.sin(3.2*s)
        return theta, psi, phi
    
    return {
        'figure-eight': figure_eight,
        'spiral': spiral, 
        'random': random_walk
    }

def compute_solid_angle(theta, psi):
    """Compute solid angle enclosed by path on S² using spherical shoelace."""
    theta_closed = np.append(theta, theta[0])
    psi_closed = np.append(psi, psi[0] + 2*np.pi)  # Close the curve
    dpsi = np.diff(psi_closed)
    spherical_area = 0.5 * np.abs(np.sum(np.sin(theta_closed[:-1]) * dpsi))
    return spherical_area

# ===== MAIN COMPUTATION =====
print("HONEST COMPARISON: Hopf vs Magnus Methods")
print("=" * 60)

# Parameters
T = 2 * np.pi
N = 500  # Reduced for reasonable computation time
ds = T / N
s_array = np.linspace(0, T, N)

test_paths = generate_test_paths()
results = []

for path_name, path_func in test_paths.items():
    print(f"\nProcessing {path_name}...")
    
    # Generate path coordinates
    coords = [path_func(s) for s in s_array]
    theta = np.array([c[0] for c in coords])
    psi = np.array([c[1] for c in coords]) 
    phi = np.array([c[2] for c in coords])
    
    # Compute path on S³
    gammas = np.array([hopf_to_R4(theta[k], psi[k], phi[k]) for k in range(N)])
    dgammas = np.gradient(gammas, ds, axis=0)
    
    gamma0 = gammas[0]
    gamma_T = gammas[-1]
    
    # Compute solid angle for reference
    solid_angle = compute_solid_angle(theta, psi)
    expected_berry = -solid_angle / 2  # Theoretical prediction
    
    print(f"  Solid angle: {solid_angle:.4f} rad")
    print(f"  Expected γ_B: {expected_berry:.4f} rad")
    
    # Method 1: Exact parallel transport (Hopf method reference)
    print("  Computing exact transport...")
    start_time = time.time()
    R_exact = compute_R_exact(gammas, dgammas, ds)
    exact_time = time.time() - start_time
    gamma_exact = berry_phase(R_exact, gamma0, gamma_T)
    
    print(f"    Exact γ_B: {gamma_exact:.4f} rad (time: {exact_time:.2f}s)")
    
    # Method 2: Magnus expansion (order 3)
    print("  Computing Magnus expansion...")
    start_time = time.time()
    Omegas = [compute_omega(g, dg) for g, dg in zip(gammas, dgammas)]
    Omega_M3 = magnus_order3_full(Omegas, ds)
    R_m3 = expm(Omega_M3)
    magnus_time = time.time() - start_time
    gamma_m3 = berry_phase(R_m3, gamma0, gamma_T)
    
    print(f"    Magnus γ_B: {gamma_m3:.4f} rad (time: {magnus_time:.2f}s)")
    
    # Compute actual errors
    error_abs = abs(gamma_m3 - gamma_exact)
    error_rel = error_abs / (abs(gamma_exact) + 1e-12) * 100
    
    # Convergence test with smaller N
    print("  Convergence check...")
    N_test = N // 2
    s_test = np.linspace(0, T, N_test)
    coords_test = [path_func(s) for s in s_test]
    theta_test = np.array([c[0] for c in coords_test])
    psi_test = np.array([c[1] for c in coords_test])
    phi_test = np.array([c[2] for c in coords_test])
    
    gammas_test = np.array([hopf_to_R4(theta_test[k], psi_test[k], phi_test[k]) 
                           for k in range(N_test)])
    dgammas_test = np.gradient(gammas_test, T/N_test, axis=0)
    
    R_exact_test = compute_R_exact(gammas_test, dgammas_test, T/N_test)
    gamma_exact_test = berry_phase(R_exact_test, gammas_test[0], gammas_test[-1])
    
    convergence_error = abs(gamma_exact_test - gamma_exact)
    
    results.append({
        'path': path_name,
        'solid_angle': solid_angle,
        'expected': expected_berry,
        'exact': gamma_exact,
        'magnus': gamma_m3,
        'error_abs': error_abs,
        'error_rel': error_rel,
        'exact_time': exact_time,
        'magnus_time': magnus_time,
        'convergence_error': convergence_error
    })

# ===== RESULTS =====
print("\n" + "=" * 80)
print("HONEST RESULTS: Method Comparison")
print("=" * 80)
print(f"{'Path':<15} {'Solid Angle':<12} {'Exact γ_B':<10} {'Magnus γ_B':<11} {'Abs Error':<10} {'Rel Error':<10} {'Time Ratio':<10}")
print("-" * 80)

for r in results:
    time_ratio = r['magnus_time'] / r['exact_time']
    print(f"{r['path']:<15} {r['solid_angle']:>11.4f} {r['exact']:>9.4f} {r['magnus']:>10.4f} "
          f"{r['error_abs']:>9.4f} {r['error_rel']:>9.1f}% {time_ratio:>9.1f}x")

print("=" * 80)

# Convergence analysis
print("\nCONVERGENCE ANALYSIS:")
print(f"{'Path':<15} {'N='+str(N):<10} {'N='+str(N//2):<10} {'Difference':<12}")
print("-" * 50)
for r in results:
    print(f"{r['path']:<15} {r['exact']:>9.4f} {r['exact'] + r['convergence_error']:>9.4f} "
          f"{r['convergence_error']:>11.4f}")

print("\n" + "=" * 80)
print("KEY FINDINGS:")
print("1. Both methods should converge to the same result as N increases")
print("2. Magnus expansion is computationally more expensive but may be faster for small N")  
print("3. Relative errors show true method accuracy")
print("4. Time ratios show computational efficiency")
print("=" * 80)