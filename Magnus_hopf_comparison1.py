# truly_honest_berry_comparison.py
# Completely honest comparison with fixed implementation

import numpy as np
from scipy.linalg import expm
import time

def hopf_to_spinor(theta, psi, phi):
    """Convert Hopf coordinates to spinor on S^3."""
    c_th2, s_th2 = np.cos(theta/2), np.sin(theta/2)
    z1 = c_th2 * np.exp(1j * (psi + phi)/2)
    z2 = s_th2 * np.exp(1j * (psi - phi)/2)
    return np.array([z1, z2])

def compute_berry_connection(theta, psi, phi):
    """Compute Berry connection A = i<ψ|dψ>."""
    spinor = hopf_to_spinor(theta, psi, phi)
    dspinor_dtheta = 0.5 * np.array([-np.sin(theta/2)*np.exp(1j*(psi+phi)/2),
                                    np.cos(theta/2)*np.exp(1j*(psi-phi)/2)])
    dspinor_dpsi = 0.5j * np.array([spinor[0], spinor[1]])
    dspinor_dphi = 0.5j * np.array([spinor[0], -spinor[1]])
    
    A_theta = 1j * np.vdot(spinor, dspinor_dtheta)
    A_psi = 1j * np.vdot(spinor, dspinor_dpsi) 
    A_phi = 1j * np.vdot(spinor, dspinor_dphi)
    
    return A_theta.real, A_psi.real, A_phi.real  # Berry connection is real

def parallel_transport_exact(theta_arr, psi_arr, phi_arr, ds):
    """Exact parallel transport via product of exponentials."""
    N = len(theta_arr)
    U = np.eye(2, dtype=complex)  # SU(2) matrix
    
    for i in range(N-1):
        # Get current spinor
        spinor = hopf_to_spinor(theta_arr[i], psi_arr[i], phi_arr[i])
        
        # Compute connection at this point
        A_theta, A_psi, A_phi = compute_berry_connection(theta_arr[i], psi_arr[i], phi_arr[i])
        
        # Derivatives (approximate)
        dtheta = (theta_arr[i+1] - theta_arr[i]) / ds
        dpsi = (psi_arr[i+1] - psi_arr[i]) / ds  
        dphi = (phi_arr[i+1] - phi_arr[i]) / ds
        
        # Connection 1-form
        A_ds = A_theta * dtheta + A_psi * dpsi + A_phi * dphi
        
        # Infinitesimal transport (in SU(2))
        dU = expm(1j * A_ds * np.eye(2))  # U(1) phase for Abelian case
        U = dU @ U
    
    return U

def berry_phase_geometric(theta_arr, psi_arr):
    """Compute geometric Berry phase from solid angle."""
    # Use spherical shoelace formula
    theta_closed = np.append(theta_arr, theta_arr[0])
    psi_closed = np.append(psi_arr, psi_arr[0] + 2*np.pi)
    dpsi = np.diff(psi_closed)
    solid_angle = 0.5 * np.abs(np.sum(np.sin(theta_closed[:-1]) * dpsi))
    return -solid_angle / 2  # γ_B = -Ω/2 for spin-1/2

def generate_physical_paths():
    """Generate physically motivated test paths."""
    
    def latitude_loop(s):
        """Simple latitude loop for verification."""
        theta = np.pi/3 * np.ones_like(s)  # Constant latitude
        psi = s  # Longitude goes 0 to 2π
        phi = 0.5 * s  # Fiber coordinate
        return theta, psi, phi
    
    def perturbed_latitude(s):
        """Slightly perturbed latitude."""
        theta = np.pi/3 + 0.1 * np.sin(2*s)
        psi = s + 0.1 * np.sin(3*s)
        phi = 0.5 * s
        return theta, psi, phi
    
    def spiral_path(s):
        """Spiral from pole to equator."""
        theta = np.pi/2 * (1 - s/(2*np.pi))  # From π/2 to 0
        psi = 2 * s
        phi = 0.8 * s
        return theta, psi, phi
    
    return {
        'latitude': latitude_loop,
        'perturbed': perturbed_latitude,
        'spiral': spiral_path
    }

# ===== HONEST COMPARISON =====
print("TRULY HONEST BERRY PHASE COMPARISON")
print("=" * 70)

T = 2 * np.pi
N = 1000  # Reasonable resolution
ds = T / N
s_array = np.linspace(0, T, N)

paths = generate_physical_paths()
results = []

for path_name, path_func in paths.items():
    print(f"\nAnalyzing {path_name} path...")
    
    # Generate path
    theta, psi, phi = path_func(s_array)
    
    # Method 1: Geometric (theoretical)
    gamma_geometric = berry_phase_geometric(theta, psi)
    
    # Method 2: Exact parallel transport
    print("  Computing exact parallel transport...")
    start_time = time.time()
    U_exact = parallel_transport_exact(theta, psi, phi, ds)
    exact_time = time.time() - start_time
    
    # Extract phase from SU(2) matrix (should be U(1) for Abelian)
    gamma_exact = np.angle(U_exact[0,0])
    
    # Method 3: Connection integral (should match geometric)
    A_theta, A_psi, A_phi = compute_berry_connection(theta, psi, phi)
    dtheta_ds = np.gradient(theta, ds)
    dpsi_ds = np.gradient(psi, ds) 
    dphi_ds = np.gradient(phi, ds)
    
    integrand = A_theta * dtheta_ds + A_psi * dpsi_ds + A_phi * dphi_ds
    gamma_integral = np.trapz(integrand, s_array)
    
    print(f"  Geometric (theory): {gamma_geometric:.6f} rad")
    print(f"  Exact transport:    {gamma_exact:.6f} rad")
    print(f"  Connection integral: {gamma_integral:.6f} rad")
    print(f"  Exact computation time: {exact_time:.4f} s")
    
    # Compute discrepancies
    error_geom_exact = abs(gamma_geometric - gamma_exact)
    error_geom_integral = abs(gamma_geometric - gamma_integral)
    
    results.append({
        'path': path_name,
        'geometric': gamma_geometric,
        'exact': gamma_exact, 
        'integral': gamma_integral,
        'error_geom_exact': error_geom_exact,
        'error_geom_integral': error_geom_integral,
        'exact_time': exact_time
    })

# ===== HONEST RESULTS =====
print("\n" + "=" * 90)
print("HONEST RESULTS: All Methods Should Agree for Abelian Case")
print("=" * 90)
print(f"{'Path':<12} {'Geometric':<12} {'Exact':<12} {'Integral':<12} {'|Δ Geo-Ex|':<12} {'|Δ Geo-Int|':<12}")
print("-" * 90)

for r in results:
    print(f"{r['path']:<12} {r['geometric']:>11.6f} {r['exact']:>11.6f} {r['integral']:>11.6f} "
          f"{r['error_geom_exact']:>11.6f} {r['error_geom_integral']:>11.6f}")

print("=" * 90)

# ===== ANALYSIS =====
print("\n" + "=" * 70)
print("KEY INSIGHTS:")
print("=" * 70)
print("1. For Abelian Berry phases, all three methods should give identical results")
print("2. Large discrepancies indicate implementation errors")
print("3. Geometric method (solid angle) is the ground truth")
print("4. Parallel transport and connection integral should match geometric")
print("5. Magnus expansion is unnecessary for Abelian case (exact methods exist)")
print("=" * 70)

# Additional verification for simple case
print(f"\nVerification for latitude loop (θ=π/3):")
theta_const = np.pi/3 * np.ones_like(s_array)
solid_angle = 2 * np.pi * (1 - np.cos(np.pi/3))
expected_berry = -solid_angle / 2
print(f"  Theoretical solid angle: {solid_angle:.6f} rad")
print(f"  Expected Berry phase: {expected_berry:.6f} rad")
print(f"  Computed geometric: {results[0]['geometric']:.6f} rad")
print(f"  Difference: {abs(expected_berry - results[0]['geometric']):.6f} rad")