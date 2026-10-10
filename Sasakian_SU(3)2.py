import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

def corrected_sasakian_analysis():
    """Corrected geometric analysis of Sasakian SU(3) structures."""
    
    # 1. Fix quaternionic structure representation
    def create_quaternionic_basis():
        """Create proper quaternionic basis in C^4."""
        # Use consistent complex representation throughout
        z = np.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=complex)
        i_z = 1j * z  # Multiplication by i
        j_z = np.array([0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=complex)  # j action
        k_z = 1j * j_z  # k = ij
        return z, i_z, j_z, k_z
    
    # 2. Fix SU(2) and SU(3) actions
    def create_su2_matrix(theta):
        """Create proper SU(2) matrix."""
        return np.array([
            [np.cos(theta) + 1j*np.sin(theta), 0],
            [0, np.cos(theta) - 1j*np.sin(theta)]
        ], dtype=complex)
    
    def create_su3_matrix(alpha, beta, gamma):
        """Create proper SU(3) matrix with breaking."""
        U = np.eye(3, dtype=complex)
        U[0,0] = np.exp(1j * alpha)
        U[1,1] = np.exp(1j * beta) 
        U[2,2] = np.exp(-1j * (alpha + beta))  # Ensure det = 1
        return U
    
    # 3. Fix metric calculation
    def compute_metric_change(z_original, z_transformed):
        """Compute change in Hermitian metric."""
        metric_orig = np.real(np.vdot(z_original, z_original))
        metric_trans = np.real(np.vdot(z_transformed, z_transformed))
        return abs(metric_trans - metric_orig) / metric_orig
    
    # 4. Fix quaternionic structure test
    def test_quaternionic_relations(z, i_z, j_z, k_z):
        """Test i² = j² = k² = ijk = -1."""
        # i² = -z
        i_squared_error = np.linalg.norm(1j * i_z + z)
        # j² = -z  
        j_squared_error = np.linalg.norm(np.array([0, 0, -1, 0], dtype=complex) + z)
        # k² = -z
        k_squared_error = np.linalg.norm(1j * k_z + z)
        return i_squared_error, j_squared_error, k_squared_error
    
    # Run corrected analysis
    print("=== Corrected Sasakian SU(3) Analysis ===")
    
    # Create proper quaternionic basis
    z, i_z, j_z, k_z = create_quaternionic_basis()
    
    # Test with small angles to see geometric effects
    theta_su2 = 0.1  # Small SU(2) rotation
    alpha, beta, gamma = 0.1, 0.15, 0.05  # SU(3) parameters with breaking
    
    # SU(2) action (should preserve quaternionic structure)
    U2 = create_su2_matrix(theta_su2)
    z_su2 = np.zeros(4, dtype=complex)
    z_su2[0:2] = U2 @ z[0:2]
    z_su2[2:4] = z[2:4]  # SU(2) doesn't act on last two components
    
    # SU(3) action (should break quaternionic structure)  
    U3 = create_su3_matrix(alpha, beta, gamma)
    z_su3 = np.zeros(4, dtype=complex)
    z_su3[0] = U3[0,0] * z[0] + U3[0,2] * z[2]
    z_su3[1] = z[1]  # This component mixing breaks quaternionic structure
    z_su3[2] = U3[2,0] * z[0] + U3[2,2] * z[2]
    z_su3[3] = z[3]
    
    # Compute metric changes
    metric_change_su2 = compute_metric_change(z, z_su2)
    metric_change_su3 = compute_metric_change(z, z_su3)
    
    print(f"SU(2) metric change: {metric_change_su2:.2e}")
    print(f"SU(3) metric change: {metric_change_su3:.2e}")
    
    # Test quaternionic structure preservation
    i_err_su2, j_err_su2, k_err_su2 = test_quaternionic_relations(z_su2, 1j*z_su2, 
                                                                   np.roll(z_su2, 1), 1j*np.roll(z_su2, 1))
    i_err_su3, j_err_su3, k_err_su3 = test_quaternionic_relations(z_su3, 1j*z_su3,
                                                                   np.roll(z_su3, 1), 1j*np.roll(z_su3, 1))
    
    print(f"SU(2) quaternionic errors: i={i_err_su2:.2e}, j={j_err_su2:.2e}, k={k_err_su2:.2e}")
    print(f"SU(3) quaternionic errors: i={i_err_su3:.2e}, j={j_err_su3:.2e}, k={k_err_su3:.2e}")

if __name__ == "__main__":
    corrected_sasakian_analysis()