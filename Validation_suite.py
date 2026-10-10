import jax.numpy as jnp
import jax
from jax import config as jax_config
import numpy as np

# enable full double-precision so your 1e-10 tolerances actually mean something
jax_config.update("jax_enable_x64", True)

class ThreeSasakianValidator:
    """
    Comprehensive validation suite for 3-Sasakian S^7 implementation
    """
    
    def __init__(self, model):
        """
        Initialize validator with a ThreeSasakianS7 model instance
        
        Parameters
        ----------
        model : ThreeSasakianS7
            The model instance to validate
        """
        self.model = model
        
    # ==================== NORMALIZATION UTILITIES ====================
    
    def normalize_to_S7(self, x):
        """Properly normalize points to lie on S^7"""
        # Force 64-bit so normalization is as accurate as possible
        x = jnp.asarray(x, dtype=jnp.float64)
        norm = jnp.linalg.norm(x)
        if norm < 1e-12:
            # Handle zero vector - return a standard point on S^7
            return jnp.array([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        return x / norm

    def verify_S7_constraint(self, x, tolerance=1e-6):
        """Verify point lies on S^7 within tolerance"""
        norm_squared = float(jnp.sum(x**2))  # Convert to float
        return abs(norm_squared - 1.0) < tolerance
    
    # ==================== CONTACT FORM VERIFICATION ====================
    
    def verify_contact_form_implementation(self, x, v, form_index):
        """
        Verify that F_i @ x gives the correct contact form α_i
        by comparing with explicit coordinate expressions from the documents
        """
        # From the documents, the contact forms should be:
        if form_index == 1:
            # α₁ = +x₀dx₁ − x₁dx₀  + x₂dx₃ − x₃dx₂  + x₄dx₅ − x₅dx₄  + x₆dx₇ − x₇dx₆
            alpha_theoretical = (
                x[0]*v[1] - x[1]*v[0]
              + x[2]*v[3] - x[3]*v[2]
              + x[4]*v[5] - x[5]*v[4]
              + x[6]*v[7] - x[7]*v[6]
            )
            F_matrix = self.model.F1
        elif form_index == 2:
            # α₂ = −x₂dx₀ + x₃dx₁  + x₀dx₂ − x₁dx₃ − x₆dx₄ + x₇dx₅  + x₄dx₆ − x₅dx₇
            alpha_theoretical = (
              - x[2]*v[0] + x[3]*v[1]
              + x[0]*v[2] - x[1]*v[3]
              - x[6]*v[4] + x[7]*v[5]
              + x[4]*v[6] - x[5]*v[7]
            )
            F_matrix = self.model.F2
        else:  # form_index == 3
            # α₃ = −x₃dx₀ − x₂dx₁  + x₁dx₂ + x₀dx₃ − x₇dx₄ − x₆dx₅  + x₅dx₆ + x₄dx₇
            alpha_theoretical = (
              - x[3]*v[0] - x[2]*v[1]
              + x[1]*v[2] + x[0]*v[3]
              - x[7]*v[4] - x[6]*v[5]
              + x[5]*v[6] + x[4]*v[7]
            )
            F_matrix = self.model.F3
        
        alpha_matrix = float(jnp.dot(F_matrix @ x, v))
        alpha_theoretical = float(alpha_theoretical)
        return abs(alpha_theoretical - alpha_matrix) < 1e-8, alpha_theoretical, alpha_matrix

#    def verify_contact_form_implementation(self, x, v, form_index):
#        # pick the right matrix
#        if form_index == 1:
#            F = self.model.F1
#        elif form_index == 2:
#            F = self.model.F2
#        else:
#            F = self.model.F3
#
#        # make sure x is on the sphere
#        x = self.normalize_to_S7(x)
#
#        # 1) compute F·x
#        Fx = F @ x
#        # 2) decompose and recombine to get theoretical
#        alpha_theoretical = float(sum(float(Fx[k]) * float(v[k]) for k in range(8)))
#        # 3) direct matrix‐dot
#        alpha_matrix = float(jnp.dot(Fx, v))
#
#        return abs(alpha_theoretical - alpha_matrix) < 1e-8, alpha_theoretical, alpha_matrix


#    def test_contact_forms(self, num_tests=20):  # Reduced for debugging
#        """Test contact form implementation on multiple random points and vectors"""
#        print("Testing Contact Form Implementation...")
#        print("-" * 40)
#        
#        failures = []
#        for test_idx in range(num_tests):
#            x = self.normalize_to_S7(jax.random.normal(size=8))
#            v = jax.random.normal(size=8)
#            
#            for form_idx in [1, 2, 3]:
#                is_valid, theoretical, matrix = self.verify_contact_form_implementation(x, v, form_idx)
#                if not is_valid:
#                    failures.append({
#                        'test': test_idx,
#                        'form': form_idx,
#                        'x': x,
#                        'v': v,
#                        'theoretical': theoretical,
#                        'matrix': matrix,
#                        'error': abs(theoretical - matrix)
#                    })
#        
#        if len(failures) == 0:
#            print(f"✓ All {num_tests * 3} contact form tests passed")
#            return True
#        else:
#            print(f"✗ {len(failures)} contact form tests failed out of {num_tests * 3}")
#            print("First few failures:")
#            for i, failure in enumerate(failures[:5]):  # Show more failures
#                print(f"  Test {failure['test']}, Form α_{failure['form']}:")
#                print(f"    Point x: {failure['x']}")
#                print(f"    Vector v: {failure['v']}")
#                print(f"    Theoretical: {failure['theoretical']:.10f}")
#                print(f"    Matrix:      {failure['matrix']:.10f}")
#                print(f"    Error:       {failure['error']:.2e}")
#                print()
#            return False

    def test_contact_forms(self, num_tests=20):
        """Test contact form implementation on multiple random points and vectors"""
        print("Testing Contact Form Implementation...")
        print("-" * 40)

        failures = []
        failures_by_form = {1: 0, 2: 0, 3: 0}
        key = jax.random.PRNGKey(0)                # initialize once
        for test_idx in range(num_tests):
            key, subkey1, subkey2 = jax.random.split(key, 3)
            # draw both x and v as JAX arrays of shape (8,)
            x_raw = jax.random.normal(subkey1, shape=(8,), dtype=jnp.float64)
            v     = jax.random.normal(subkey2, shape=(8,), dtype=jnp.float64)
            
            # normalize x
            x = self.normalize_to_S7(x_raw)
            
            for form_idx in [1, 2, 3]:
                is_valid, theoretical, matrix = \
                    self.verify_contact_form_implementation(x, v, form_idx)
                if not is_valid:
                    failures_by_form[form_idx] += 1
                    failures.append({
                        'test': test_idx,
                        'form': form_idx,
                        'x': x,
                        'v': v,
                        'theoretical': theoretical,
                        'matrix': matrix,
                        'error': abs(theoretical - matrix)
                    })

        print("Failures by form:", failures_by_form)

        if not failures:
            print(f"✓ All {num_tests * 3} contact form tests passed")
            return True
        else:
            print(f"✗ {len(failures)} contact form tests failed out of {num_tests * 3}")
            # … rest of your failure reporting …
            return False

    
    # ==================== MATRIX ANALYSIS ====================
    
    def analyze_matrix_structure(self):
        """Analyze the structure of the F matrices"""
        print("Matrix Structure Analysis:")
        print("=" * 40)
        
        for name, F in [("F1", self.model.F1), ("F2", self.model.F2), ("F3", self.model.F3)]:
            print(f"\n{name} matrix:")
            print(F)
            print(f"Shape: {F.shape}")
            print(f"Skew-symmetric check: ||F + F^T|| = {jnp.linalg.norm(F + F.T):.2e}")
            
            # Check if it's block diagonal
            top_right = F[:4, 4:]
            bottom_left = F[4:, :4]
            print(f"Block structure check:")
            print(f"  Top-right block max: {jnp.max(jnp.abs(top_right)):.2e}")
            print(f"  Bottom-left block max: {jnp.max(jnp.abs(bottom_left)):.2e}")
    
    # ==================== GEOMETRIC INTERPRETATION ====================
    
    def geometric_interpretation_check(self, x, verbose=True):
        """
        Provide detailed geometric interpretation of each computation step
        """
        x = self.normalize_to_S7(x)
        
        if verbose:
            print(f"Point on S^7: {x}")
            print(f"Verification: ||x||² = {float(jnp.sum(x**2)):.10f}")
        
        # Step 1: Contact forms as 1-forms
        a1, a2, a3 = self.model.F1 @ x, self.model.F2 @ x, self.model.F3 @ x
        
        if verbose:
            print(f"\nContact form vectors:")
            print(f"α₁ vector: {a1}")
            print(f"α₂ vector: {a2}")  
            print(f"α₃ vector: {a3}")
        
        # Step 2: Verify orthogonality to original point (radial direction)
        radial_orthogonality = [
            float(jnp.dot(a1, x)),
            float(jnp.dot(a2, x)), 
            float(jnp.dot(a3, x))
        ]
        
        if verbose:
            print(f"\nOrthogonality to radial direction:")
            print(f"⟨α₁, x⟩ = {radial_orthogonality[0]:.10f}")
            print(f"⟨α₂, x⟩ = {radial_orthogonality[1]:.10f}")
            print(f"⟨α₃, x⟩ = {radial_orthogonality[2]:.10f}")
        
        # Step 3: Verify quaternionic relations (mutual orthogonality)
        mutual_orthogonality = [
            float(jnp.dot(a1, a2)),
            float(jnp.dot(a1, a3)),
            float(jnp.dot(a2, a3))
        ]
        
        if verbose:
            print(f"\nQuaternionic structure verification:")
            print(f"⟨α₁, α₂⟩ = {mutual_orthogonality[0]:.10f} (should be 0)")
            print(f"⟨α₁, α₃⟩ = {mutual_orthogonality[1]:.10f} (should be 0)")
            print(f"⟨α₂, α₃⟩ = {mutual_orthogonality[2]:.10f} (should be 0)")
        
        # Step 4: Verify norms (should all be equal for 3-Sasakian)
        norms = [
            float(jnp.linalg.norm(a1)),
            float(jnp.linalg.norm(a2)),
            float(jnp.linalg.norm(a3))
        ]
        
        if verbose:
            print(f"\nContact form norms:")
            print(f"||α₁|| = {norms[0]:.6f}")
            print(f"||α₂|| = {norms[1]:.6f}")
            print(f"||α₃|| = {norms[2]:.6f}")
        
        # Return validation metrics
        return {
            'radial_orthogonality_max': max(abs(x) for x in radial_orthogonality),
            'mutual_orthogonality_max': max(abs(x) for x in mutual_orthogonality),
            'norm_variation': max(norms) - min(norms),
            'contact_vectors': (a1, a2, a3),
            'norms': norms
        }
    
    # ==================== HORIZONTAL DISTRIBUTION VERIFICATION ====================
    
    def verify_horizontal_distribution(self, x, verbose=True):
        """
        Verify that the horizontal distribution is computed correctly
        """
        x = self.normalize_to_S7(x)
        
        # Get Reeb vectors (should span 3-dimensional vertical space + radial)
        a1, a2, a3 = self.model.F1 @ x, self.model.F2 @ x, self.model.F3 @ x
        
        # Build vertical space [x, ξ₁, ξ₂, ξ₃] using Gram-Schmidt
        def gram_schmidt(v, basis):
            w = v - sum((float(jnp.dot(b, v))) * b for b in basis)
            norm = float(jnp.linalg.norm(w))
            return jnp.where(norm > 1e-12, w/norm, jnp.zeros_like(w))
        
        xi1 = a1 / jnp.linalg.norm(a1)
        xi2 = gram_schmidt(a2, [xi1])
        xi3 = gram_schmidt(a3, [xi1, xi2])
        
        vertical_basis = jnp.stack([x, xi1, xi2, xi3], axis=1)
        
        # Verify orthogonality
        gram_matrix = vertical_basis.T @ vertical_basis
        max_off_diagonal = float(jnp.max(jnp.abs(gram_matrix - jnp.eye(4))))
        
        if verbose:
            print("Vertical basis Gram matrix:")
            print(gram_matrix)
            print(f"Should be identity: max off-diagonal = {max_off_diagonal:.2e}")
        
        # Get horizontal space (should be 4-dimensional)
        Q, _ = jnp.linalg.qr(vertical_basis, mode='complete')
        horizontal_basis = Q[:, 4:]  # Last 4 columns
        
        if verbose:
            print(f"\nHorizontal distribution dimension: {horizontal_basis.shape[1]} (should be 4)")
        
        # Verify horizontal ⊥ vertical
        orthogonality_check = horizontal_basis.T @ vertical_basis
        max_orthogonality_error = float(jnp.max(jnp.abs(orthogonality_check)))
        
        if verbose:
            print(f"Horizontal ⊥ Vertical check: max = {max_orthogonality_error:.2e}")
        
        return {
            'horizontal_basis': horizontal_basis,
            'vertical_basis': vertical_basis,
            'gram_matrix_error': max_off_diagonal,
            'orthogonality_error': max_orthogonality_error
        }
    
    # ==================== CURVATURE VALIDATION ====================
    
    def validate_sectional_curvatures(self, x, verbose=True):
        """
        Validate the sectional curvature computation with geometric meaning
        """
        x = self.normalize_to_S7(x)
        dist_info = self.verify_horizontal_distribution(x, verbose=False)
        H = dist_info['horizontal_basis']
        
        if verbose:
            print("Sectional Curvature Analysis:")
            print("=" * 40)
        
        results = {}
        
        for i, (F, name) in enumerate([(self.model.F1, "F1"), (self.model.F2, "F2"), (self.model.F3, "F3")]):
            # Restrict F to horizontal distribution
            F_horizontal = H.T @ F @ H  # 4×4 skew-symmetric matrix
            
            # Check skew-symmetry
            skew_symmetry_error = float(jnp.linalg.norm(F_horizontal + F_horizontal.T))
            
            if verbose:
                print(f"\n{name} restricted to horizontal distribution:")
                print(f"Matrix shape: {F_horizontal.shape}")
                print(f"Skew-symmetry check: ||F + F^T|| = {skew_symmetry_error:.2e}")
            
            # Convert to symmetric matrix for eigenanalysis
            M = -F_horizontal @ F_horizontal
            eigenvals = jnp.linalg.eigvals(M)
            
            # Ensure real eigenvalues (should be real for symmetric matrix)
            eigenvals_real = jnp.real(eigenvals)  # Take real part only
            eigenvals_sorted = jnp.sort(eigenvals_real)[::-1]  # Descending order
            
            # Extract sectional curvatures (should come in pairs)
            sqrt_eigenvals = jnp.sqrt(jnp.maximum(eigenvals_sorted, 0))
            
            if verbose:
                print(f"Eigenvalues of -F^T F: {eigenvals_sorted}")
                print(f"Sectional curvatures: {sqrt_eigenvals}")
                print(f"Sum: {float(jnp.sum(sqrt_eigenvals)):.6f}")
                
                # Verify pairing (for quaternionic structure, should have repeated eigenvalues)
                print(f"Eigenvalue pairing check:")
                print(f"  |λ₁ - λ₂| = {float(abs(eigenvals_sorted[0] - eigenvals_sorted[1])):.2e}")
                print(f"  |λ₃ - λ₄| = {float(abs(eigenvals_sorted[2] - eigenvals_sorted[3])):.2e}")
            
            results[name] = {
                'skew_symmetry_error': skew_symmetry_error,
                'eigenvalues': eigenvals_sorted,
                'sectional_curvatures': sqrt_eigenvals,
                'curvature_sum': float(jnp.sum(sqrt_eigenvals)),
                'eigenvalue_pairing_errors': [
                    float(abs(eigenvals_sorted[0] - eigenvals_sorted[1])),
                    float(abs(eigenvals_sorted[2] - eigenvals_sorted[3]))
                ]
            }
        
        return results
    
    # ==================== COMPLETE VALIDATION SUITE ====================
    
    def run_complete_validation(self, custom_points=None):
        """Run all validation checks"""
        print("3-Sasakian S^7 Implementation Validation")
        print("=" * 50)
        
        # First analyze the matrix structure
        self.analyze_matrix_structure()
        
        # Test points (properly normalized)
        if custom_points is None:
            test_points = {
                'Generic': self.normalize_to_S7([0.5,0.5,0.5,0.5,0,0,0,0]),
                'Random': self.normalize_to_S7([0.34,0.45,0.69,0.79,0.87,0.98,0.34,0.2]),
                'Symmetric': self.normalize_to_S7(np.ones(8)),
                'Zeros': self.normalize_to_S7(np.zeros(8))
            }
        else:
            test_points = {name: self.normalize_to_S7(point) for name, point in custom_points.items()}
        
        # Verify all points are normalized
        print(f"\nTest point normalization check:")
        for name, x in test_points.items():
            norm_sq = float(jnp.sum(x**2))
            print(f"  {name}: ||x||² = {norm_sq:.10f}")
        
        # Global validation results
        validation_summary = {
            'contact_forms_valid': True,
            'all_points_valid': True,
            'point_results': {}
        }
        
        # Test contact forms once globally
        print("\n" + "="*20 + " GLOBAL TESTS " + "="*20)
        contact_forms_valid = self.test_contact_forms(num_tests=10)  # Reduced for debugging
        validation_summary['contact_forms_valid'] = contact_forms_valid
        
        # Test each point
        for name, x in test_points.items():
            print(f"\n{'='*15} {name.upper()} POINT {'='*15}")
            
            point_valid = True
            
            try:
                # 1. Verify normalization
                if not self.verify_S7_constraint(x):
                    print(f"✗ Point {name} not properly normalized to S^7")
                    point_valid = False
                    continue
                else:
                    print(f"✓ Point {name} properly lies on S^7")
                
                # 2. Geometric interpretation
                geom_results = self.geometric_interpretation_check(x, verbose=True)
                
                # Check geometric validity with relaxed tolerances
                if geom_results['radial_orthogonality_max'] > 1e-6:
                    print(f"✗ Poor radial orthogonality: {geom_results['radial_orthogonality_max']:.2e}")
                    point_valid = False
                else:
                    print(f"✓ Good radial orthogonality: {geom_results['radial_orthogonality_max']:.2e}")
                
                if geom_results['mutual_orthogonality_max'] > 1e-6:
                    print(f"✗ Poor mutual orthogonality: {geom_results['mutual_orthogonality_max']:.2e}")
                    point_valid = False
                else:
                    print(f"✓ Good mutual orthogonality: {geom_results['mutual_orthogonality_max']:.2e}")
                
                if geom_results['norm_variation'] > 1e-4:
                    print(f"✗ Large norm variation: {geom_results['norm_variation']:.2e}")
                    point_valid = False
                else:
                    print(f"✓ Good norm consistency: {geom_results['norm_variation']:.2e}")
                
                # 3. Validate horizontal distribution
                dist_results = self.verify_horizontal_distribution(x, verbose=True)
                
                if dist_results['gram_matrix_error'] > 1e-6:
                    print(f"✗ Poor orthogonalization: {dist_results['gram_matrix_error']:.2e}")
                    point_valid = False
                else:
                    print(f"✓ Good orthogonalization: {dist_results['gram_matrix_error']:.2e}")
                
                if dist_results['orthogonality_error'] > 1e-6:
                    print(f"✗ Poor horizontal-vertical orthogonality: {dist_results['orthogonality_error']:.2e}")
                    point_valid = False
                else:
                    print(f"✓ Good horizontal-vertical orthogonality: {dist_results['orthogonality_error']:.2e}")
                
                # 4. Validate curvatures
                curv_results = self.validate_sectional_curvatures(x, verbose=True)
                
                # Check curvature validity
                for fname, cdata in curv_results.items():
                    if cdata['skew_symmetry_error'] > 1e-6:
                        print(f"✗ {fname} not skew-symmetric: {cdata['skew_symmetry_error']:.2e}")
                        point_valid = False
                    else:
                        print(f"✓ {fname} is skew-symmetric: {cdata['skew_symmetry_error']:.2e}")
                    
                    # Check if curvatures are close to expected values (1.0 each)
                    expected_curvatures = jnp.array([1.0, 1.0, 1.0, 1.0])
                    curvature_error = float(jnp.max(jnp.abs(cdata['sectional_curvatures'] - expected_curvatures)))
                    if curvature_error > 1e-4:
                        print(f"✗ {fname} curvatures deviate from expected: max error {curvature_error:.2e}")
                        point_valid = False
                    else:
                        print(f"✓ {fname} curvatures match expected: max error {curvature_error:.2e}")
                
                if point_valid:
                    print(f"\n✓ All checks passed for {name} point")
                else:
                    print(f"\n✗ Some checks failed for {name} point")
                
                validation_summary['point_results'][name] = {
                    'valid': point_valid,
                    'geometry': geom_results,
                    'distribution': dist_results,
                    'curvatures': curv_results
                }
                
            except Exception as e:
                print(f"\n✗ Exception during validation of {name} point: {e}")
                import traceback
                traceback.print_exc()
                point_valid = False
                validation_summary['point_results'][name] = {'valid': False, 'error': str(e)}
            
            if not point_valid:
                validation_summary['all_points_valid'] = False
        
        # Final summary
        print(f"\n{'='*20} VALIDATION SUMMARY {'='*20}")
        print(f"Contact forms implementation: {'✓ PASS' if validation_summary['contact_forms_valid'] else '✗ FAIL'}")
        print(f"All test points: {'✓ PASS' if validation_summary['all_points_valid'] else '✗ FAIL'}")
        
        return validation_summary

def print_sign_matrix(matrix, name):
    sign_matrix = jnp.sign(matrix)
    print(f"\n{name} Sign Matrix:")
    for row in sign_matrix:
        print(" ".join(f"{int(x):+d}" for x in row))

# ==================== USAGE EXAMPLE ====================

if __name__ == '__main__':
    # Import your original model
    try:
        from Quaternionic_contact_flow_GS1 import ThreeSasakianS7
    except ImportError:
        print("Could not import ThreeSasakianS7. Please ensure the file is in the same directory.")
        exit(1)
    
    # Create model and validator
    model = ThreeSasakianS7()
    validator = ThreeSasakianValidator(model)
    
    # Run complete validation
    results = validator.run_complete_validation()

def print_sign_matrix(M, name):
    import jax.numpy as jnp
    signs = jnp.sign(M)
    print(f"\n{name} sign pattern (±1 entries):")
    for row in signs:
        # convert each element to int for a clean +1 / -1
        print(" ".join(f"{int(x):+d}" for x in row))

if __name__ == "__main__":
    from Quaternionic_contact_flow_GS1 import ThreeSasakianS7
    model = ThreeSasakianS7()
    validator = ThreeSasakianValidator(model)
    results = validator.run_complete_validation()
    
    # Now print out the sign‐patterns
    print_sign_matrix(model.F1, "F1")
    print_sign_matrix(model.F2, "F2")
    print_sign_matrix(model.F3, "F3")

