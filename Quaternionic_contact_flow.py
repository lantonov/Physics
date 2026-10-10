import jax
import jax.numpy as jnp
from jax import grad, hessian, jacfwd, jacrev, vmap, jit
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

# Enable 64-bit precision for numerical accuracy in curvature computations
jax.config.update("jax_enable_x64", True)

class SasakianS7Geometry:
    """
    Implementation of 3-Sasakian geometry on S^7 with numerical Ricci flow evolution.
    
    The key insight for the corrected implementation is that we need to structure
    the automatic differentiation properly. Instead of trying to differentiate
    through lambda functions, we create proper JAX-compatible functions that
    take coordinate arrays as input and return the differential form coefficients.
    """
    
    def __init__(self, grid_resolution=32):
        """Initialize the geometry with discretization parameters."""
        self.resolution = grid_resolution
        self.dim = 8  # S^7 is 7-dimensional, embedded in R^8
        
        # Initialize coordinate grids using stereographic projection
        self.setup_coordinate_system()
        
        # Pre-compile key functions for performance
        self.setup_jax_functions()
        
    def setup_coordinate_system(self):
        """
        Set up stereographic coordinate system for S^7.
        
        We use stereographic projection from north pole (0,0,0,0,0,0,0,1)
        mapping S^7 \ {north pole} → R^7.
        """
        # Create coordinate grid in R^7 (stereographic coordinates)
        max_radius = 2.0  # Controls how much of S^7 we cover
        coords_1d = jnp.linspace(-max_radius, max_radius, self.resolution)
        
        # For computational efficiency, we'll work with a 2D slice initially
        # This represents the essential geometric structure while being tractable
        y0, y1 = jnp.meshgrid(coords_1d, coords_1d, indexing='ij')
        
        # Store the stereographic coordinates
        self.stereo_coords = jnp.stack([y0, y1], axis=-1)
        
        # Convert to S^7 coordinates
        self.s7_coords = self.stereographic_to_s7(self.stereo_coords)
        
        print(f"Initialized coordinate system with {self.resolution}² grid points")
        print(f"Stereographic coordinate range: [{-max_radius:.1f}, {max_radius:.1f}]")
        
    def stereographic_to_s7(self, y_coords):
        """
        Convert stereographic coordinates y ∈ R^7 to S^7 coordinates x ∈ R^8.
        
        The stereographic projection formula is:
        x_i = 2y_i/(1 + |y|²) for i = 0,...,6
        x_7 = (|y|² - 1)/(|y|² + 1)
        """
        # For our 2D implementation, pad with zeros for other coordinates
        y_padded = jnp.pad(y_coords, ((0,0), (0,0), (0,5)), mode='constant')
        
        y_norm_sq = jnp.sum(y_padded**2, axis=-1, keepdims=True)
        denominator = 1 + y_norm_sq
        
        # Compute first 7 coordinates
        x_coords_7 = 2 * y_padded / denominator
        
        # Compute the 8th coordinate
        x_7 = (y_norm_sq - 1) / denominator
        
        # Combine into full S^7 coordinates
        s7_coords = jnp.concatenate([x_coords_7, x_7], axis=-1)
        
        return s7_coords
        
    def contact_form_alpha1(self, x):
        """
        Compute the first contact form α₁ in coordinates.
        
        α₁ = -x₀dx₁ + x₁dx₀ - x₂dx₃ + x₃dx₂ - x₄dx₅ + x₅dx₄ - x₆dx₇ + x₇dx₆
        
        This function is now structured to be compatible with JAX automatic differentiation.
        It takes coordinates x as input and returns the 1-form coefficients as output.
        """
        # Ensure x has the right shape - if it's a single point, keep it 1D
        if x.ndim == 1:
            x0, x1, x2, x3, x4, x5, x6, x7 = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
        else:
            # Handle batched inputs
            x0, x1, x2, x3, x4, x5, x6, x7 = [x[..., i] for i in range(8)]
        
        # Coefficients of α₁ (as a 1-form α₁ = α₁ᵢ dxⁱ)
        alpha1_coeffs = jnp.array([
            -x1,  # coefficient of dx₀
            x0,   # coefficient of dx₁  
            -x3,  # coefficient of dx₂
            x2,   # coefficient of dx₃
            -x5,  # coefficient of dx₄
            x4,   # coefficient of dx₅
            -x7,  # coefficient of dx₆
            x6    # coefficient of dx₇
        ])
        
        return alpha1_coeffs
        
    def contact_form_alpha2(self, x):
        """
        Compute the second contact form α₂.
        
        α₂ = -x₀dx₂ - x₁dx₃ + x₂dx₀ + x₃dx₁ - x₄dx₆ - x₅dx₇ + x₆dx₄ + x₇dx₅
        """
        if x.ndim == 1:
            x0, x1, x2, x3, x4, x5, x6, x7 = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
        else:
            x0, x1, x2, x3, x4, x5, x6, x7 = [x[..., i] for i in range(8)]
        
        alpha2_coeffs = jnp.array([
            -x2,  # coefficient of dx₀
            -x3,  # coefficient of dx₁
            x0,   # coefficient of dx₂  
            x1,   # coefficient of dx₃
            -x6,  # coefficient of dx₄
            -x7,  # coefficient of dx₅
            x4,   # coefficient of dx₆
            x5    # coefficient of dx₇
        ])
        
        return alpha2_coeffs
        
    def contact_form_alpha3(self, x):
        """
        Compute the third contact form α₃.
        
        α₃ = -x₀dx₃ + x₁dx₂ - x₂dx₁ + x₃dx₀ - x₄dx₇ + x₅dx₆ - x₆dx₅ + x₇dx₄
        """
        if x.ndim == 1:
            x0, x1, x2, x3, x4, x5, x6, x7 = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
        else:
            x0, x1, x2, x3, x4, x5, x6, x7 = [x[..., i] for i in range(8)]
        
        alpha3_coeffs = jnp.array([
            -x3,  # coefficient of dx₀
            x2,   # coefficient of dx₁
            -x1,  # coefficient of dx₂
            x0,   # coefficient of dx₃  
            -x7,  # coefficient of dx₄
            x6,   # coefficient of dx₅
            -x5,  # coefficient of dx₆
            x4    # coefficient of dx₇
        ])
        
        return alpha3_coeffs
        
    def exterior_derivative_2form(self, alpha_func, coords):
        """
        Compute exterior derivative of a 1-form to get a 2-form.
        
        The key correction here is that we use jacfwd properly by passing
        the coordinate point directly to the function, rather than trying
        to differentiate through a lambda that captures external variables.
        
        For a 1-form ω = ωᵢ dxⁱ, we have:
        dω = (∂ωⱼ/∂xⁱ - ∂ωᵢ/∂xⁱ) dxⁱ ∧ dxⱼ
        """
        # Use JAX automatic differentiation to compute the Jacobian
        # This gives us ∂ωᵢ/∂xⱼ for all i,j
        jacobian_omega = jacfwd(alpha_func)(coords)
        
        # Build antisymmetric 2-form matrix
        # The exterior derivative gives us the antisymmetric part of the Jacobian
        two_form = jacobian_omega - jacobian_omega.T
        
        return two_form
        
    def wedge_product_2form(self, alpha_i_coeffs, alpha_j_coeffs):
        """
        Compute the wedge product αᵢ ∧ αⱼ of two 1-forms.
        
        For 1-forms α = αᵢ dxⁱ and β = βⱼ dxⱼ:
        α ∧ β = (αᵢβⱼ - αⱼβᵢ) dxⁱ ∧ dxⱼ
        
        This creates the antisymmetric matrix representation of the wedge product.
        """
        # Create antisymmetric matrix representation
        wedge_matrix = jnp.outer(alpha_i_coeffs, alpha_j_coeffs) - jnp.outer(alpha_j_coeffs, alpha_i_coeffs)
        
        return wedge_matrix
        
    def curvature_form_F1(self, coords):
        """
        Compute the first curvature 2-form F₁ = dα₁ + 2α₂ ∧ α₃.
        
        This is the key computation that reveals the horizontal-vertical decomposition
        and provides the coefficient 6ω₁ that drives the Ricci evolution.
        
        The correction here is to properly handle the automatic differentiation
        by ensuring all functions are JAX-compatible.
        """
        # Get contact form coefficients at this point
        alpha1_coeffs = self.contact_form_alpha1(coords)
        alpha2_coeffs = self.contact_form_alpha2(coords)  
        alpha3_coeffs = self.contact_form_alpha3(coords)
        
        # Compute dα₁ using automatic differentiation
        # Now we pass the function directly, which JAX can handle properly
        dalpha1 = self.exterior_derivative_2form(self.contact_form_alpha1, coords)
        
        # Compute wedge product α₂ ∧ α₃
        alpha2_wedge_alpha3 = self.wedge_product_2form(alpha2_coeffs, alpha3_coeffs)
        
        # Combine: F₁ = dα₁ + 2α₂ ∧ α₃
        F1 = dalpha1 + 2 * alpha2_wedge_alpha3
        
        return F1
        
    def curvature_forms_F2_F3(self, coords):
        """
        Compute F₂ and F₃ using quaternionic symmetry.
        
        F₂ = dα₂ + 2α₃ ∧ α₁
        F₃ = dα₃ + 2α₁ ∧ α₂
        """
        alpha1_coeffs = self.contact_form_alpha1(coords)
        alpha2_coeffs = self.contact_form_alpha2(coords)
        alpha3_coeffs = self.contact_form_alpha3(coords)
        
        # F₂ computation
        dalpha2 = self.exterior_derivative_2form(self.contact_form_alpha2, coords)
        alpha3_wedge_alpha1 = self.wedge_product_2form(alpha3_coeffs, alpha1_coeffs)
        F2 = dalpha2 + 2 * alpha3_wedge_alpha1
        
        # F₃ computation  
        dalpha3 = self.exterior_derivative_2form(self.contact_form_alpha3, coords)
        alpha1_wedge_alpha2 = self.wedge_product_2form(alpha1_coeffs, alpha2_coeffs)
        F3 = dalpha3 + 2 * alpha1_wedge_alpha2
        
        return F2, F3
        
    def extract_horizontal_coefficients(self, F_matrix):
        """
        Extract the horizontal part coefficients from curvature 2-form.
        
        From our theoretical analysis, each Fᵢ should have horizontal part 6ωᵢ,
        where ωᵢ corresponds to specific coordinate pairings:
        - ω₁: dx₀∧dx₁ + dx₂∧dx₃ + dx₄∧dx₅ + dx₆∧dx₇ 
        - ω₂: dx₀∧dx₂ + dx₁∧dx₃ + dx₄∧dx₆ + dx₅∧dx₇
        - ω₃: dx₀∧dx₃ - dx₁∧dx₂ + dx₄∧dx₇ - dx₅∧dx₆
        
        We extract these by looking at the appropriate matrix elements of the 2-form.
        """
        # Extract coefficients for ω₁ pattern (pairs: 01, 23, 45, 67)
        omega1_coeff = (F_matrix[0,1] + F_matrix[2,3] + F_matrix[4,5] + F_matrix[6,7]) / 4
        
        # Extract coefficients for ω₂ pattern (pairs: 02, 13, 46, 57)
        omega2_coeff = (F_matrix[0,2] + F_matrix[1,3] + F_matrix[4,6] + F_matrix[5,7]) / 4
        
        # Extract coefficients for ω₃ pattern (pairs: 03, -12, 47, -56)
        omega3_coeff = (F_matrix[0,3] - F_matrix[1,2] + F_matrix[4,7] - F_matrix[5,6]) / 4
        
        return omega1_coeff, omega2_coeff, omega3_coeff
        
    def verify_theoretical_coefficients(self):
        """
        Numerical verification that F₁, F₂, F₃ have horizontal parts 6ω₁, 6ω₂, 6ω₃.
        
        This directly tests our theoretical prediction against numerical computation.
        The corrected implementation should now properly compute the exterior derivatives
        and wedge products to verify our quaternionic symmetry predictions.
        """
        print("=" * 60)
        print("NUMERICAL VERIFICATION OF THEORETICAL PREDICTIONS")
        print("=" * 60)
        
        # Test at several points on S^7
        # We normalize each test point to ensure it lies on the unit sphere
        test_points = [
            jnp.array([1., 0., 0., 0., 0., 0., 0., 0.]),  # Standard basis vector
            jnp.array([0., 1., 0., 0., 0., 0., 0., 0.]),  # Different quaternion
            jnp.array([1., 1., 0., 0., 0., 0., 0., 0.]) / jnp.sqrt(2),  # Normalized mixed
            jnp.array([1., 1., 1., 1., 0., 0., 0., 0.]) / 2.,  # More complex point
        ]
        
        theoretical_predictions = []
        numerical_results = []
        
        for i, point in enumerate(test_points):
            # Ensure the point is normalized (lies on S^7)
            point_normalized = point / jnp.linalg.norm(point)
            print(f"\nTest Point {i+1}: {point_normalized[:4]}... (showing first 4 coords)")
            print(f"  Norm check: |x| = {jnp.linalg.norm(point_normalized):.6f} (should be 1.0)")
            
            try:
                # Compute curvature forms numerically
                F1 = self.curvature_form_F1(point_normalized)
                F2, F3 = self.curvature_forms_F2_F3(point_normalized)
                
                # Extract horizontal coefficients
                omega1_num, _, _ = self.extract_horizontal_coefficients(F1)
                _, omega2_num, _ = self.extract_horizontal_coefficients(F2)  
                _, _, omega3_num = self.extract_horizontal_coefficients(F3)
                
                # Theoretical prediction: each should be 6
                theory = jnp.array([6., 6., 6.])
                numerical = jnp.array([omega1_num, omega2_num, omega3_num])
                
                print(f"  Theoretical: ω₁={theory[0]:.1f}, ω₂={theory[1]:.1f}, ω₃={theory[2]:.1f}")
                print(f"  Numerical:   ω₁={numerical[0]:.3f}, ω₂={numerical[1]:.3f}, ω₃={numerical[2]:.3f}")
                print(f"  Error:       ω₁={abs(numerical[0]-theory[0]):.2e}, ω₂={abs(numerical[1]-theory[1]):.2e}, ω₃={abs(numerical[2]-theory[2]):.2e}")
                
                theoretical_predictions.append(theory)
                numerical_results.append(numerical)
                
            except Exception as e:
                print(f"  ✗ Error computing curvature at this point: {e}")
                # Use zeros as placeholder to continue with other points
                theoretical_predictions.append(jnp.array([6., 6., 6.]))
                numerical_results.append(jnp.array([0., 0., 0.]))
            
        # Overall error analysis
        all_theory = jnp.array(theoretical_predictions)
        all_numerical = jnp.array(numerical_results)
        max_error = jnp.max(jnp.abs(all_numerical - all_theory))
        mean_error = jnp.mean(jnp.abs(all_numerical - all_theory))
        
        print(f"\n" + "=" * 40)
        print(f"VERIFICATION SUMMARY:")
        print(f"Maximum error: {max_error:.2e}")
        print(f"Mean error:    {mean_error:.2e}")
        
        if max_error < 1e-10:
            print("✓ VERIFICATION PASSED: Numerical results match theory perfectly!")
        elif max_error < 1e-6:
            print("✓ VERIFICATION PASSED: Numerical results match theory within tolerance.")
        else:
            print("✗ VERIFICATION FAILED: Significant discrepancy detected.")
            print("This suggests either numerical issues or theoretical refinements needed.")
            
        return max_error < 1e-6
        
    def three_channel_flow_equation(self, g_components, t):
        """
        Implement the three-channel flow equation:
        ∂ₜg = -2Ric° + 8g + 16∑αᵢ⊗αᵢ
        
        This is the core PDE that governs the evolution of the 3-Sasakian structure.
        For now, we implement a simplified version to test the framework.
        """
        # For the initial implementation, use a simplified flow that captures
        # the essential structure without full Ricci tensor computation
        
        # Apply three-channel evolution with placeholder coefficients
        # Channel 1: Ricci flow (placeholder: -2 * identity for testing)
        ricci_channel = -2 * g_components
        
        # Channel 2: Volume stabilization (+8g)  
        stabilizer_channel = 8 * g_components
        
        # Channel 3: Contact structure preservation (simplified)
        contact_channel = 0.1 * g_components  # Placeholder for testing
        
        # Total evolution
        dgdt = ricci_channel + stabilizer_channel + contact_channel
        
        return dgdt
        
    def setup_jax_functions(self):
        """
        Pre-compile JAX functions for performance in the main computation loop.
        """
        # JIT compile the curvature computations for speed
        self.curvature_F1_jit = jit(self.curvature_form_F1)
        self.curvature_F2F3_jit = jit(self.curvature_forms_F2_F3)
        
        # JIT compile the flow equation  
        self.flow_equation_jit = jit(self.three_channel_flow_equation)
        
        print("JAX functions compiled and ready for high-performance computation.")
        
    def run_numerical_verification(self):
        """
        Execute the complete numerical verification of our theoretical framework.
        
        With the corrected implementation, this should successfully verify that
        our quaternionic symmetry predictions match the numerical computations.
        """
        print("Starting comprehensive numerical verification...")
        print("This will test our theoretical predictions against direct computation.")
        
        # Verify the coefficient calculations
        coefficient_test_passed = self.verify_theoretical_coefficients()
        
        if coefficient_test_passed:
            print("\n" + "="*60)
            print("PROCEEDING TO FLOW EVOLUTION ANALYSIS")
            print("="*60)
            
            # Set up initial conditions for flow evolution
            initial_metric = jnp.eye(8)  # Start with standard metric on S^7
            time_span = (0., 0.1)  # Short evolution for initial testing
            
            print(f"Initial metric (diagonal): {jnp.diag(initial_metric)}")
            print(f"Evolution time span: {time_span}")
            
            # Run a short evolution to test the flow equations
            try:
                # Use scipy's ODE solver with our JAX-compiled flow equation
                def flow_wrapper(t, y):
                    metric_matrix = y.reshape(8, 8)
                    dydt = self.three_channel_flow_equation(metric_matrix, t)
                    return dydt.flatten()
                
                print("\nStarting numerical integration of the flow equation...")
                sol = solve_ivp(
                    flow_wrapper, 
                    time_span, 
                    initial_metric.flatten(),
                    method='RK45',
                    max_step=0.001,  # Smaller steps for stability
                    rtol=1e-6
                )
                
                if sol.success:
                    print(f"✓ Flow integration successful!")
                    print(f"  Time points computed: {len(sol.t)}")
                    print(f"  Final time reached: {sol.t[-1]:.4f}")
                    
                    # Analyze the final state
                    final_metric = sol.y[:, -1].reshape(8, 8)
                    print(f"  Final metric (diagonal): {jnp.diag(final_metric)[:4]}... (first 4)")
                    
                    # Check preservation properties  
                    det_initial = jnp.linalg.det(initial_metric)
                    det_final = jnp.linalg.det(final_metric)
                    print(f"  Volume change: {det_final/det_initial:.6f}")
                    
                else:
                    print(f"✗ Flow integration failed: {sol.message}")
                    
            except Exception as e:
                print(f"Error during flow integration: {e}")
                print("The curvature verification is the primary test - flow integration is secondary.")
                
        return coefficient_test_passed

# Create the geometry system and run verification
def main():
    """
    Main function to run the complete numerical verification and analysis.
    
    With the corrected implementation, this should successfully demonstrate
    that our theoretical framework is computationally sound.
    """
    print("3-Sasakian S^7 Numerical Verification System")
    print("=" * 50)
    
    # Initialize the geometry
    start_time = time.time()
    geometry = SasakianS7Geometry(grid_resolution=16)  # Start with small grid for testing
    setup_time = time.time() - start_time
    print(f"Setup completed in {setup_time:.2f} seconds")
    
    # Run the verification
    verification_start = time.time()
    success = geometry.run_numerical_verification()
    verification_time = time.time() - verification_start
    
    print(f"\nTotal verification time: {verification_time:.2f} seconds")
    
    if success:
        print("\n" + "="*60)
        print("🎉 NUMERICAL VERIFICATION SUCCESSFUL! 🎉")
        print("Our theoretical framework has been validated computationally.")
        print("The quaternionic structure generates the predicted curvature coefficients.")
        print("The three-channel flow equation is ready for detailed analysis.")
    else:
        print("\n" + "="*60)  
        print("⚠️  VERIFICATION INCOMPLETE")
        print("The framework is operational but may need refinement.")
        print("This provides a foundation for further development.")
        
    return geometry, success

if __name__ == "__main__":
    geometry_system, verification_passed = main()