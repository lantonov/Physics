import jax
import jax.numpy as jnp
from jax import grad, hessian, jacfwd, jacrev, vmap, jit
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

# Enable 64-bit precision for numerical accuracy in curvature computations
jax.config.update("jax_enable_x64", True)

class SasakianS7DiagnosticSystem:
    """
    Diagnostic version of our 3-Sasakian geometry implementation.
    
    This class is designed specifically to diagnose the discrepancy between
    our theoretical predictions and numerical results. We'll examine each
    computational step in detail and compare it with hand calculations.
    """
    
    def __init__(self):
        """Initialize the diagnostic system."""
        self.diagnostic_data = {}
        print("Diagnostic System Initialized")
        print("This will help us understand the source of the discrepancy.")
        
    def contact_form_alpha1(self, x):
        """Contact form α₁ with detailed diagnostic output."""
        x0, x1, x2, x3, x4, x5, x6, x7 = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
        
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
        """Contact form α₂ with detailed diagnostic output."""
        x0, x1, x2, x3, x4, x5, x6, x7 = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
        
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
        """Contact form α₃ with detailed diagnostic output."""
        x0, x1, x2, x3, x4, x5, x6, x7 = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
        
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
        
    def hand_calculate_exterior_derivative_alpha1(self, x):
        """
        Hand calculation of dα₁ for comparison with automatic differentiation.
        
        This performs the exterior derivative calculation manually to verify
        our understanding of what dα₁ should be. We'll compute each component
        step by step and compare with the JAX automatic differentiation result.
        """
        print(f"\n--- HAND CALCULATION OF dα₁ ---")
        print(f"Point: {x}")
        
        # α₁ = -x₁dx₀ + x₀dx₁ - x₃dx₂ + x₂dx₃ - x₅dx₄ + x₄dx₅ - x₇dx₆ + x₆dx₇
        # 
        # For the exterior derivative dα₁, we need:
        # dα₁ = d(-x₁dx₀ + x₀dx₁ - x₃dx₂ + x₂dx₃ - x₅dx₄ + x₄dx₅ - x₇dx₆ + x₆dx₇)
        #
        # Using d(f·dg) = df ∧ dg (since d(dg) = 0), we get:
        # d(-x₁dx₀) = -dx₁ ∧ dx₀ = dx₀ ∧ dx₁
        # d(x₀dx₁) = dx₀ ∧ dx₁
        # So d(-x₁dx₀ + x₀dx₁) = dx₀ ∧ dx₁ + dx₀ ∧ dx₁ = 2dx₀ ∧ dx₁
        
        print("Computing each pair contribution:")
        
        # Pair (x₀, x₁): contributes 2dx₀ ∧ dx₁
        pair_01_contribution = 2
        print(f"  (x₀,x₁) pair: 2dx₀∧dx₁ → coefficient = {pair_01_contribution}")
        
        # Pair (x₂, x₃): contributes 2dx₂ ∧ dx₃
        pair_23_contribution = 2
        print(f"  (x₂,x₃) pair: 2dx₂∧dx₃ → coefficient = {pair_23_contribution}")
        
        # Pair (x₄, x₅): contributes 2dx₄ ∧ dx₅
        pair_45_contribution = 2   
        print(f"  (x₄,x₅) pair: 2dx₄∧dx₅ → coefficient = {pair_45_contribution}")
        
        # Pair (x₆, x₇): contributes 2dx₆ ∧ dx₇
        pair_67_contribution = 2
        print(f"  (x₆,x₇) pair: 2dx₆∧dx₇ → coefficient = {pair_67_contribution}")
        
        # Create the hand-calculated exterior derivative matrix
        dalpha1_hand = jnp.zeros((8, 8))
        dalpha1_hand = dalpha1_hand.at[0, 1].set(pair_01_contribution)
        dalpha1_hand = dalpha1_hand.at[1, 0].set(-pair_01_contribution)
        dalpha1_hand = dalpha1_hand.at[2, 3].set(pair_23_contribution)
        dalpha1_hand = dalpha1_hand.at[3, 2].set(-pair_23_contribution)
        dalpha1_hand = dalpha1_hand.at[4, 5].set(pair_45_contribution)
        dalpha1_hand = dalpha1_hand.at[5, 4].set(-pair_45_contribution)
        dalpha1_hand = dalpha1_hand.at[6, 7].set(pair_67_contribution)
        dalpha1_hand = dalpha1_hand.at[7, 6].set(-pair_67_contribution)
        
        print(f"Hand-calculated dα₁ matrix (non-zero elements):")
        print(f"  [0,1] = {dalpha1_hand[0,1]}, [1,0] = {dalpha1_hand[1,0]}")
        print(f"  [2,3] = {dalpha1_hand[2,3]}, [3,2] = {dalpha1_hand[3,2]}")
        print(f"  [4,5] = {dalpha1_hand[4,5]}, [5,4] = {dalpha1_hand[5,4]}")
        print(f"  [6,7] = {dalpha1_hand[6,7]}, [7,6] = {dalpha1_hand[7,6]}")
        
        return dalpha1_hand
        
    def hand_calculate_wedge_product_alpha2_alpha3(self, x):
        """
        Hand calculation of α₂ ∧ α₃ for comparison.
        
        This is where much of the complexity lies. The wedge product of two
        8-component 1-forms creates a 64-term expansion that needs to be
        carefully organized according to the quaternionic structure.
        """
        print(f"\n--- HAND CALCULATION OF α₂ ∧ α₃ ---")
        
        alpha2 = self.contact_form_alpha2(x)
        alpha3 = self.contact_form_alpha3(x)
        
        print(f"α₂ coefficients: {alpha2}")
        print(f"α₃ coefficients: {alpha3}")
        
        # For the wedge product α₂ ∧ α₃, we compute:
        # (α₂ᵢ dxⁱ) ∧ (α₃ⱼ dxⱼ) = α₂ᵢα₃ⱼ dxⁱ ∧ dxⱼ
        # This gives us a matrix where entry [i,j] = α₂ᵢα₃ⱼ - α₂ⱼα₃ᵢ
        
        wedge_matrix = jnp.outer(alpha2, alpha3) - jnp.outer(alpha3, alpha2)
        
        print(f"Key wedge product coefficients:")
        print(f"  [0,1] = {wedge_matrix[0,1]:.3f}")
        print(f"  [2,3] = {wedge_matrix[2,3]:.3f}")
        print(f"  [4,5] = {wedge_matrix[4,5]:.3f}")
        print(f"  [6,7] = {wedge_matrix[6,7]:.3f}")
        
        # Let's also check what this gives for the ω₁ pattern
        omega1_from_wedge = (wedge_matrix[0,1] + wedge_matrix[2,3] + wedge_matrix[4,5] + wedge_matrix[6,7]) / 4
        print(f"  ω₁ contribution from α₂∧α₃: {omega1_from_wedge:.3f}")
        
        return wedge_matrix
        
    def automatic_differentiation_dalpha1(self, x):
        """
        Compute dα₁ using JAX automatic differentiation for comparison.
        """
        print(f"\n--- AUTOMATIC DIFFERENTIATION OF dα₁ ---")
        
        # Use JAX to compute the Jacobian of α₁
        jacobian_alpha1 = jacfwd(self.contact_form_alpha1)(x)
        
        # Create the antisymmetric exterior derivative matrix
        dalpha1_auto = jacobian_alpha1 - jacobian_alpha1.T
        
        print(f"JAX-computed dα₁ matrix (non-zero elements):")
        print(f"  [0,1] = {dalpha1_auto[0,1]:.6f}, [1,0] = {dalpha1_auto[1,0]:.6f}")
        print(f"  [2,3] = {dalpha1_auto[2,3]:.6f}, [3,2] = {dalpha1_auto[3,2]:.6f}")
        print(f"  [4,5] = {dalpha1_auto[4,5]:.6f}, [5,4] = {dalpha1_auto[5,4]:.6f}")
        print(f"  [6,7] = {dalpha1_auto[6,7]:.6f}, [7,6] = {dalpha1_auto[7,6]:.6f}")
        
        return dalpha1_auto
        
    def compare_hand_vs_automatic(self, x):
        """
        Direct comparison between hand calculation and automatic differentiation.
        """
        print(f"\n" + "="*60)
        print(f"DIAGNOSTIC COMPARISON AT POINT {x[:4]}...")
        print(f"="*60)
        
        # Compute both versions
        dalpha1_hand = self.hand_calculate_exterior_derivative_alpha1(x)
        dalpha1_auto = self.automatic_differentiation_dalpha1(x)
        wedge_alpha2_alpha3 = self.hand_calculate_wedge_product_alpha2_alpha3(x)
        
        # Compare the exterior derivatives
        print(f"\n--- COMPARING dα₁ CALCULATIONS ---")
        diff_dalpha1 = dalpha1_auto - dalpha1_hand
        max_diff = jnp.max(jnp.abs(diff_dalpha1))
        print(f"Maximum difference between hand and auto dα₁: {max_diff:.2e}")
        
        if max_diff < 1e-10:
            print("✓ dα₁ calculations match perfectly!")
        else:
            print("✗ Significant difference in dα₁ calculations!")
            print("Non-zero differences:")
            for i in range(8):
                for j in range(8):
                    if abs(diff_dalpha1[i,j]) > 1e-10:
                        print(f"  [{i},{j}]: hand={dalpha1_hand[i,j]:.6f}, auto={dalpha1_auto[i,j]:.6f}, diff={diff_dalpha1[i,j]:.2e}")
        
        # Now compute F₁ = dα₁ + 2α₂∧α₃
        print(f"\n--- COMPUTING F₁ = dα₁ + 2α₂∧α₃ ---")
        F1_total = dalpha1_auto + 2 * wedge_alpha2_alpha3
        
        # Extract the horizontal coefficients using our pattern
        omega1_coeff = (F1_total[0,1] + F1_total[2,3] + F1_total[4,5] + F1_total[6,7]) / 4
        omega2_coeff = (F1_total[0,2] + F1_total[1,3] + F1_total[4,6] + F1_total[5,7]) / 4
        omega3_coeff = (F1_total[0,3] - F1_total[1,2] + F1_total[4,7] - F1_total[5,6]) / 4
        
        print(f"Final F₁ horizontal coefficients:")
        print(f"  ω₁: {omega1_coeff:.6f} (theory predicts 6.0)")
        print(f"  ω₂: {omega2_coeff:.6f} (theory predicts 6.0)")  
        print(f"  ω₃: {omega3_coeff:.6f} (theory predicts 6.0)")
        
        # Let's also break down the contributions
        dalpha1_omega1 = (dalpha1_auto[0,1] + dalpha1_auto[2,3] + dalpha1_auto[4,5] + dalpha1_auto[6,7]) / 4
        wedge_omega1 = (wedge_alpha2_alpha3[0,1] + wedge_alpha2_alpha3[2,3] + wedge_alpha2_alpha3[4,5] + wedge_alpha2_alpha3[6,7]) / 4
        
        print(f"\nBreaking down ω₁ contributions:")
        print(f"  From dα₁: {dalpha1_omega1:.6f}")
        print(f"  From 2α₂∧α₃: {2*wedge_omega1:.6f}")
        print(f"  Total: {dalpha1_omega1 + 2*wedge_omega1:.6f}")
        
        return {
            'dalpha1_hand': dalpha1_hand,
            'dalpha1_auto': dalpha1_auto,
            'wedge_alpha2_alpha3': wedge_alpha2_alpha3,
            'F1_total': F1_total,
            'omega_coeffs': [omega1_coeff, omega2_coeff, omega3_coeff]
        }
        
    def theoretical_expectation_analysis(self, x):
        """
        Analyze what our theoretical calculation actually predicts step by step.
        
        This helps us understand if our theoretical framework matches what
        we're computing numerically, or if there are hidden assumptions.
        """
        print(f"\n" + "="*60)
        print(f"THEORETICAL EXPECTATION ANALYSIS")
        print(f"="*60)
        
        print("Our theoretical prediction was based on:")
        print("1. Computing dα₁ on S⁷ gives -2(dx₀∧dx₁ + dx₂∧dx₃ + dx₄∧dx₅ + dx₆∧dx₇)")
        print("2. Computing α₂∧α₃ gives +4(dx₀∧dx₁ + dx₂∧dx₃ + dx₄∧dx₅ + dx₆∧dx₇)")
        print("3. Therefore F₁ = dα₁ + 2α₂∧α₃ = (-2 + 8)(dx₀∧dx₁ + ...) = 6ω₁")
        
        print(f"\nLet's check these predictions numerically at point {x[:4]}...")
        
        # Check our theoretical prediction for dα₁
        print("\nTheoretical dα₁ prediction check:")
        print("  According to theory: dα₁ should give coefficient -2 for each coordinate pair")
        
        # Our hand calculation should match this
        dalpha1_hand = self.hand_calculate_exterior_derivative_alpha1(x)
        actual_coeff_01 = dalpha1_hand[0,1]
        actual_coeff_23 = dalpha1_hand[2,3]
        actual_coeff_45 = dalpha1_hand[4,5]
        actual_coeff_67 = dalpha1_hand[6,7]
        
        print(f"  Actual coefficients: ({actual_coeff_01}, {actual_coeff_23}, {actual_coeff_45}, {actual_coeff_67})")
        print(f"  Theory predicted: (-2, -2, -2, -2)")
        
        # The issue might be here! Our hand calculation gives +2, but theory predicts -2
        if abs(actual_coeff_01 + 2) < 1e-6:
            print("  ✓ Signs match theoretical prediction")
        elif abs(actual_coeff_01 - 2) < 1e-6:
            print("  ⚠️  Sign discrepancy! Hand calculation gives +2, theory predicts -2")
            print("     This suggests a sign convention issue in our theoretical derivation")
        else:
            print(f"  ✗ Unexpected coefficient value: {actual_coeff_01}")
            
        # Check the wedge product prediction
        print(f"\nTheoretical α₂∧α₃ prediction check:")
        print("  According to theory: α₂∧α₃ should give coefficient +4 for ω₁ pattern")
        
        wedge_matrix = self.hand_calculate_wedge_product_alpha2_alpha3(x)
        wedge_omega1_coeff = (wedge_matrix[0,1] + wedge_matrix[2,3] + wedge_matrix[4,5] + wedge_matrix[6,7]) / 4
        
        print(f"  Actual α₂∧α₃ ω₁ coefficient: {wedge_omega1_coeff:.6f}")
        print(f"  Theory predicted: +4")
        
        if abs(wedge_omega1_coeff - 4) < 1e-6:
            print("  ✓ Wedge product matches theoretical prediction")
        else:
            print(f"  ⚠️  Wedge product discrepancy! Expected +4, got {wedge_omega1_coeff:.6f}")
            
        # Final combination check
        print(f"\nFinal F₁ coefficient prediction:")
        predicted_final = actual_coeff_01 + 2 * wedge_omega1_coeff
        print(f"  Hand calculation: {actual_coeff_01:.3f} + 2×{wedge_omega1_coeff:.3f} = {predicted_final:.3f}")
        print(f"  Theory predicted: -2 + 2×4 = 6")
        
        return {
            'dalpha1_coeffs': [actual_coeff_01, actual_coeff_23, actual_coeff_45, actual_coeff_67],
            'wedge_omega1_coeff': wedge_omega1_coeff,
            'predicted_final': predicted_final
        }
        
    def run_comprehensive_diagnosis(self):
        """
        Execute the complete diagnostic analysis to pinpoint the discrepancy source.
        """
        print("COMPREHENSIVE DIAGNOSTIC ANALYSIS")
        print("This will systematically identify the source of our discrepancy.")
        
        # Test points for diagnosis
        test_points = [
            jnp.array([1., 0., 0., 0., 0., 0., 0., 0.]),  # Simple test case
            jnp.array([0., 1., 0., 0., 0., 0., 0., 0.]),  # Different orientation
            jnp.array([1., 1., 1., 1., 0., 0., 0., 0.]) / 2.,  # More symmetric case
        ]
        
        all_results = []
        
        for i, point in enumerate(test_points):
            print(f"\n" + "="*80)
            print(f"DIAGNOSTIC TEST {i+1}: POINT {point[:4]}")
            print(f"="*80)
            
            # Run the comparison analysis
            comparison_result = self.compare_hand_vs_automatic(point)
            theoretical_result = self.theoretical_expectation_analysis(point)
            
            combined_result = {
                'point': point,
                'comparison': comparison_result,
                'theoretical': theoretical_result
            }
            
            all_results.append(combined_result)
            
        # Summary analysis
        print(f"\n" + "="*80)
        print(f"SUMMARY DIAGNOSIS")
        print(f"="*80)
        
        print("Key findings from our diagnostic analysis:")
        
        # Check if the issue is consistent across points
        omega1_values = [result['comparison']['omega_coeffs'][0] for result in all_results]
        print(f"ω₁ coefficients across test points: {[f'{val:.3f}' for val in omega1_values]}")
        
        if len(set([f'{val:.2f}' for val in omega1_values])) == 1:
            print("✓ Results are consistent across different points - suggests systematic issue")
        else:
            print("⚠️  Results vary across points - suggests computational instability")
            
        # Identify the most likely source
        avg_omega1 = jnp.mean(jnp.array(omega1_values))
        print(f"Average ω₁ coefficient: {avg_omega1:.3f} (theory: 6.0, ratio: {avg_omega1/6:.3f})")
        
        if abs(avg_omega1 + 6) < 1:
            print("DIAGNOSIS: Sign convention error - our result is approximately -6 instead of +6")
        elif abs(avg_omega1 - 6/4) < 0.5:
            print("DIAGNOSIS: Normalization factor error - missing factor of 4")
        elif abs(avg_omega1 - 6/2) < 0.5:
            print("DIAGNOSIS: Factor of 2 error - possibly in wedge product or exterior derivative")
        else:
            print("DIAGNOSIS: Complex combination of factors - requires deeper investigation")
            
        return all_results

# Run the comprehensive diagnosis
def main():
    """
    Main diagnostic function to systematically identify the discrepancy source.
    """
    print("3-SASAKIAN S^7 DIAGNOSTIC SYSTEM")
    print("="*50)
    print("This will systematically diagnose the discrepancy between theory and computation.")
    
    diagnostic_system = SasakianS7DiagnosticSystem()
    
    # Run the comprehensive analysis
    results = diagnostic_system.run_comprehensive_diagnosis()
    
    print(f"\nDiagnostic analysis complete!")
    print("The results should reveal exactly where our theoretical and numerical approaches diverge.")
    
    return diagnostic_system, results

if __name__ == "__main__":
    system, diagnostic_results = main()