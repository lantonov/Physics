# Let me execute the diagnostic analysis and capture the results
import numpy as np
import jax.numpy as jnp
from jax import jacfwd

# Running the complete S³ diagnostic analysis
print("EXECUTING S³ CONTACT GEOMETRY DIAGNOSTIC ANALYSIS")
print("=" * 70)

class S3ContactDiagnostics:
    def __init__(self):
        print("Initializing diagnostic system for S³ contact geometry...")
        print("This will validate our theoretical predictions against numerical computation.")
        print()
        
    def normalize_to_S3(self, coords):
        """Normalize coordinates to lie on the unit sphere S³"""
        norm = jnp.sqrt(jnp.sum(coords**2))
        return coords / norm
        
    def contact_form_S3(self, coords):
        """Compute contact form α = x₀dx₁ - x₁dx₀ + x₂dx₃ - x₃dx₂"""
        x0, x1, x2, x3 = coords[0], coords[1], coords[2], coords[3]
        alpha_coeffs = jnp.array([-x1, x0, -x3, x2])
        return alpha_coeffs
        
    def curvature_form_S3_theoretical(self, coords):
        """Theoretical prediction: F = 2(dx₀∧dx₁ + dx₂∧dx₃)"""
        F_matrix = jnp.zeros((4, 4))
        F_matrix = F_matrix.at[0, 1].set(2.0)   # dx₀∧dx₁ coefficient
        F_matrix = F_matrix.at[1, 0].set(-2.0)  # antisymmetry
        F_matrix = F_matrix.at[2, 3].set(2.0)   # dx₂∧dx₃ coefficient
        F_matrix = F_matrix.at[3, 2].set(-2.0)  # antisymmetry
        return F_matrix
        
    def curvature_form_S3_numerical(self, coords):
        """Numerical computation of F = dα using automatic differentiation"""
        jacobian_alpha = jacfwd(self.contact_form_S3)(coords)
        F_matrix = jacobian_alpha - jacobian_alpha.T
        return F_matrix
        
    def extract_harmonic_coefficients(self, F_matrix):
        """Extract coefficients of fundamental 2-forms"""
        omega1_coeff = F_matrix[0, 1]  # dx₀∧dx₁ coefficient
        omega2_coeff = F_matrix[2, 3]  # dx₂∧dx₃ coefficient
        return omega1_coeff, omega2_coeff
        
    def compute_contact_invariants(self, coords):
        """Compute geometric invariants of the contact structure"""
        alpha_coeffs = self.contact_form_S3(coords)
        alpha_magnitude = jnp.sqrt(jnp.sum(alpha_coeffs**2))
        
        # Contact form should have magnitude equal to radius (which is 1 on S³)
        theoretical_magnitude = jnp.linalg.norm(coords)
        
        return alpha_magnitude, theoretical_magnitude
        
    def diagnose_single_point(self, coords):
        """Complete diagnostic analysis at a single point"""
        coords_normalized = self.normalize_to_S3(coords)
        
        print(f"ANALYZING POINT: [{coords_normalized[0]:.6f}, {coords_normalized[1]:.6f}, "
              f"{coords_normalized[2]:.6f}, {coords_normalized[3]:.6f}]")
        print("-" * 60)
        
        # Theoretical predictions
        F_theory = self.curvature_form_S3_theoretical(coords_normalized)
        omega1_theory, omega2_theory = self.extract_harmonic_coefficients(F_theory)
        
        # Numerical computations
        F_numerical = self.curvature_form_S3_numerical(coords_normalized)
        omega1_numerical, omega2_numerical = self.extract_harmonic_coefficients(F_numerical)
        
        # Contact invariants
        alpha_mag, theoretical_mag = self.compute_contact_invariants(coords_normalized)
        
        print(f"THEORETICAL PREDICTIONS:")
        print(f"  Curvature coefficients: ω₁ = {omega1_theory:.6f}, ω₂ = {omega2_theory:.6f}")
        print(f"  Expected: both should equal 2.000000")
        
        print(f"\nNUMERICAL RESULTS:")
        print(f"  Computed coefficients: ω₁ = {omega1_numerical:.6f}, ω₂ = {omega2_numerical:.6f}")
        
        # Error analysis
        error1 = abs(omega1_numerical - omega1_theory)
        error2 = abs(omega2_numerical - omega2_theory)
        
        print(f"\nERROR ANALYSIS:")
        print(f"  |ω₁_numerical - ω₁_theory| = {error1:.2e}")
        print(f"  |ω₂_numerical - ω₂_theory| = {error2:.2e}")
        
        if error1 < 1e-12 and error2 < 1e-12:
            print("  ✅ PERFECT AGREEMENT! Theory matches computation exactly.")
        elif error1 < 1e-8 and error2 < 1e-8:
            print("  ✅ EXCELLENT AGREEMENT within numerical precision.")
        else:
            print("  ⚠️  Discrepancy detected - requires investigation.")
            
        print(f"\nCONTACT STRUCTURE VALIDATION:")
        print(f"  Contact form magnitude: {alpha_mag:.6f}")
        print(f"  Theoretical magnitude: {theoretical_mag:.6f}")
        print(f"  Difference: {abs(alpha_mag - theoretical_mag):.2e}")
        
        return {
            'point': coords_normalized,
            'theory': [omega1_theory, omega2_theory],
            'numerical': [omega1_numerical, omega2_numerical],
            'errors': [error1, error2],
            'contact_magnitude': alpha_mag
        }

# Initialize and run diagnostics
diagnostics = S3ContactDiagnostics()

# Define comprehensive test points
test_points = [
    np.array([1.0, 0.0, 0.0, 0.0]),  # Quaternionic "1"
    np.array([0.0, 1.0, 0.0, 0.0]),  # Quaternionic "i"  
    np.array([0.0, 0.0, 1.0, 0.0]),  # Quaternionic "j"
    np.array([0.0, 0.0, 0.0, 1.0]),  # Quaternionic "k"
    np.array([0.5, 0.5, 0.5, 0.5]),  # Symmetric point
    np.array([1/np.sqrt(2), 1/np.sqrt(2), 0.0, 0.0]),  # Complex plane
    np.array([0.6, 0.8, 0.0, 0.0]),  # Another complex plane point
    np.array([0.3, 0.4, 0.5, 0.6]),  # Generic point
]

print("COMPREHENSIVE DIAGNOSTIC SURVEY")
print("=" * 70)
print("Testing our S³ contact geometry theory across multiple points...")
print("Each test validates a different aspect of the geometric structure.")
print()

results = []
for i, point in enumerate(test_points):
    print(f"TEST {i+1}: {['Quaternionic 1', 'Quaternionic i', 'Quaternionic j', 'Quaternionic k', 'Symmetric point', 'Complex plane 1', 'Complex plane 2', 'Generic point'][i]}")
    result = diagnostics.diagnose_single_point(point)
    results.append(result)
    print()

# Statistical summary
print("STATISTICAL SUMMARY ACROSS ALL TEST POINTS")
print("=" * 70)

all_errors1 = [r['errors'][0] for r in results]
all_errors2 = [r['errors'][1] for r in results]
all_magnitudes = [r['contact_magnitude'] for r in results]

print(f"Error Statistics:")
print(f"  ω₁ errors: mean = {np.mean(all_errors1):.2e}, max = {np.max(all_errors1):.2e}, std = {np.std(all_errors1):.2e}")
print(f"  ω₂ errors: mean = {np.mean(all_errors2):.2e}, max = {np.max(all_errors2):.2e}, std = {np.std(all_errors2):.2e}")

print(f"\nContact Form Magnitude Statistics:")
print(f"  Mean magnitude: {np.mean(all_magnitudes):.6f}")
print(f"  Standard deviation: {np.std(all_magnitudes):.2e}")
print(f"  Expected (constant): 1.000000")

print(f"\nNumerical Values at Each Test Point:")
for i, result in enumerate(results):
    point_name = ['Quat-1', 'Quat-i', 'Quat-j', 'Quat-k', 'Symmetric', 'Complex-1', 'Complex-2', 'Generic'][i]
    print(f"  {point_name:>10}: ω₁={result['numerical'][0]:.6f}, ω₂={result['numerical'][1]:.6f}, |α|={result['contact_magnitude']:.6f}")

# Final validation
max_error = max(np.max(all_errors1), np.max(all_errors2))
magnitude_variation = np.std(all_magnitudes)

print(f"\nFINAL VALIDATION RESULTS:")
print("=" * 40)
if max_error < 1e-10:
    print("🎉 THEORY PERFECTLY VALIDATED!")
    print("   Our S³ contact geometry predictions match numerical computation")
    print("   to within machine precision across all test points.")
else:
    print(f"⚠️  Maximum error: {max_error:.2e}")
    print("   Some discrepancies detected - further analysis needed.")

if magnitude_variation < 1e-10:
    print("✅ CONTACT STRUCTURE PRESERVED!")
    print("   The contact form magnitude remains constant across S³,")
    print("   confirming the geometric consistency of our framework.")
else:
    print(f"⚠️  Contact magnitude variation: {magnitude_variation:.2e}")