import jax.numpy as jnp
from jax import jacfwd, jacrev
import numpy as np

class S7QuaternionicDiagnostics:
    def __init__(self):
        print("\n=== S⁷ Quaternionic Contact (3-Sasakian) Diagnostics ===")
        print("CORRECTED: Using proper 1-forms and quaternionic structure equations")
        print("Structure equations from quaternionic geometry:")
        print("  F₁ = dα₁ + 2α₂ ∧ α₃")
        print("  F₂ = dα₂ + 2α₃ ∧ α₁") 
        print("  F₃ = dα₃ + 2α₁ ∧ α₂")
        print("Expected: each k_i = 2.000000, trace = 6.000000")
        print()

    def normalize_to_S7(self, coords):
        """Normalize coordinates to S⁷"""
        return coords / jnp.linalg.norm(coords)

    def alpha1_contact_form(self, coords):
        """α₁: First contact 1-form on S⁷ ⊂ ℍ²"""
        x = self.normalize_to_S7(coords)
        x0, x1, x2, x3, x4, x5, x6, x7 = x
        r_sq = jnp.dot(x, x)  # Should be 1 on S⁷
        
        # From document: α₁ = (1/r²) * numerator
        # numerator = -x₀dx₁ + x₁dx₀ - x₂dx₃ + x₃dx₂ - x₄dx₅ + x₅dx₄ - x₆dx₇ + x₇dx₆
        coeffs = jnp.array([0, -x0, 0, -x2, 0, -x4, 0, -x6])  # coefficients for dx₀, dx₁, dx₂, ...
        coeffs = coeffs.at[0].set(-x1)    # dx₀ coefficient: x₁
        coeffs = coeffs.at[1].set(-x0)    # dx₁ coefficient: -x₀  
        coeffs = coeffs.at[2].set(x3)     # dx₂ coefficient: x₃
        coeffs = coeffs.at[3].set(-x2)    # dx₃ coefficient: -x₂
        coeffs = coeffs.at[4].set(x5)     # dx₄ coefficient: x₅
        coeffs = coeffs.at[5].set(-x4)    # dx₅ coefficient: -x₄
        coeffs = coeffs.at[6].set(x7)     # dx₆ coefficient: x₇
        coeffs = coeffs.at[7].set(-x6)    # dx₇ coefficient: -x₆
        
        return coeffs / r_sq

    def alpha2_contact_form(self, coords):
        """α₂: Second contact 1-form on S⁷ ⊂ ℍ²"""
        x = self.normalize_to_S7(coords)
        x0, x1, x2, x3, x4, x5, x6, x7 = x
        r_sq = jnp.dot(x, x)
        
        # From document: corresponds to j-component of quaternionic structure
        coeffs = jnp.array([-x2, -x3, x0, x1, -x6, -x7, x4, x5])
        return coeffs / r_sq

    def alpha3_contact_form(self, coords):
        """α₃: Third contact 1-form on S⁷ ⊂ ℍ²"""
        x = self.normalize_to_S7(coords)
        x0, x1, x2, x3, x4, x5, x6, x7 = x
        r_sq = jnp.dot(x, x)
        
        # From document: corresponds to k-component of quaternionic structure
        coeffs = jnp.array([-x3, x2, -x1, x0, -x7, x6, -x5, x4])
        return coeffs / r_sq

    def compute_exterior_derivative_1form(self, alpha_func, coords):
        """Compute exterior derivative of a 1-form using proper differential geometry"""
        x = self.normalize_to_S7(coords)
        
        # Get the 1-form coefficients
        alpha = alpha_func(coords)
        
        # Compute Jacobian of the 1-form coefficients
        J = jacfwd(alpha_func)(coords)
        
        # Exterior derivative: dα = ∂ᵢαⱼ - ∂ⱼαᵢ (antisymmetric part)
        return (J - J.T) / 2

    def compute_wedge_product_1forms(self, alpha_j, alpha_k):
        """Wedge product of two 1-forms"""
        # For 1-forms α and β, (α ∧ β)ᵢⱼ = αᵢβⱼ - αⱼβᵢ
        return jnp.outer(alpha_j, alpha_k) - jnp.outer(alpha_k, alpha_j)

    def compute_all_curvatures(self, coords):
        """Compute F₁, F₂, F₃ with CORRECT 3-Sasakian structure equations"""
        x = self.normalize_to_S7(coords)
        
        # Get contact 1-forms
        alpha1 = self.alpha1_contact_form(coords)
        alpha2 = self.alpha2_contact_form(coords)
        alpha3 = self.alpha3_contact_form(coords)
        
        # Compute exterior derivatives
        dalpha1 = self.compute_exterior_derivative_1form(self.alpha1_contact_form, coords)
        dalpha2 = self.compute_exterior_derivative_1form(self.alpha2_contact_form, coords)
        dalpha3 = self.compute_exterior_derivative_1form(self.alpha3_contact_form, coords)
        
        # Compute wedge products
        wedge23 = self.compute_wedge_product_1forms(alpha2, alpha3)
        wedge31 = self.compute_wedge_product_1forms(alpha3, alpha1)
        wedge12 = self.compute_wedge_product_1forms(alpha1, alpha2)
        
        # CORRECTED structure equations with factor 2
        F1 = dalpha1 + 2 * wedge23
        F2 = dalpha2 + 2 * wedge31
        F3 = dalpha3 + 2 * wedge12
        
        return F1, F2, F3

    def extract_sectional_curvatures(self, F1, F2, F3):
        """Extract sectional curvatures from the three 2-forms"""
        # From the documents: each Fᵢ should give curvature 6 on appropriate 2-planes
        def extract_planes(Fi):
            return (float(Fi[0,1]), float(Fi[2,3]), float(Fi[4,5]), float(Fi[6,7]))
        
        k1_planes = extract_planes(F1)
        k2_planes = extract_planes(F2)
        k3_planes = extract_planes(F3)
        
        # Average over the four quaternionic 2-planes
        k1 = sum(k1_planes) / 4
        k2 = sum(k2_planes) / 4
        k3 = sum(k3_planes) / 4
        
        return k1, k2, k3, k1_planes, k2_planes, k3_planes

    def verify_contact_structure(self, coords):
        """Verify orthogonality and normalization of contact forms"""
        alpha1 = self.alpha1_contact_form(coords)
        alpha2 = self.alpha2_contact_form(coords) 
        alpha3 = self.alpha3_contact_form(coords)
        
        norm1 = float(jnp.linalg.norm(alpha1))
        norm2 = float(jnp.linalg.norm(alpha2))
        norm3 = float(jnp.linalg.norm(alpha3))
        
        dot12 = float(jnp.dot(alpha1, alpha2))
        dot13 = float(jnp.dot(alpha1, alpha3))
        dot23 = float(jnp.dot(alpha2, alpha3))
        
        return norm1, norm2, norm3, dot12, dot13, dot23

    def diagnose_point(self, coords):
        print(f"Testing point: {coords}")
        x = self.normalize_to_S7(coords)
        print(f"Constraint check: ||x||²={jnp.dot(x,x):.6f}")
        
        # Contact structure verification
        norm1, norm2, norm3, dot12, dot13, dot23 = self.verify_contact_structure(coords)
        print(f"Contact form norms: ||α₁||={norm1:.6f}, ||α₂||={norm2:.6f}, ||α₃||={norm3:.6f}")
        print(f"Contact form orthogonality: α₁·α₂={dot12:.6f}, α₁·α₃={dot13:.6f}, α₂·α₃={dot23:.6f}")
        
        # Curvature computation
        F1, F2, F3 = self.compute_all_curvatures(coords)
        k1, k2, k3, k1_planes, k2_planes, k3_planes = self.extract_sectional_curvatures(F1, F2, F3)
        
        print(f"Sectional curvatures by 2-planes:")
        print(f"  F₁ (i-channel): {k1_planes} → avg = {k1:.6f}")
        print(f"  F₂ (j-channel): {k2_planes} → avg = {k2:.6f}")
        print(f"  F₃ (k-channel): {k3_planes} → avg = {k3:.6f}")
        
        ricci_trace = k1 + k2 + k3
        print(f"3-Sasakian Ricci trace: {ricci_trace:.6f}")
        print(f"Theoretical: each kᵢ = 2.000000, trace = 6.000000")
        print(f"Individual errors: |k₁-2|={abs(k1-2):.2e}, |k₂-2|={abs(k2-2):.2e}, |k₃-2|={abs(k3-2):.2e}")
        print(f"Trace error: {abs(ricci_trace - 6.0):.2e}")
        
        return k1, k2, k3, ricci_trace

# Test the corrected implementation
print("CORRECTED S⁷ QUATERNIONIC CONTACT DIAGNOSTICS")
print("=" * 60)

s7_diag = S7QuaternionicDiagnostics()

test_cases = [
    ("canonical point [1,0,0,0,0,0,0,0]", jnp.array([1., 0., 0., 0., 0., 0., 0., 0.])),
    ("quaternionic point [0,1,0,0,0,0,0,0]", jnp.array([0., 1., 0., 0., 0., 0., 0., 0.])),
    ("generic point", jnp.array([1., 1., 1., 1., 0., 0., 0., 0.]) / 2),
    ("symmetric point", jnp.ones(8) / jnp.sqrt(8))
]

for name, point in test_cases:
    print(f"\nTesting S⁷ {name}:")
    s7_diag.diagnose_point(point)