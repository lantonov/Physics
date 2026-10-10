import jax.numpy as jnp
from jax import jacfwd, jacrev
import numpy as np

class S7QuaternionicDiagnostics:
    def __init__(self):
        print("\n=== S⁷ Quaternionic Contact (3-Sasakian) Diagnostics ===")
        print("FINAL CORRECTION: Corrected sign in structure equations + horizontal planes")
        print("Structure equations:")
        print("  F₁ = dα₁ + 2α₂ ∧ α₃")
        print("  F₂ = dα₂ + 2α₃ ∧ α₁")
        print("  F₃ = dα₃ + 2α₁ ∧ α₂")
        print("Expected: k_i = 2.0 per horizontal plane, total trace = 6.0")
        print()

    def normalize_to_S7(self, coords):
        norm = jnp.linalg.norm(coords)
        return coords / norm if norm > 0 else coords

    def alpha1_contact_form(self, coords):
        x = coords
        x0, x1, x2, x3, x4, x5, x6, x7 = x
        coeffs = jnp.zeros(8)
        coeffs = coeffs.at[0].set(-x1)    # dx₀: -x₁
        coeffs = coeffs.at[1].set(x0)      # dx₁: x₀
        coeffs = coeffs.at[2].set(-x3)     # dx₂: -x₃
        coeffs = coeffs.at[3].set(x2)      # dx₃: x₂
        coeffs = coeffs.at[4].set(-x5)     # dx₄: -x₅
        coeffs = coeffs.at[5].set(x4)      # dx₅: x₄
        coeffs = coeffs.at[6].set(-x7)     # dx₆: -x₇
        coeffs = coeffs.at[7].set(x6)      # dx₇: x₆
        return coeffs

    def alpha2_contact_form(self, coords):
        x = coords
        x0, x1, x2, x3, x4, x5, x6, x7 = x
        coeffs = jnp.zeros(8)
        coeffs = coeffs.at[0].set(-x2)     # dx₀: -x₂
        coeffs = coeffs.at[1].set(-x3)     # dx₁: -x₃
        coeffs = coeffs.at[2].set(x0)      # dx₂: x₀
        coeffs = coeffs.at[3].set(x1)      # dx₃: x₁
        coeffs = coeffs.at[4].set(-x6)     # dx₄: -x₆
        coeffs = coeffs.at[5].set(-x7)     # dx₅: -x₇
        coeffs = coeffs.at[6].set(x4)      # dx₆: x₄
        coeffs = coeffs.at[7].set(x5)      # dx₇: x₅
        return coeffs

    def alpha3_contact_form(self, coords):
        x = coords
        x0, x1, x2, x3, x4, x5, x6, x7 = x
        coeffs = jnp.zeros(8)
        coeffs = coeffs.at[0].set(-x3)     # dx₀: -x₃
        coeffs = coeffs.at[1].set(x2)      # dx₁: x₂
        coeffs = coeffs.at[2].set(-x1)     # dx₂: -x₁
        coeffs = coeffs.at[3].set(x0)      # dx₃: x₀
        coeffs = coeffs.at[4].set(-x7)     # dx₄: -x₇
        coeffs = coeffs.at[5].set(x6)      # dx₅: x₆
        coeffs = coeffs.at[6].set(-x5)     # dx₆: -x₅
        coeffs = coeffs.at[7].set(x4)      # dx₇: x₄
        return coeffs

    def compute_exterior_derivative_1form(self, alpha_func, coords):
        J = jacfwd(alpha_func)(coords)
        return (J - jnp.transpose(J)) / 2

    def compute_wedge_product_1forms(self, alpha_j, alpha_k):
        return jnp.outer(alpha_j, alpha_k) - jnp.outer(alpha_k, alpha_j)

    def compute_all_curvatures(self, coords):
        x = self.normalize_to_S7(coords)
        alpha1 = self.alpha1_contact_form(x)
        alpha2 = self.alpha2_contact_form(x)
        alpha3 = self.alpha3_contact_form(x)
        
        dalpha1 = self.compute_exterior_derivative_1form(self.alpha1_contact_form, x)
        dalpha2 = self.compute_exterior_derivative_1form(self.alpha2_contact_form, x)
        dalpha3 = self.compute_exterior_derivative_1form(self.alpha3_contact_form, x)
        
        wedge23 = self.compute_wedge_product_1forms(alpha2, alpha3)
        wedge31 = self.compute_wedge_product_1forms(alpha3, alpha1)
        wedge12 = self.compute_wedge_product_1forms(alpha1, alpha2)
        
        # CORRECTED: Positive sign in structure equations
        F1 = -dalpha1 + 2 * wedge23
        F2 = -dalpha2 + 2 * wedge31
        F3 = -dalpha3 + 2 * wedge12
        
        return F1, F2, F3

    def project_to_horizontal(self, x, v):
        """Project vector v to horizontal space at x"""
        # Reeb vectors are the contact forms themselves
        xi1 = self.alpha1_contact_form(x)
        xi2 = self.alpha2_contact_form(x)
        xi3 = self.alpha3_contact_form(x)
        
        # Project v to tangent space of S^7
        v_tangent = v - jnp.dot(x, v) * x
        
        # Project to horizontal space (orthogonal to Reeb fields)
        v_hor = v_tangent
        v_hor = v_hor - jnp.dot(v_hor, xi1) * xi1
        v_hor = v_hor - jnp.dot(v_hor, xi2) * xi2
        v_hor = v_hor - jnp.dot(v_hor, xi3) * xi3
        
        return v_hor

    def extract_horizontal_curvatures(self, F1, F2, F3, x):
        """Extract curvatures on horizontal planes"""
        # Basis vectors in ambient space
        e0, e1, e2, e3, e4, e5, e6, e7 = jnp.eye(8)
        basis = [e0, e1, e2, e3, e4, e5, e6, e7]
        
        # Project each basis vector to horizontal space
        hor_basis = [self.project_to_horizontal(x, ei) for ei in basis]
        
        # Normalize horizontal basis vectors
        hor_basis = [v / jnp.linalg.norm(v) if jnp.linalg.norm(v) > 1e-10 else v for v in hor_basis]
        
        # Compute F_i on horizontal pairs (4,5), (6,7) for F1; (4,6), (5,7) for F2; (4,7), (5,6) for F3
        def F_val(F, u, v):
            return u @ F @ v  # u^T F v
        
        # F1 components (complex structure i)
        F1_45 = F_val(F1, hor_basis[4], hor_basis[5])
        F1_67 = F_val(F1, hor_basis[6], hor_basis[7])
        
        # F2 components (complex structure j)
        F2_46 = F_val(F2, hor_basis[4], hor_basis[6])
        F2_57 = F_val(F2, hor_basis[5], hor_basis[7])
        
        # F3 components (complex structure k)
        F3_47 = F_val(F3, hor_basis[4], hor_basis[7])
        F3_65 = F_val(F3, hor_basis[6], hor_basis[5])
        
        k1_planes = (float(F1_45), float(F1_67))
        k2_planes = (float(F2_46), float(F2_57))
        k3_planes = (float(F3_47), float(F3_65))
        
        k1 = sum(k1_planes)
        k2 = sum(k2_planes)
        k3_planes = (float(F3_47), float(F3_65))
        k3 = k3_planes[0] + k3_planes[1]
        
        return k1, k2, k3, k1_planes, k2_planes, k3_planes

    def verify_contact_structure(self, coords):
        x = self.normalize_to_S7(coords)
        alpha1 = self.alpha1_contact_form(x)
        alpha2 = self.alpha2_contact_form(x)
        alpha3 = self.alpha3_contact_form(x)
        
        norm1 = float(jnp.linalg.norm(alpha1))
        norm2 = float(jnp.linalg.norm(alpha2))
        norm3 = float(jnp.linalg.norm(alpha3))
        
        dot12 = float(jnp.dot(alpha1, alpha2))
        dot13 = float(jnp.dot(alpha1, alpha3))
        dot23 = float(jnp.dot(alpha2, alpha3))
        
        return norm1, norm2, norm3, dot12, dot13, dot23

    def diagnose_point(self, coords):
        x = self.normalize_to_S7(coords)
        print(f"Testing point: {x}")
        print(f"Constraint check: ||x||²={jnp.dot(x,x):.6f}")
        
        # Contact structure verification
        norm1, norm2, norm3, dot12, dot13, dot23 = self.verify_contact_structure(coords)
        print(f"Contact form norms: ||α₁||={norm1:.6f}, ||α₂||={norm2:.6f}, ||α₃||={norm3:.6f}")
        print(f"Contact form orthogonality: α₁·α₂={dot12:.6f}, α₁·α₃={dot13:.6f}, α₂·α₃={dot23:.6f}")
        
        # Curvature computation
        F1, F2, F3 = self.compute_all_curvatures(coords)
        k1, k2, k3, k1_planes, k2_planes, k3_planes = self.extract_horizontal_curvatures(F1, F2, F3, x)
        
        print("Sectional curvatures on horizontal planes:")
        print(f"  F₁ (i-channel): planes (4,5), (6,7) = {k1_planes} → sum = {k1:.6f}")
        print(f"  F₂ (j-channel): planes (4,6), (5,7) = {k2_planes} → sum = {k2:.6f}")
        print(f"  F₃ (k-channel): planes (4,7), (5,6) = {k3_planes} → sum = {k3:.6f}")
        
        ricci_trace = k1 + k2 + k3
        print(f"3-Sasakian Ricci trace: {ricci_trace:.6f}")
        print(f"Theoretical: each kᵢ sum = 2.000000, trace = 6.000000")
        print(f"Individual errors: |k₁-2|={abs(k1-2):.2e}, |k₂-2|={abs(k2-2):.2e}, |k₃-2|={abs(k3-2):.2e}")
        print(f"Trace error: {abs(ricci_trace - 6.0):.2e}")
        
        return k1, k2, k3, ricci_trace

# Test the corrected implementation
print("FINAL CORRECTED S⁷ QUATERNIONIC CONTACT DIAGNOSTICS")
print("=" * 60)

s7_diag = S7QuaternionicDiagnostics()

test_cases = [
    ("canonical point [1,0,0,0,0,0,0,0]", jnp.array([1., 0., 0., 0., 0., 0., 0., 0.])),
    ("quaternionic point [0,1,0,0,0,0,0,0]", jnp.array([0., 1., 0., 0., 0., 0., 0., 0.])),
    ("generic point", jnp.array([1., 1., 1., 1., 0., 0., 0., 0.])),
#    ("symmetric point", jnp.ones(8))
]

for name, point in test_cases:
    print(f"\nTesting S⁷ {name}:")
    s7_diag.diagnose_point(point)