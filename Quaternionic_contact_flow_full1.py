import jax.numpy as jnp
from jax import jacfwd
import numpy as np

class S3PseudohermitianDiagnostics:
    def __init__(self):
        print("=== S³ Pseudohermitian (CR) Diagnostics ===")
        print("Corrected implementation with proper complex geometry")
        print()

    def normalize_to_S3(self, coords):
        return coords / jnp.linalg.norm(coords)

    def to_complex(self, coords):
        x0, x1, x2, x3 = coords
        z1 = x0 + 1j * x1
        z2 = x2 + 1j * x3
        return z1, z2

    def theta_contact_form_real(self, coords):
        """Real part of contact form θ = (i/2)(z̄₁dz₁ - z₁dz̄₁ + z̄₂dz₂ - z₂dz̄₂)"""
        x0, x1, x2, x3 = coords
        # θ = x₁dx₀ - x₀dx₁ + x₃dx₂ - x₂dx₃ (real part of i/2 times the complex form)
        return jnp.array([x1, -x0, x3, -x2])

    def compute_dtheta(self, coords):
        """Exterior derivative dθ"""
        x = self.normalize_to_S3(coords)
        J = jacfwd(self.theta_contact_form_real)(x)
        return J - J.T

    def extract_webster_curvature(self, dtheta):
        """Extract scalar Webster curvature from dθ"""
        # For S³, dθ should equal 2(dx₀∧dx₁ + dx₂∧dx₃)
        # The Webster curvature is the coefficient of this fundamental 2-form
        curvature_01 = dtheta[0, 1]  # coefficient of dx₀∧dx₁
        curvature_23 = dtheta[2, 3]  # coefficient of dx₂∧dx₃
        
        # Both should be equal and give the Webster curvature
        return float((curvature_01 + curvature_23) / 2.0)

    def compute_levi_form(self, coords):
        """Compute Levi form L(X,Y) = dθ(X,JY) where J is complex structure"""
        dtheta = self.compute_dtheta(coords)
        # On S³, the Levi form should be positive definite with eigenvalue 1
        levi_eigenvalues = jnp.linalg.eigvals(dtheta[::2, ::2] + dtheta[1::2, 1::2])
        return jnp.real(levi_eigenvalues)

    def diagnose_point(self, coords):
        print(f"Testing point: {coords}")
        x = self.normalize_to_S3(coords)
        print(f"Normalized: {x}")
        
        z1, z2 = self.to_complex(x)
        print(f"Complex coords: z₁={z1:.4f}, z₂={z2:.4f}")
        print(f"Constraint check: |z₁|²+|z₂|²={abs(z1)**2 + abs(z2)**2:.6f}")
        
        # Contact form
        theta = self.theta_contact_form_real(x)
        print(f"Contact form θ coeffs: {theta}")
        
        # Exterior derivative
        dtheta = self.compute_dtheta(x)
        print(f"Exterior derivative dθ:")
        print(dtheta)
        
        # Webster curvature
        webster_curv = self.extract_webster_curvature(dtheta)
        print(f"Webster curvature:")
        print(f"  Numerical: {webster_curv:.6f}")
        print(f"  Theoretical: 2.000000")
        print(f"  Error: {abs(webster_curv - 2.0):.2e}")
        
        # Levi form
        levi_eigenvals = self.compute_levi_form(x)
        print(f"Levi form eigenvalues: {levi_eigenvals}")
        
        return webster_curv, levi_eigenvals

class S7QuaternionicDiagnostics:
    def __init__(self):
        print("\n=== S⁷ Quaternionic Contact (3-Sasakian) Diagnostics ===")
        print("Corrected implementation with proper quaternionic geometry")
        print()

    def normalize_to_S7(self, coords):
        return coords / jnp.linalg.norm(coords)

    def alpha1_contact_form(self, coords):
        """α₁ = -x₀dx₁ + x₁dx₀ - x₂dx₃ + x₃dx₂ - x₄dx₅ + x₅dx₄ - x₆dx₇ + x₇dx₆"""
        x0, x1, x2, x3, x4, x5, x6, x7 = coords
        return jnp.array([-x1, x0, -x3, x2, -x5, x4, -x7, x6])

    def alpha2_contact_form(self, coords):
        """α₂ = -x₀dx₂ - x₁dx₃ + x₂dx₀ + x₃dx₁ - x₄dx₆ - x₅dx₇ + x₆dx₄ + x₇dx₅"""
        x0, x1, x2, x3, x4, x5, x6, x7 = coords
        return jnp.array([-x2, -x3, x0, x1, -x6, -x7, x4, x5])

    def alpha3_contact_form(self, coords):
        """α₃ = -x₀dx₃ + x₁dx₂ - x₂dx₁ + x₃dx₀ - x₄dx₇ + x₅dx₆ - x₆dx₅ + x₇dx₄"""
        x0, x1, x2, x3, x4, x5, x6, x7 = coords
        return jnp.array([-x3, x2, -x1, x0, -x7, x6, -x5, x4])

    def compute_curvature_Fi(self, coords, i):
        """Compute Fᵢ = dαᵢ + 2∑ εᵢⱼₖ αⱼ ∧ αₖ"""
        x = self.normalize_to_S7(coords)
        
        if i == 0:  # F₁
            alpha = self.alpha1_contact_form
            alpha_j = self.alpha2_contact_form(x)
            alpha_k = self.alpha3_contact_form(x)
        elif i == 1:  # F₂  
            alpha = self.alpha2_contact_form
            alpha_j = self.alpha3_contact_form(x)
            alpha_k = self.alpha1_contact_form(x)
        else:  # F₃
            alpha = self.alpha3_contact_form
            alpha_j = self.alpha1_contact_form(x)
            alpha_k = self.alpha2_contact_form(x)
        
        # Compute dαᵢ
        J = jacfwd(alpha)(x)
        dalpha = J - J.T
        
        # Compute 2αⱼ ∧ αₖ
        wedge_jk = 2 * (jnp.outer(alpha_j, alpha_k) - jnp.outer(alpha_k, alpha_j))
        
        # Total curvature
        Fi = dalpha + wedge_jk
        return Fi

    def extract_horizontal_curvature(self, Fi):
        """Extract horizontal sectional curvature from Fᵢ"""
        # The horizontal part should give constant curvature 6 on round S⁷
        # Extract from the (0,1), (2,3), (4,5), (6,7) diagonal blocks
        curv_01 = Fi[0, 1]
        curv_23 = Fi[2, 3] 
        curv_45 = Fi[4, 5]
        curv_67 = Fi[6, 7]
        
        # Average the four 2-plane curvatures
        return float((curv_01 + curv_23 + curv_45 + curv_67) / 4.0)

    def compute_all_curvatures(self, coords):
        """Compute all three curvatures F₁, F₂, F₃"""
        F1 = self.compute_curvature_Fi(coords, 0)
        F2 = self.compute_curvature_Fi(coords, 1)
        F3 = self.compute_curvature_Fi(coords, 2)
        
        k1 = self.extract_horizontal_curvature(F1)
        k2 = self.extract_horizontal_curvature(F2)
        k3 = self.extract_horizontal_curvature(F3)
        
        return k1, k2, k3, F1, F2, F3

    def diagnose_point(self, coords):
        print(f"Testing point: {coords}")
        x = self.normalize_to_S7(coords)
        print(f"Normalized: {x}")
        print(f"Constraint check: ||x||²={jnp.dot(x,x):.6f}")
        
        # Contact forms
        alpha1 = self.alpha1_contact_form(x)
        alpha2 = self.alpha2_contact_form(x)  
        alpha3 = self.alpha3_contact_form(x)
        print(f"Contact form norms: ||α₁||={jnp.linalg.norm(alpha1):.4f}, "
              f"||α₂||={jnp.linalg.norm(alpha2):.4f}, ||α₃||={jnp.linalg.norm(alpha3):.4f}")
        
        # Check orthogonality of contact forms
        dot12 = jnp.dot(alpha1, alpha2)
        dot13 = jnp.dot(alpha1, alpha3)
        dot23 = jnp.dot(alpha2, alpha3)
        print(f"Contact form orthogonality: α₁·α₂={dot12:.4f}, α₁·α₃={dot13:.4f}, α₂·α₃={dot23:.4f}")
        
        # Curvatures
        k1, k2, k3, F1, F2, F3 = self.compute_all_curvatures(coords)
        ricci_trace = k1 + k2 + k3
        
        print(f"Sectional curvatures:")
        print(f"  k₁ (i-channel): {k1:.6f}")
        print(f"  k₂ (j-channel): {k2:.6f}")  
        print(f"  k₃ (k-channel): {k3:.6f}")
        print(f"  Ricci trace: {ricci_trace:.6f}")
        print(f"  Theoretical: each kᵢ ≈ 6.000000")
        print(f"  Theoretical trace: ~18.000000")
        print(f"  Trace error: {abs(ricci_trace - 18.0):.2e}")
        
        return k1, k2, k3, ricci_trace

# Run corrected diagnostics
print("CORRECTED PSEUDOHERMITIAN CONTACT GEOMETRY DIAGNOSTICS")
print("=" * 60)

# Test S³
s3_diag = S3PseudohermitianDiagnostics()

print("Testing S³ fixed point [1,0,0,0]:")
s3_diag.diagnose_point(jnp.array([1., 0., 0., 0.]))

print("\nTesting S³ symmetric point [0.5,0.5,0.5,0.5]:")
s3_diag.diagnose_point(jnp.array([0.5, 0.5, 0.5, 0.5]))

# Test S⁷
s7_diag = S7QuaternionicDiagnostics()

print("\nTesting S⁷ fixed point [1,0,0,0,0,0,0,0]:")
s7_diag.diagnose_point(jnp.array([1., 0., 0., 0., 0., 0., 0., 0.]))

print("\nTesting S⁷ symmetric point:")
symmetric_pt = jnp.ones(8) / jnp.sqrt(8)
s7_diag.diagnose_point(symmetric_pt)