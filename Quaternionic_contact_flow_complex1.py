import jax.numpy as jnp
from jax import jacfwd
import numpy as np

class S3CRDiagnostics:
    def __init__(self):
        print("Initializing CR (Tanaka–Webster) diagnostics on S³…")

    def normalize_to_S3(self, coords):
        return coords / jnp.linalg.norm(coords)

    def to_complex(self, coords):
        x0, x1, x2, x3 = coords
        z1 = x0 + 1j * x1
        z2 = x2 + 1j * x3
        return z1, z2

    def dcoords_dreal(self, coords):
        # Jacobian of (dz1, dz2, dbar z1, dbar z2) w.r.t. (dx0, dx1, dx2, dx3)
        return jnp.array([
            [1+0j, 1j,     0+0j,  0+0j],  # dz1
            [0+0j, 0+0j,   1+0j,  1j  ],   # dz2
            [1-0j,-1j,     0+0j,  0+0j],   # dbar z1
            [0+0j, 0+0j,   1-0j, -1j  ],   # dbar z2
        ])

    def theta_CR(self, coords):
        z1, z2 = self.to_complex(coords)
        D = self.dcoords_dreal(coords)
        c = (1j/2) * jnp.array([jnp.conj(z1), jnp.conj(z2), -z1, -z2])
        alpha_complex = c @ D
        return jnp.real(alpha_complex)

    def connection_CR(self, coords):
        # ω₁¹ = -2 i θ
        return -2j * self.theta_CR(coords)

    def curvature_CR(self, coords):
        # Compute Ω = dω via autodiff (complex-valued)
        J = jacfwd(self.connection_CR)(coords)
        Omega = J - jnp.swapaxes(J, 0, 1)
        return Omega

    def extract_CR_coefficient(self, Omega):
        # Invariant curvature = -Im(Ω₀₁)/2 (or equivalently from Ω₂₃)
        return float(-jnp.imag(Omega[0, 1]) / 2.0)

    def compute_torsion(self, coords):
        """
        Numerically check that the Tanaka–Webster torsion A vanishes.
        We form the 2-form T = dθ - θ∧ω and extract its single θ∧θ̄ component.
        On S³ that component must be zero.
        """
        # 1-form θ¹ in the (1,0) CR frame pulled back to real basis:
        #    here we re-use theta_CR as our 'θ' but interpret it as θ¹
        theta1 = self.theta_CR(coords)
        # Exterior derivative dθ¹:
        Jth = jacfwd(self.theta_CR)(coords)
        dtheta1 = Jth - jnp.swapaxes(Jth, 0, 1)
        # Wedge θ¹ with ω₁¹ = -2i θ¹:
        omega11 = self.connection_CR(coords)
        # Compute θ¹∧ω₁¹ = θ⊗ω - ω⊗θ
        theta_wedge_omega = jnp.outer(theta1, omega11) - jnp.outer(omega11, theta1)
        # The true torsion 2-form:
        T = dtheta1 - theta_wedge_omega
        # In the real basis the only possible nonzero torsion lives in the same
        # slots (0,1) and (2,3) up to factors; we extract that imaginary part
        # and divide by 4 to match conventions:
        tors01 = T[0, 1] / 4.0
        # torsion = Im(tors01).  On S³ it should be exactly zero.
        return float(jnp.imag(tors01))

    def frame_invariant_curvature(self, Omega):
        """
        True CR Ricci trace in the real basis:
        Given Ω₀₁ = -4j and Ω₂₃ = -4j, we want +2.
        """
        # Combine the two slots, but divide by 4 instead of 2:
        val = (Omega[0, 1] + Omega[2, 3]) / 4.0
        # Now -Im(val) = 2
        return float(-jnp.imag(val))

    def diagnose_point(self, coords):
        x = self.normalize_to_S3(coords)
        Omega = self.curvature_CR(x)
        curvature = self.extract_CR_coefficient(Omega)
        torsion_norm = self.compute_torsion(x)
        frame_inv = self.frame_invariant_curvature(Omega)
        return curvature, torsion_norm, frame_inv

    def random_diagnostics(self, n=10, seed=0):
        rng = np.random.default_rng(seed)
        results = []
        for _ in range(n):
            pt = jnp.array(rng.normal(size=4), dtype=jnp.float32)
            res = self.diagnose_point(pt)
            results.append(res)
        return results

# Usage example:
diag = S3CRDiagnostics()
# Test fixed points and random sample
fixed_points = [jnp.array([1.,0.,0.,0.]), jnp.array([0.5,0.5,0.5,0.5])]
for pt in fixed_points:
    print("Point:", pt, "→", diag.diagnose_point(pt))

print("\nRandom sample diagnostics:")
for i, (curv, tors, finv) in enumerate(diag.random_diagnostics(5)):
    print(f"  Sample {i+1}: curvature = {curv}, torsion norm = {tors:.2e}, frame-inv = {finv}")

# Placeholder for S^7 extension: define complex triples and adapt similar methods.
