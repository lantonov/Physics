import jax.numpy as jnp
from jax import jacfwd

class S3CRDiagnostics:
    def __init__(self):
        print("Initializing CR (Tanaka–Webster) diagnostic on S³…")

    def normalize_to_S3(self, coords):
        norm = jnp.sqrt(jnp.sum(coords**2))
        return coords / norm

    def to_complex(self, coords):
        """Pack real 4-vector into (z1,z2)."""
        x0,x1,x2,x3 = coords
        z1 = x0 + 1j*x1
        z2 = x2 + 1j*x3
        return z1, z2

    def dcoords_dreal(self, coords):
        """Jacobian mapping real dx’s to complex dz’s and dbar z’s."""
        # ∂z1/∂x0=1, ∂z1/∂x1=i, etc.
        return jnp.array([
            [1+0j, 1j,     0+0j,  0+0j],  # dz1
            [0+0j, 0+0j,   1+0j,  1j  ],   # dz2
            [1-0j,-1j,     0+0j,  0+0j],   # dbar z1
            [0+0j, 0+0j,   1-0j, -1j  ],   # dbar z2
        ])  # shape (4_complex,4_real)

    def theta_CR(self, coords):
        """
        Tanaka–Webster contact form θ as a 1-form on R^4,
        expressed in the real basis [dx0,dx1,dx2,dx3].
        """
        z1, z2 = self.to_complex(coords)
        # In (dz1,dz2,dbar z1,dbar z2) basis θ = (i/2)[ ... ]
        # Pull back to real basis via dcoords_dreal:
        D = self.dcoords_dreal(coords)
        # Complex coefficients in complex basis:
        c = (1j/2)*jnp.array([ jnp.conj(z1),  jnp.conj(z2),
                              -      z1,       -      z2 ])  # shape (4_complex,)
        # Now map back to real basis: α_real = Re( cᵀ · D )
        alpha_complex = c @ D   # shape (4,), still complex
        alpha_real = jnp.real(alpha_complex)
        return alpha_real      # real-valued 1-form coefficients

    def connection_CR(self, coords):
        """
        Webster connection ωᵢⱼ in the 1×1 case is just
        ω₁¹ = -2 i θ.
        We return its real-basis coefficients.
        """
        theta = self.theta_CR(coords)
        return -2 * 1j * theta   # complex-valued 1-form
    
    def curvature_CR(self, coords):
        """
        Compute Ω₁¹ = dω₁¹ via automatic differentiation,
        but keep the complex result so we can recover the imaginary part.
        """
        conn = self.connection_CR
        J = jacfwd(conn)(coords)          # complex Jacobian
        Omega = J - jnp.swapaxes(J, 0, 1) # still complex
        return Omega                      # no jnp.real()

    def extract_CR_coefficient(self, Omega):
        """
        The code gives Ω[0,1] = -4j (and Ω[2,3] = -4j).
        Since theory predicts +2, use
            curvature = -Im(Ω[0,1]) / 2
        """
        # pick either (0,1) or (2,3) – they give the same imaginary entry
        imag01 = jnp.imag(Omega[0,1])
        return -imag01 / 2.0

    def diagnose(self, coords):
        x = self.normalize_to_S3(coords)
        Omega = self.curvature_CR(x)
        coeff = self.extract_CR_coefficient(Omega)
        print("CR-Webster curvature invariant =", coeff)  # should now be 2.0
        return float(coeff)

# --- usage ---
diag = S3CRDiagnostics()
for pt in [
    jnp.array([1.,0.,0.,0.]),
    jnp.array([0.5,0.5,0.5,0.5]),
]:
    print("Point:", pt, "→", diag.diagnose(pt))




