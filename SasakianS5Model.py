import jax.numpy as jnp

class YourS5Model:
    """
    Standard Sasakian structure on S^5 ⊂ ℝ^6 ≅ ℂ^3.
    - J : the constant “complex‐structure” 6×6 matrix (block‐diag of three 2×2 J’s)
    - Reeb(x) = J @ x
    - φ_x(v) = Jv − ⟨Jx, v⟩ x
    - α(x,v) = ⟨Reeb(x), v⟩
    """

    def __init__(self):
        # build J = block_diag([ [0 -1;1 0], [0 -1;1 0], [0 -1;1 0] ])
        J = jnp.zeros((6,6), dtype=jnp.float64)
        for i in range(3):
            a = 2*i
            J = J.at[a,   a+1].set(-1.0)
            J = J.at[a+1, a].set( 1.0)
        self.J = J

    def Reeb(self, x):
        """ξ_x = J x"""
        return self.J @ x

    def phi(self, x, v):
        """
        φ_x(v) = J v - α(x,v) x,
        where α(x,v) = ⟨Jx, v⟩
        """
        alpha = jnp.dot(self.Reeb(x), v)
        return self.J @ v - alpha * x

    def alpha(self, x, v):
        """Contact form α(x,v) = ⟨Jx, v⟩."""
        return float(jnp.dot(self.Reeb(x), v))
