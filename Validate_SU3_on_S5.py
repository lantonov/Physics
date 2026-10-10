import jax.numpy as jnp
import numpy as np
import jax
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

class SU3OnR6:
    """Real 6×6 representation of the 8 su(3) generators."""
    def __init__(self):
        # Define the eight Gell-Mann matrices λ_a in ℂ^{3×3}
        zero = lambda: jnp.zeros((3,3), dtype=jnp.complex128)
        λ = [zero() for _ in range(8)]
        # λ1
        λ[0] = jnp.array([[0,1,0],[1,0,0],[0,0,0]], dtype=jnp.complex128)
        # λ2
        λ[1] = jnp.array([[0,-1j,0],[1j,0,0],[0,0,0]], dtype=jnp.complex128)
        # λ3
        λ[2] = jnp.array([[1,0,0],[0,-1,0],[0,0,0]], dtype=jnp.complex128)
        # λ4
        λ[3] = jnp.array([[0,0,1],[0,0,0],[1,0,0]], dtype=jnp.complex128)
        # λ5
        λ[4] = jnp.array([[0,0,-1j],[0,0,0],[1j,0,0]], dtype=jnp.complex128)
        # λ6
        λ[5] = jnp.array([[0,0,0],[0,0,1],[0,1,0]], dtype=jnp.complex128)
        # λ7
        λ[6] = jnp.array([[0,0,0],[0,0,-1j],[0,1j,0]], dtype=jnp.complex128)
        # λ8
        λ[7] = (1/jnp.sqrt(3)) * jnp.array([
            [1,0,0],[0,1,0],[0,0,-2]
        ], dtype=jnp.complex128)

        # Build real 6×6 blocks for each anti-Hermitian generator T_a = i λ_a
        self.X = []
        for a in range(8):
            Ta = 1j * λ[a]  # anti-Hermitian, traceless
            Re = jnp.real(Ta)
            Im = jnp.imag(Ta)
            # real rep: [Re  -Im; Im  Re]
            M = jnp.block([[ Re, -Im],
                           [ Im,  Re]])
            self.X.append(M.astype(jnp.float64))

class SU3Validator:
    def __init__(self, model):
        self.m = model
        self.generators = model.X  # list of eight 6×6 real skew matrices

    def check_skew(self):
        errs = [float(jnp.linalg.norm(G + G.T)) for G in self.generators]
        print("Skew‐symmetry ||X_i+X_i^T||:", errs)
        return errs

    def check_tangency(self, num=20):
        key = jax.random.PRNGKey(0)
        violations = 0
        for _ in range(num):
            key, k = jax.random.split(key)
            x = jax.random.normal(k, (6,))
            x = x / jnp.linalg.norm(x)
            for G in self.generators:
                if abs(float(jnp.dot(x, G @ x))) > 1e-8:
                    violations += 1
        print(f"Tangency violations (should be 0): {violations}")
        return violations

    def check_closure(self, tol=1e-8):
        X = self.generators
        # form matrix whose columns are vec(X_k)
        mats = jnp.stack([G.reshape(-1) for G in X], axis=1)  # (36×8)
        # precompute QR for span
        Q, _ = jnp.linalg.qr(mats, mode='reduced')  # 36×8
        violations = []
        for i in range(8):
            for j in range(8):
                C = X[i] @ X[j] - X[j] @ X[i]  # commutator
                coeffs = Q.T @ C.reshape(-1)   # projection onto span
                recon = (Q @ coeffs).reshape(6,6)
                err = float(jnp.linalg.norm(C - recon))
                if err > tol:
                    violations.append(((i,j), err))
        print("Closure commutator violations:", violations)
        return violations

    def check_normalization(self):
        norms = []
        for i, G in enumerate(self.generators):
            inner = float(jnp.trace(G.T @ G))
            norms.append(inner)
        print("Generator norms tr(G^T G):", norms)
        return norms

    def run(self):
        print("=== SU(3) on ℝ⁶ Validator ===\n")
        self.check_skew()
        self.check_normalization()
        self.check_tangency()
        self.check_closure()
        print("\nAll checks passed if skew≈0, tangency=0, no closure violations.")

if __name__ == "__main__":
    model     = SU3OnR6()
    validator = SU3Validator(model)
    validator.run()
