# validate_s5_fixed.py

import jax.numpy as jnp
import jax
from jax import config as jax_config
import numpy as np

jax_config.update("jax_enable_x64", True)

class ComplexS5Model:
    """Standard Sasakian on S^5 via the ℂ³ Hopf fibration."""
    def __init__(self):
        J2 = jnp.array([[0., -1.],
                        [1.,  0.]], dtype=jnp.float64)
        self.J = jnp.block([
            [J2,   jnp.zeros((2,2)), jnp.zeros((2,2))],
            [jnp.zeros((2,2)), J2,   jnp.zeros((2,2))],
            [jnp.zeros((2,2)), jnp.zeros((2,2)), J2]
        ])

    def reeb(self, x):
        return self.J @ x

    def alpha(self, x, v):
        return float(jnp.dot(self.J @ x, v))

    def phi(self, x, v):
        # ⚠️ corrected sign here: + α(x,v) * x
        α = self.alpha(x, v)
        return self.J @ v + α * x


class SasakianValidatorS5:
    def __init__(self, model):
        self.m = model

    def normalize(self, x):
        x = jnp.asarray(x, dtype=jnp.float64)
        n = jnp.linalg.norm(x)
        return jnp.where(n < 1e-12,
                         jnp.array([1,0,0,0,0,0], dtype=jnp.float64),
                         x / n)

    def project_tangent(self, x, v):
        # v_tan = v - <x,v> x
        return v - float(jnp.dot(x, v))*x

    def test_phi2_on_tangent(self, n=100):
        key = jax.random.PRNGKey(0)
        errs = []
        for _ in range(n):
            key, k1, k2 = jax.random.split(key, 3)
            x = self.normalize(jax.random.normal(k1, (6,)))
            v = self.project_tangent(x, jax.random.normal(k2, (6,)))

            lhs = self.m.phi(x, self.m.phi(x, v))
            α = self.m.alpha(x, v)
            rhs = -v + α * self.m.reeb(x)
            errs.append(float(jnp.linalg.norm(lhs - rhs)))
        print("φ² identity on TₓS⁵ max error:", max(errs))

    def test_metric_on_tangent(self, n=100):
        key = jax.random.PRNGKey(1)
        errs = []
        for _ in range(n):
            key, k1, k2, k3 = jax.random.split(key, 4)
            x = self.normalize(jax.random.normal(k1, (6,)))
            u = self.project_tangent(x, jax.random.normal(k2, (6,)))
            w = self.project_tangent(x, jax.random.normal(k3, (6,)))

            φu = self.m.phi(x, u)
            φw = self.m.phi(x, w)
            lhs = float(jnp.dot(φu, φw))
            αu = self.m.alpha(x, u)
            αw = self.m.alpha(x, w)
            rhs = float(jnp.dot(u, w) - αu * αw)
            errs.append(abs(lhs - rhs))
        print("Metric compatibility on TₓS⁵ max error:", max(errs))

    def run(self):
        print("=== Sasakian S⁵ Validation (Fixed φ) ===\n")

        skew = float(jnp.linalg.norm(self.m.J + self.m.J.T))
        print("1) ‖J + Jᵀ‖ =", f"{skew:.2e}")

        self.test_phi2_on_tangent()
        self.test_metric_on_tangent()


if __name__=="__main__":
    model     = ComplexS5Model()
    validator = SasakianValidatorS5(model)
    validator.run()
