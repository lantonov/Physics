# validate_s5_full.py

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
        α = self.alpha(x, v)
        return self.J @ v + α * x  # the “+” is the Sasakian sign


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
        return v - float(jnp.dot(x, v)) * x

    def analyze_matrix(self):
        print("Matrix Structure Analysis:")
        print("="*40)
        skew = float(jnp.linalg.norm(self.m.J + self.m.J.T))
        print(f"J (6×6) skew‐symmetric check: ||J+Jᵀ|| = {skew:.2e}")
        return skew

    def test_normalization(self):
        print("\nTest point normalization:")
        for name, p in [
            ('Generic', [0.5,0.5,0.2,0.1,0.4,0.3]),
            ('Random', np.random.rand(6)),
            ('Zero',   [0,0,0,0,0,0])
        ]:
            x = self.normalize(p)
            nsq = float(jnp.sum(x**2))
            print(f"  {name}: ||x||² = {nsq:.10f} → {'OK' if abs(nsq-1)<1e-6 else 'FAIL'}")

    def test_phi2(self, num_tests=20):
        print("\nTesting φ² = −I + α⊗ξ on TₓS⁵...")
        failures = 0
        max_err = 0.0
        key = jax.random.PRNGKey(0)
        for _ in range(num_tests):
            key, k1, k2 = jax.random.split(key, 3)
            x = self.normalize(jax.random.normal(k1, (6,)))
            v = self.project_tangent(x, jax.random.normal(k2, (6,)))
            lhs = self.m.phi(x, self.m.phi(x, v))
            α = self.m.alpha(x, v)
            rhs = -v + α * self.m.reeb(x)
            err = float(jnp.linalg.norm(lhs - rhs))
            max_err = max(max_err, err)
            if err > 1e-8:
                failures += 1
        print(f"  max error = {max_err:.2e}, failures = {failures}/{num_tests}")
        return failures, max_err

    def test_metric(self, num_tests=20):
        print("\nTesting metric compatibility on TₓS⁵...")
        failures = 0
        max_err = 0.0
        key = jax.random.PRNGKey(1)
        for _ in range(num_tests):
            key, k1, k2, k3 = jax.random.split(key, 4)
            x = self.normalize(jax.random.normal(k1, (6,)))
            u = self.project_tangent(x, jax.random.normal(k2, (6,)))
            w = self.project_tangent(x, jax.random.normal(k3, (6,)))
            φu, φw = self.m.phi(x, u), self.m.phi(x, w)
            lhs = float(jnp.dot(φu, φw))
            rhs = float(jnp.dot(u, w) - self.m.alpha(x, u)*self.m.alpha(x, w))
            err = abs(lhs - rhs)
            max_err = max(max_err, err)
            if err > 1e-8:
                failures += 1
        print(f"  max error = {max_err:.2e}, failures = {failures}/{num_tests}")
        return failures, max_err

    def run_complete_validation(self):
        print("Sasakian S^5 Implementation Validation")
        print("="*50)
        skew_err = self.analyze_matrix()
        self.test_normalization()
        phi2_fails, phi2_err = self.test_phi2()
        met_fails, met_err = self.test_metric()

        # Finally, report the “twist count”
        # On S^5 you have a single complex structure acting on a 4-dim horizontal space → 2 twists
        print("\nTwist count:", "2 (i.e. 2 principal twists on the 4-dimensional horizontal distribution)")

        return {
            'skew_error': skew_err,
            'phi2_failures': phi2_fails,
            'phi2_max_error': phi2_err,
            'metric_failures': met_fails,
            'metric_max_error': met_err,
            'twists': 2
        }


if __name__ == '__main__':
    model     = ComplexS5Model()
    validator = SasakianValidatorS5(model)
    summary   = validator.run_complete_validation()
    print("\nValidation summary:", summary)
