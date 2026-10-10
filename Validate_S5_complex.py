# validate_s5_complex.py

import jax.numpy as jnp
import jax
from jax import config as jax_config
import numpy as np

jax_config.update("jax_enable_x64", True)

class ComplexS5Model:
    """Standard Sasakian on S^5 via the ℂ³ Hopf fibration."""
    def __init__(self):
        # Three copies of the 2×2 complex‐structure J
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
        return self.J @ v - α * x


class SasakianValidatorS5:
    def __init__(self, model):
        self.m = model

    def normalize(self, x):
        x = jnp.asarray(x, dtype=jnp.float64)
        n = jnp.linalg.norm(x)
        return jnp.where(n<1e-12,
                         jnp.array([1,0,0,0,0,0], dtype=jnp.float64),
                         x/n)

    def check_norm(self, x, tol=1e-6):
        nsq = float(jnp.sum(x**2))
        return abs(nsq-1.0)<tol, nsq

    def test_contact(self, n=20, tol=1e-8):
        key = jax.random.PRNGKey(0)
        fails = []
        for i in range(n):
            key, k1, k2 = jax.random.split(key, 3)
            x = self.normalize(jax.random.normal(k1, (6,)))
            v = jax.random.normal(k2, (6,))
            αt = self.m.alpha(x, v)
            αm = float(jnp.dot(self.m.reeb(x), v))
            if abs(αt-αm) > tol:
                fails.append((i, αt, αm))
        return fails

    def test_phi2(self, tol=1e-8):
        x = self.normalize([1,2,3,4,5,6])
        φx = self.m.phi(x, x)
        lhs = self.m.phi(x, φx)
        # φ² = -I + α⊗ξ, but α(x,x)=0 so φ²(x) = -x
        rhs = -x
        err = float(jnp.linalg.norm(lhs - rhs))
        return err, err<tol

    def test_metric(self, tol=1e-8):
        key = jax.random.PRNGKey(1)
        x = self.normalize([.3,.7,.2,.5,.1,.4])
        v = jax.random.normal(key, (6,))
        w = jax.random.normal(key, (6,))
        φv = self.m.phi(x, v)
        φw = self.m.phi(x, w)
        lhs = float(jnp.dot(φv, φw))
        αv = self.m.alpha(x, v)
        αw = self.m.alpha(x, w)
        rhs = float(jnp.dot(v, w) - αv*αw)
        return abs(lhs-rhs), abs(lhs-rhs)<tol

    def run(self):
        print("=== Sasakian S^5 (Complex) Validation ===\n")

        # 1) J skew‐symmetry
        skew = float(jnp.linalg.norm(self.m.J + self.m.J.T))
        print(f"1) ‖J+Jᵀ‖ = {skew:.2e}")

        # 2) normalization on S^5
        for name, p in [('Generic',[.4,.4,.4,.4,.1,.2]),
                        ('Random', np.random.rand(6)),
                        ('Zero',[0]*6)]:
            x = self.normalize(p)
            ok, nsq = self.check_norm(x)
            print(f"2) {name}: ‖x‖²={nsq:.6f} → {'OK' if ok else 'FAIL'}")

        # 3) contact‐form consistency
        cf = self.test_contact()
        print(f"3) contact mismatches: {len(cf)}")

        # 4) φ² identity
        e2,ok2 = self.test_phi2()
        print(f"4) φ² error = {e2:.2e} → {'OK' if ok2 else 'FAIL'}")

        # 5) metric compatibility
        em,okm = self.test_metric()
        print(f"5) metric error = {em:.2e} → {'OK' if okm else 'FAIL'}")

        return {
            'skew':skew,
            'contact_failures':len(cf),
            'phi2_err':e2,
            'metric_err':em
        }

if __name__=="__main__":
    model     = ComplexS5Model()
    validator = SasakianValidatorS5(model)
    summary   = validator.run()
    print("\nSummary:", summary)
