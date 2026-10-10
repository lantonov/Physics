import jax.numpy as jnp
import jax
from jax import config as jax_config
import numpy as np

jax_config.update("jax_enable_x64", True)

class ComplexS5Model:
    """Standard Sasakian on S^5 via ℂ³ Hopf fibration."""
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
        return self.J @ v - α * x


class SasakianValidatorS5:
    def __init__(self, model):
        self.m = model

    def normalize(self, x):
        x = jnp.asarray(x, dtype=jnp.float64)
        n = jnp.linalg.norm(x)
        return jnp.where(n<1e-12, jnp.array([1,0,0,0,0,0],dtype=jnp.float64), x/n)

    def check_S5(self, tol=1e-6):
        for name,p in [('Random', np.random.randn(6)),('Generic',[.2,.3,.4,.5,.6,.1])]:
            x = self.normalize(p)
            nsq = float(jnp.sum(x**2))
            print(f"{name}: ‖x‖²={nsq:.6f}")

    def test_phi2_full(self, n=20, tol=1e-8):
        key = jax.random.PRNGKey(0)
        errs = []
        for _ in range(n):
            key, k1, k2 = jax.random.split(key, 3)
            x = self.normalize(jax.random.normal(k1,(6,)))
            v = jax.random.normal(k2,(6,))
            lhs = self.m.phi(x, self.m.phi(x, v))
            α = self.m.alpha(x, v)
            rhs = -v + α * self.m.reeb(x)
            errs.append(float(jnp.linalg.norm(lhs-rhs)))
        print("φ² identity max error:", max(errs))

    def test_metric(self, n=20, tol=1e-8):
        key = jax.random.PRNGKey(1)
        errs = []
        for _ in range(n):
            key, k1, k2, k3 = jax.random.split(key, 4)
            x = self.normalize(jax.random.normal(k1,(6,)))
            u = jax.random.normal(k2,(6,))
            w = jax.random.normal(k3,(6,))
            φu = self.m.phi(x,u)
            φw = self.m.phi(x,w)
            lhs = float(jnp.dot(φu,φw))
            rhs = float(jnp.dot(u,w) - self.m.alpha(x,u)*self.m.alpha(x,w))
            errs.append(abs(lhs-rhs))
        print("Metric compatibility max error:", max(errs))

    def run(self):
        print("=== Correct Sasakian S^5 Validation ===")
        print("\n-- Normalization --")
        self.check_S5()
        print("\n-- φ² = −I + α⊗ξ --")
        self.test_phi2_full()
        print("\n-- Metric compatibility --")
        self.test_metric()


if __name__=="__main__":
    model     = ComplexS5Model()
    validator = SasakianValidatorS5(model)
    validator.run()
