import jax.numpy as jnp
import jax
from jax import config as jax_config
import numpy as np

# enable full double‐precision
jax_config.update("jax_enable_x64", True)

class SasakianValidatorS5:
    """
    Validation suite for the standard Sasakian S^5 implementation.
    Expects your model to have:
      - model.J        : 6×6 skew‐symmetric complex‐structure matrix
      - model.Reeb(x)  : returns ξ_x in ℝ^6
      - model.phi(x,v) : returns φ_x(v) in ℝ^6
      - model.alpha(x,v): returns α(x,v) = ⟨ξ_x, v⟩
    """

    def __init__(self, model):
        self.model = model

    def normalize(self, x):
        x = jnp.asarray(x, dtype=jnp.float64)
        norm = jnp.linalg.norm(x)
        return jnp.where(norm < 1e-12,
                         jnp.array([1.0,0,0,0,0,0], dtype=jnp.float64),
                         x / norm)

    def check_normalization(self, x, tol=1e-6):
        nsq = float(jnp.sum(x**2))
        return abs(nsq - 1.0) < tol, nsq

    def test_contact_form(self, n=20, tol=1e-8):
        key = jax.random.PRNGKey(0)
        fails = []
        for i in range(n):
            key, k1, k2 = jax.random.split(key, 3)
            x = self.normalize(jax.random.normal(k1, (6,)))
            v = jax.random.normal(k2, (6,))
            α_theo = self.model.alpha(x, v)
            α_mat  = float(jnp.dot(self.model.phi(x, x), v))
            if abs(α_theo - α_mat) > tol:
                fails.append((i, α_theo, α_mat, abs(α_theo - α_mat)))
        return fails

    def test_phi_squared(self, tol=1e-8):
        # φ_x(φ_x(x)) ?= -x  since α(x,x)=0 for any x on S^5
        x = self.normalize([1,2,3,4,5,6])
        φx = self.model.phi(x, x)
        lhs = self.model.phi(x, φx)
        rhs = -x
        err = float(jnp.linalg.norm(lhs - rhs))
        return err, err < tol

    def test_metric(self, tol=1e-8):
        key = jax.random.PRNGKey(1)
        x = self.normalize([.3,.7,.2,.5,.1,.4])
        v = jax.random.normal(key, (6,))
        w = jax.random.normal(key, (6,))
        φv = self.model.phi(x, v)
        φw = self.model.phi(x, w)
        lhs = float(jnp.dot(φv, φw))
        αv = self.model.alpha(x, v)
        αw = self.model.alpha(x, w)
        rhs = float(jnp.dot(v, w) - αv * αw)
        return abs(lhs - rhs), abs(lhs - rhs) < tol

    def run(self):
        print("=== Sasakian S^5 Validation ===\n")

        # 1) J skew‐symmetry
        J = self.model.J
        skew = float(jnp.linalg.norm(J + J.T))
        print(f"1) ‖J + Jᵀ‖ = {skew:.2e}")

        # 2) Point normalization
        for name, p in [('Generic',[.4,.4,.4,.4,.1,.2]),
                        ('Random', np.random.rand(6)),
                        ('Zero',[0]*6)]:
            x = self.normalize(p)
            ok, nsq = self.check_normalization(x)
            print(f"2) {name}: ‖x‖² = {nsq:.6f} → {'OK' if ok else 'FAIL'}")

        # 3) Contact form vs φ
        cfails = self.test_contact_form()
        print(f"3) Contact‐form mismatches: {len(cfails)}")
        if cfails:
            i, αt, αm, diff = cfails[0]
            print(f"   First failure at test {i}: αᵗ={αt:.6e}, αᵐ={αm:.6e}, Δ={diff:.2e}")

        # 4) φ² identity
        err2, ok2 = self.test_phi_squared()
        print(f"4) φ²‐identity error = {err2:.2e} → {'OK' if ok2 else 'FAIL'}")

        # 5) Metric compatibility
        errm, okm = self.test_metric()
        print(f"5) Metric compatibility error = {errm:.2e} → {'OK' if okm else 'FAIL'}")

        return {
            'J_skew': skew,
            'contact_failures': len(cfails),
            'phi2_err': err2,
            'metric_err': errm
        }

if __name__ == '__main__':
    from SasakianS5Model import YourS5Model

    model     = YourS5Model()
    validator = SasakianValidatorS5(model)
    summary   = validator.run()

    print("\nValidation summary:", summary)
