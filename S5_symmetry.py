import jax.numpy as jnp
import jax
from jax import config as jax_config
import numpy as np

# enable full double‐precision
jax_config.update("jax_enable_x64", True)

class SasakianValidatorS5:
    """
    Validation suite for the standard Sasakian S^5 implementation.
    Assumes your model provides:
      - self.F : 6×6 skew‐symmetric matrix encoding φ (the almost‐complex structure)
      - self.Reeb : vector field ξ of length 6 (Reeb direction)
      - self.alpha(x, v) : function computing the contact form α on (x,v)
    """

    def __init__(self, model):
        self.model = model

    def normalize_to_S5(self, x):
        x = jnp.asarray(x, dtype=jnp.float64)
        norm = jnp.linalg.norm(x)
        if norm < 1e-12:
            return jnp.array([1.0, 0, 0, 0, 0, 0], dtype=jnp.float64)
        return x / norm

    def verify_S5_constraint(self, x, tol=1e-6):
        norm2 = float(jnp.sum(x**2))
        return abs(norm2 - 1.0) < tol, norm2

    def verify_contact_form(self, x, v, tol=1e-8):
        # alpha = g(ξ, v) = <ξ, v> since g is standard dot
        alpha_theo = float(jnp.dot(self.model.Reeb, v))
        # from F: φ(x) = F @ x, and α(v)=g(φ(x), φ(v))?  Actually contact form α = g(φ(x), ·)
        alpha_matrix = float(jnp.dot(self.model.F @ x, v))
        return abs(alpha_theo - alpha_matrix) < tol, alpha_theo, alpha_matrix

    def verify_phi_squared(self, x, tol=1e-8):
        # check φ²(x) = -x + α(x) ξ
        phi_x = self.model.F @ x
        lhs = self.model.F @ phi_x
        alpha_x = float(jnp.dot(self.model.Reeb, x))
        rhs = -x + alpha_x * self.model.Reeb
        err = jnp.linalg.norm(lhs - rhs)
        return float(err) < tol, float(err), lhs, rhs

    def verify_metric_compatibility(self, x, v, w, tol=1e-8):
        # check g(φv, φw) = g(v,w) - α(v) α(w)
        phi_v = self.model.F @ v
        phi_w = self.model.F @ w
        lhs = float(jnp.dot(phi_v, phi_w))
        alpha_v = float(jnp.dot(self.model.Reeb, v))
        alpha_w = float(jnp.dot(self.model.Reeb, w))
        rhs = float(jnp.dot(v, w) - alpha_v * alpha_w)
        return abs(lhs - rhs) < tol, lhs, rhs

    def test_contact_form(self, num=20):
        key = jax.random.PRNGKey(42)
        failures = []
        for i in range(num):
            key, k1, k2 = jax.random.split(key, 3)
            x = self.normalize_to_S5(jax.random.normal(k1, (6,), dtype=jnp.float64))
            v = jax.random.normal(k2, (6,), dtype=jnp.float64)
            valid, theo, mat = self.verify_contact_form(x, v)
            if not valid:
                failures.append((i, theo, mat, abs(theo-mat)))
        return failures

    def run_complete_validation(self):
        print("Sasakian S^5 Implementation Validation")
        print("="*50)

        # 1. Matrix structure
        F = self.model.F
        skew_err = float(jnp.linalg.norm(F + F.T))
        print(f"φ‐matrix skew‐symmetry ||F+Fᵀ|| = {skew_err:.2e}")

        # 2. Test normalization
        pts = {
            'Generic': [0.4,0.4,0.4,0.4,0.4,0.1],
            'Random': np.random.rand(6),
            'Zeros': [0,0,0,0,0,0]
        }
        for name,p in pts.items():
            x = self.normalize_to_S5(p)
            ok, norm2 = self.verify_S5_constraint(x)
            print(f"{name}: ||x||² = {norm2:.10f} → {'OK' if ok else 'FAIL'}")

        # 3. Contact form tests
        failures = self.test_contact_form()
        if not failures:
            print("✓ All contact form tests passed")
        else:
            print(f"✗ {len(failures)} contact form failures; first:", failures[0])

        # 4. φ² identity
        x = self.normalize_to_S5([1,2,3,4,5,6])
        valid, err, lhs, rhs = self.verify_phi_squared(x)
        print(f"φ² identity error = {err:.2e} → {'OK' if valid else 'FAIL'}")

        # 5. Metric compatibility
        key = jax.random.PRNGKey(7)
        v = jax.random.normal(key, (6,), dtype=jnp.float64)
        w = jax.random.normal(key, (6,), dtype=jnp.float64)
        mv, lv, rv = self.verify_metric_compatibility(x, v, w)
        print(f"Metric compatibility error = |{lv:.6f} - {rv:.6f}| → {'OK' if mv else 'FAIL'}")

        return {
            'skew_symmetry_error': skew_err,
            'contact_failures': failures,
            'phi2_error': err,
            'metric_error': abs(lv-rv)
        }

if __name__ == '__main__':
    # Replace `YourS5Model` with your actual class that defines F and Reeb
    try:
        from SasakianS5Model import YourS5Model
    except ImportError:
        print("Could not import YourS5Model; ensure it’s defined.")
        exit(1)

    model = YourS5Model()
    validator = SasakianValidatorS5(model)
    summary = validator.run_complete_validation()
    print("Validation summary:", summary)
