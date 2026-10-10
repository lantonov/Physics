# Validate_S5_curvatures_fix.py

from jax import config
config.update("jax_enable_x64", True)    # Must come before any jax.numpy import!

import jax.numpy as jnp
import numpy as np

# ------------------------------
# 1) Build the 6×6 complex‐structure
# ------------------------------
def make_F6():
    # three copies of [0 -1;1 0]
    J2 = jnp.array([[0., -1.],
                    [1.,  0.]], dtype=jnp.float64)
    return jnp.block([
        [J2,                 jnp.zeros((2,2), dtype=jnp.float64), jnp.zeros((2,2), dtype=jnp.float64)],
        [jnp.zeros((2,2), dtype=jnp.float64),   J2,                 jnp.zeros((2,2), dtype=jnp.float64)],
        [jnp.zeros((2,2), dtype=jnp.float64),   jnp.zeros((2,2), dtype=jnp.float64),   J2]
    ])

F = make_F6()

# ------------------------------
# 2) Extraction function
# ------------------------------
def extract_horizontal_curvatures(F, x):
    """
    Given F (6×6 skew) and x∈S⁵ (float64), return
      s0, s1  = the two horizontal sectional curvatures,
      s_sum   = s0 + s1  (the transverse Ricci in that channel)
    """
    # Reeb
    xi = F @ x

    # Gram–Schmidt on [x, xi]
    def gs(v, basis):
        w = v - sum(jnp.dot(b, v) * b for b in basis)
        n = jnp.linalg.norm(w)
        return jnp.where(n>1e-12, w/n, jnp.zeros_like(w))

    e0 = x / jnp.linalg.norm(x)
    e1 = gs(xi, [e0])

    # QR to get horizontal 4-frame
    P = jnp.stack([e0, e1], axis=1)   # 6×2
    Q, _ = jnp.linalg.qr(P, mode='complete')  # 6×6
    H = Q[:, 2:]                        # 6×4

    # Restrict F, get eigenvalues of -F_H^2
    FH = H.T @ F @ H                    # 4×4 skew
    M  = -FH @ FH
    e  = jnp.linalg.eigvalsh(M)         # ascending [λ0…λ3]
    s0 = jnp.sqrt(jnp.maximum(e[3], 0.0))
    s1 = jnp.sqrt(jnp.maximum(e[2], 0.0))
    return float(s0), float(s1), float(s0 + s1)

# ------------------------------
# 3) Main
# ------------------------------
def main():
    # prepare test points in float64
    pts = {
        'Generic':   jnp.array([0.5,0.5,0.2,0.1,0.4,0.3], dtype=jnp.float64),
        'Random':    jnp.array(np.random.randn(6),       dtype=jnp.float64),
        'Symmetric': jnp.ones(6, dtype=jnp.float64)
    }
    # normalize
    for k in pts:
        x = pts[k]
        pts[k] = x / jnp.linalg.norm(x)

    for name, x in pts.items():
        s0, s1, trace = extract_horizontal_curvatures(F, x)
        print(f"{name}:")
        print(f"  horizontal sectional curvatures = ({s0:.6f}, {s1:.6f})")
        print(f"  transverse Ricci (s0+s1)       = {trace:.6f}\n")

if __name__=="__main__":
    main()
