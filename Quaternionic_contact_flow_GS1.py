import jax.numpy as jnp

class ThreeSasakianS7:
    def __init__(self):
        # Imaginary quaternions i,j,k acting on H^2 ≅ R^8
        # In block form F₁ = [[I₂⊗A_i, 0],[0, I₂⊗A_i]], etc.
        # Here A_i, A_j, A_k are the 4×4 real representations on a single H
        A_i = jnp.array([[ 0,-1, 0, 0],
                         [ 1, 0, 0, 0],
                         [ 0, 0, 0,-1],
                         [ 0, 0, 1, 0]], dtype=float)
        A_j = jnp.array([[ 0, 0,-1, 0],
                         [ 0, 0, 0, 1],
                         [ 1, 0, 0, 0],
                         [ 0,-1, 0, 0]], dtype=float)
        A_k = jnp.array([[ 0, 0, 0,-1],
                         [ 0, 0,-1, 0],
                         [ 0, 1, 0, 0],
                         [ 1, 0, 0, 0]], dtype=float)

        # Build 8×8 F’s by acting diagonally on two copies of H
        self.F1 = jnp.block([[A_i, jnp.zeros((4,4))],
                             [jnp.zeros((4,4)), A_i]])
        self.F2 = jnp.block([[A_j, jnp.zeros((4,4))],
                             [jnp.zeros((4,4)), A_j]])
        self.F3 = jnp.block([[A_k, jnp.zeros((4,4))],
                             [jnp.zeros((4,4)), A_k]])

    def alpha1_contact_form(self, x):
        # α₁(v) = ⟨F₁ x, v⟩  ⇒ vector representative = F₁ x
        return self.F1 @ x

    def alpha2_contact_form(self, x):
        return self.F2 @ x

    def alpha3_contact_form(self, x):
        return self.F3 @ x

    def extract_horizontal_curvatures(self, F1, F2, F3, x):
        """
        Extract the sum of the two horizontal sectional curvatures
        in each quaternionic channel Fᵢ at point x∈S⁷.

        Returns
        -------
        k1, k2, k3 : float
            Sum of the two sectional curvatures in each channel.
        k1_planes, k2_planes, k3_planes : (float, float)
            The individual sectional curvatures in each channel.
        """
        # 1) Get contact‐form vectors αᵢ = Fᵢ x
        a1 = F1 @ x
        a2 = F2 @ x
        a3 = F3 @ x

        # 2) Orthonormalize to get Reeb vectors ξ₁, ξ₂, ξ₃
        def gram_schmidt(v, basis):
            w = v - sum((b @ v) * b for b in basis)
            norm = jnp.linalg.norm(w)
            # avoid divide-by-zero
#            return jnp.where(norm>1e-12, w/norm, jnp.zeros_like(w))
            return jnp.where(norm > jnp.finfo(w.dtype).eps, w/norm, jnp.zeros_like(w))


        xi1 = a1 / jnp.linalg.norm(a1)
        xi2 = gram_schmidt(a2, [xi1])
        xi3 = gram_schmidt(a3, [xi1, xi2])

        # 3) Build vertical span [x, ξ₁, ξ₂, ξ₃] ∈ ℝ⁸×4
        P = jnp.stack([x, xi1, xi2, xi3], axis=1)

        # 4) Complete QR → Qfull ∈ ℝ⁸×8
        Qfull, _ = jnp.linalg.qr(P, mode='complete')

        # 5) Horizontal frame = last 4 cols of Qfull
        H = Qfull[:, 4:]   # shape (8,4)

        # 6) For each quaternionic F, restrict to H and read off s₀,s₁
        def channel(F):
            FH = H.T @ F @ H             # 4×4 skew‐symmetric
            M  = -FH @ FH                # symmetric, eigenvalues s₀²,s₀²,s₁²,s₁²
            e  = jnp.linalg.eigvalsh(M)  # ascending
            s0 = jnp.sqrt(jnp.maximum(e[3], 0.))
            s1 = jnp.sqrt(jnp.maximum(e[2], 0.))
            return float(s0), float(s1)

        k1p, k2p, k3p = channel(F1), channel(F2), channel(F3)
        k1 = k1p[0] + k1p[1]
        k2 = k2p[0] + k2p[1]
        k3 = k3p[0] + k3p[1]

        return k1, k2, k3, k1p, k2p, k3p


# --------------------
# Quick sanity‐check
# --------------------
def main():
    model = ThreeSasakianS7()

    # Define your test points
    points = {
        'Generic':  jnp.array([0.5,0.5,0.5,0.5,0,0,0,0], float),
        'Random':   jnp.array([0.34,0.45,0.69,0.79,87,98,34,2], float),
        'Symmetric': jnp.ones(8)/jnp.linalg.norm(jnp.ones(8))
    }

    for name, x in points.items():
        k1, k2, k3, k1p, k2p, k3p = \
            model.extract_horizontal_curvatures(model.F1,
                                                model.F2,
                                                model.F3,
                                                x)

        trace = k1 + k2 + k3
        print(f"{name}:")
        print(f"  k1,k2,k3 sums = {k1:.6f}, {k2:.6f}, {k3:.6f}")
        print(f"  plane curvatures:")
        print(f"    F1 → {k1p}")
        print(f"    F2 → {k2p}")
        print(f"    F3 → {k3p}")
        print(f"  Ricci trace = {trace:.6f}\n")

if __name__ == '__main__':
    main()

