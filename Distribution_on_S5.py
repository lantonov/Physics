import numpy as np
import matplotlib.pyplot as plt

# Step 1: Sample points on the unit 5-sphere S^5 in C^3
def sample_points_on_s5(n_points):
    z = np.random.randn(n_points, 6)  # 6 real numbers = 3 complex components
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    return z

# Step 2: Convert 6D real vector to 3D complex vector
def real_to_complex(z_real):
    return z_real[::2] + 1j * z_real[1::2]  # assumes [Re0, Im0, Re1, Im1, Re2, Im2]

# Step 3: Build an orthonormal basis in C^3 orthogonal to the contact direction
def get_local_basis_vectors(z):
    basis = []
    standard_basis = [np.array([1,0,0], dtype=complex), np.array([0,1,0], dtype=complex), np.array([0,0,1], dtype=complex)]
    for b in standard_basis:
        proj = np.vdot(z, b) * z / np.vdot(z, z)  # project b onto z
        ortho = b - proj
        if np.linalg.norm(ortho) > 1e-10:
            ortho = ortho / np.linalg.norm(ortho)
            basis.append(ortho)
        if len(basis) == 2:
            break
    return np.array(basis)

# Step 4: Project basis vectors to lie in the contact plane (Im⟨z, v⟩ = 0)
def compute_contact_plane_at_point(z_real):
    z = real_to_complex(z_real)
    basis = get_local_basis_vectors(z)

    contact_basis = []
    for v in basis:
        inner = np.vdot(z, v)
        imag_part = np.imag(inner)
        if np.abs(imag_part) > 1e-8:
            # subtract component in the Reeb direction (i z)
            iz = 1j * z
            proj_coeff = np.vdot(iz, v) / np.vdot(iz, iz)
            v_corrected = v - proj_coeff * iz
        else:
            v_corrected = v
        contact_basis.append(v_corrected)

    return np.array(contact_basis)

# Step 5: Project complex vectors to 2D for visualization
def complex_vector_to_2d(v):
    # We'll just use real parts of first and second components
    return np.real(v[0]), np.real(v[1])

# === MAIN EXECUTION ===

n_samples = 100
s5_points = sample_points_on_s5(n_samples)

points_2d = []
contact_dirs_2d = []

for p in s5_points:
    z = real_to_complex(p)
    contact_plane = compute_contact_plane_at_point(p)
    
    if contact_plane.shape[0] >= 1:
        x, y = complex_vector_to_2d(z)
        dx, dy = complex_vector_to_2d(contact_plane[0])
        
        points_2d.append((x, y))
        contact_dirs_2d.append((dx, dy))

points_2d = np.array(points_2d)
contact_dirs_2d = np.array(contact_dirs_2d)

# === PLOT ===
plt.figure(figsize=(8,8))
plt.quiver(points_2d[:,0], points_2d[:,1], contact_dirs_2d[:,0], contact_dirs_2d[:,1],
           angles='xy', scale_units='xy', scale=1.5, width=0.003)
plt.title("Projected Contact Planes on S⁵ (Real 2D slice)")
plt.xlabel("Re(z₀)")
plt.ylabel("Re(z₁)")
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.show()
