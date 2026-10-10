import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create a figure and axis with an equal aspect ratio.
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')

# --- Draw the sphere ---
# Use a circle of radius 2 centered at (0,0)
sphere = patches.Circle((0, 0), 2, edgecolor='black', facecolor='none', linewidth=2)
ax.add_patch(sphere)

# --- Mark the poles ---
N_pole = (0, 2)   # North pole
S_pole = (0, -2)  # South pole
ax.plot(*N_pole, 'ko')  # black dot
ax.plot(*S_pole, 'ko')
ax.text(N_pole[0], N_pole[1] + 0.2, 'N', ha='center', va='bottom', fontsize=12)
ax.text(S_pole[0], S_pole[1] - 0.2, 'S', ha='center', va='top', fontsize=12)

# --- Draw vertical axis ---
ax.plot([0, 0], [N_pole[1], S_pole[1]], color='gray', linestyle='--', linewidth=1)

# --- Draw tangent planes as horizontal dashed lines ---
# Tangent plane at S (for the North Chart) at y = -2.5
ax.plot([-3, 3], [-2.5, -2.5], color='black', linestyle='--', linewidth=1)
ax.text(0, -2.7, 'Tangent Plane at S\n(North Chart)', ha='center', va='top', fontsize=10)
# Tangent plane at N (for the South Chart) at y = 2.5
ax.plot([-3, 3], [2.5, 2.5], color='black', linestyle='--', linewidth=1)
ax.text(0, 2.7, 'Tangent Plane at N\n(South Chart)', ha='center', va='bottom', fontsize=10)

# --- Sample point on the sphere ---
A = (1.5, 0.5)
ax.plot(A[0], A[1], 'ko')  # Mark point A
ax.text(A[0] + 0.1, A[1], 'A', fontsize=12, color='black')

# --- Compute and draw the North Chart Projection ---
# Stereographic projection from the north pole: A_N = N + 3*(A - N)
t = 3
A_N = (N_pole[0] + t * (A[0] - N_pole[0]), N_pole[1] + t * (A[1] - N_pole[1]))
# This yields (4.5, -2.5) - note the y-coordinate lies on the tangent plane.
ax.plot([N_pole[0], A_N[0]], [N_pole[1], A_N[1]], color='blue', linestyle='--', linewidth=1)
ax.plot(A_N[0], A_N[1], 'bo')
ax.text(A_N[0] + 0.1, A_N[1], 'North Chart\nProjection', color='blue', fontsize=10)

# --- Compute and draw the South Chart Projection ---
# Stereographic projection from the south pole: A_S = S + 1.8*(A - S)
s = 1.8
A_S = (S_pole[0] + s * (A[0] - S_pole[0]), S_pole[1] + s * (A[1] - S_pole[1]))
# This yields (2.7, 2.5) - the point lies on the tangent plane at N.
ax.plot([S_pole[0], A_S[0]], [S_pole[1], A_S[1]], color='red', linestyle='--', linewidth=1)
ax.plot(A_S[0], A_S[1], 'ro')
ax.text(A_S[0] + 0.1, A_S[1], 'South Chart\nProjection', color='red', fontsize=10)

# --- Blended projection ---
# Compute the average of the two projections
A_blend = ((A_N[0] + A_S[0]) / 2, (A_N[1] + A_S[1]) / 2)
ax.plot(A_blend[0], A_blend[1], 'o', color='magenta')
ax.text(A_blend[0] + 0.1, A_blend[1] + 0.1, 'Blended Projection', color='magenta', fontsize=10)
ax.text(A_blend[0] + 0.1, A_blend[1] + 0.3, 'Smooth Transition', color='magenta', fontsize=10)

# Draw arrows from each projection to the blended point
ax.annotate('', xy=A_blend, xytext=A_N, arrowprops=dict(facecolor='magenta', arrowstyle='->'))
ax.annotate('', xy=A_blend, xytext=A_S, arrowprops=dict(facecolor='magenta', arrowstyle='->'))

# --- Final touches ---
ax.set_xlim(-4, 6)
ax.set_ylim(-4, 4)
ax.axis('off')
plt.title("Stereographic Projections with Smooth Transition", fontsize=14)

plt.show()
