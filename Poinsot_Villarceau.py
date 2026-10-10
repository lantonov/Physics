import numpy as np
import matplotlib.pyplot as plt

# Use a clean aesthetic style
#plt.style.use('seaborn-v0_8')

# Define psi range and theta values with better mathematical spacing
psi = np.linspace(-10 * np.pi, 10 * np.pi, 2000)

# Choose theta values that give distinct radii for better visualization
theta_values = [np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]
colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple']

# Calculate exact radii for mathematical verification
print("Hopf fiber radii:")
for theta in theta_values:
    radius = np.abs(np.sin(theta))
    print(f"θ = {theta:.4f} ({theta/np.pi:.3f}π): radius = {radius:.4f}")

# Prepare figure with subplots for better analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Left plot: All Hopf fibers
for i, (theta, color) in enumerate(zip(theta_values, colors)):
    # Hopf coordinates (your original calculation)
    z1 = np.cos(theta / 2) * np.exp(1j * (psi / 2))
    z2 = np.sin(theta / 2) * np.exp(1j * (-psi / 2))
    
    # Hopf map
    x = 2 * np.real(z1 * np.conj(z2))
    y = 2 * np.imag(z1 * np.conj(z2))
    
    # Verification: this should equal sin(theta) * cos(psi), sin(theta) * sin(psi)
    x_theoretical = np.sin(theta) * np.cos(psi)
    y_theoretical = np.sin(theta) * np.sin(psi)
    
    # Check mathematical consistency
    max_error_x = np.max(np.abs(x - x_theoretical))
    max_error_y = np.max(np.abs(y - y_theoretical))
    print(f"θ = {theta:.3f}: max error x = {max_error_x:.2e}, y = {max_error_y:.2e}")
    
    ax1.plot(x, y, label=f"θ = {theta:.3f} (r = {np.sin(theta):.3f})", 
             color=color, linewidth=2, alpha=0.8)

# Overlay Poinsot spiral on left plot
k = 0.2
r_sech = 1 / np.cosh(k * psi)
ax1.plot(r_sech * np.cos(psi), r_sech * np.sin(psi), 'k--', 
         label="Poinsot spiral: sech(kψ)", linewidth=2)

ax1.set_title("All Hopf Fibers with Poinsot Spiral")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_aspect('equal')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right plot: Your original three theta values with transparency
theta_original = [np.pi / 4, np.pi / 2, 3 * np.pi / 4]
colors_original = ['tab:blue', 'tab:green', 'tab:orange']

for theta, color in zip(theta_original, colors_original):
    z1 = np.cos(theta / 2) * np.exp(1j * (psi / 2))
    z2 = np.sin(theta / 2) * np.exp(1j * (-psi / 2))
    x = 2 * np.real(z1 * np.conj(z2))
    y = 2 * np.imag(z1 * np.conj(z2))
    
    # Use transparency and different line styles to show overlap
    linestyle = '-' if np.sin(theta) == np.sin(np.pi/4) and theta == np.pi/4 else '--' if theta == 3*np.pi/4 else '-'
    ax2.plot(x, y, label=f"θ = {theta:.3f} (r = {np.sin(theta):.3f})", 
             color=color, linewidth=3, alpha=0.7, linestyle=linestyle)

# Add the Poinsot spiral
ax2.plot(r_sech * np.cos(psi), r_sech * np.sin(psi), 'k--', 
         label="Poinsot spiral: sech(kψ)", linewidth=2)

ax2.set_title("Original Problem: Overlapping Fibers\n(Blue and Orange have same radius)")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_aspect('equal')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Mathematical analysis
print(f"\nWhy you see only green and orange:")
print(f"θ = π/4 gives radius = sin(π/4) = {np.sin(np.pi/4):.6f}")
print(f"θ = 3π/4 gives radius = sin(3π/4) = {np.sin(3*np.pi/4):.6f}")
print(f"These are identical! The orange curve completely overlays the blue one.")
print(f"θ = π/2 gives radius = sin(π/2) = {np.sin(np.pi/2):.6f} (the green curve)")