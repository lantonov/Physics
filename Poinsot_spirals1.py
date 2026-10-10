import numpy as np
import matplotlib.pyplot as plt

# Parameters with mathematical significance
a = 1.0      # characteristic length scale
k = 0.3      # spiral tightness parameter (smaller = tighter spiral)
R_bounded = 1.0     # scale for bounded spiral
R_asymptotic = 0.5  # scale for asymptotic spiral (smaller for better visualization)

# Extended theta range for better spiral visualization
theta = np.linspace(-15, 15, 3000)

# Bounded Poinsot spiral: r = R * sech(k*theta)
# This is always well-defined and bounded
r_sech = R_bounded / np.cosh(k * theta)
x_sech = r_sech * np.cos(theta)
y_sech = r_sech * np.sin(theta)

# Asymptotic Poinsot spiral: r = R * csch(k*theta)
# Handle the singularity at theta = 0 mathematically
sinh_values = np.sinh(k * theta)
# Define a threshold below which we consider the function undefined
epsilon = 1e-10
valid_indices = np.abs(sinh_values) > epsilon

# Create arrays with NaN for invalid points (mathematically rigorous approach)
r_csch = np.full_like(theta, np.nan)
r_csch[valid_indices] = R_asymptotic / sinh_values[valid_indices]

x_csch = r_csch * np.cos(theta)
y_csch = r_csch * np.sin(theta)

# Alternative: create separate arrays for positive and negative theta regions
# This gives a cleaner mathematical representation
theta_pos = theta[theta > epsilon]
theta_neg = theta[theta < -epsilon]

r_csch_pos = R_asymptotic / np.sinh(k * theta_pos)
x_csch_pos = r_csch_pos * np.cos(theta_pos)
y_csch_pos = r_csch_pos * np.sin(theta_pos)

r_csch_neg = R_asymptotic / np.sinh(k * theta_neg)
x_csch_neg = r_csch_neg * np.cos(theta_neg)
y_csch_neg = r_csch_neg * np.sin(theta_neg)

# Enhanced plotting
plt.figure(figsize=(12, 10))

# Plot bounded spiral
plt.subplot(2, 2, 1)
plt.plot(x_sech, y_sech, label=f'$r = {R_bounded} \cdot \mathrm{{sech}}({k}\\theta)$', 
         color='navy', linewidth=2)
plt.title("Bounded Poinsot Spiral")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True, alpha=0.3)
plt.legend()

# Plot asymptotic spiral (with singularity handling)
plt.subplot(2, 2, 2)
plt.plot(x_csch_pos, y_csch_pos, color='crimson', linewidth=2, label='$\\theta > 0$')
plt.plot(x_csch_neg, y_csch_neg, color='darkred', linewidth=2, linestyle='--', label='$\\theta < 0$')
plt.title(f"Asymptotic Poinsot Spiral\n$r = {R_asymptotic} \cdot \mathrm{{csch}}({k}\\theta)$")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True, alpha=0.3)
plt.legend()

# Combined plot
plt.subplot(2, 1, 2)
plt.plot(x_sech, y_sech, label='Bounded: $r = \mathrm{sech}(k\\theta)$', 
         color='navy', linewidth=2)
plt.plot(x_csch_pos, y_csch_pos, color='crimson', linewidth=2, 
         label='Asymptotic: $r = \mathrm{csch}(k\\theta)$, $\\theta > 0$')
plt.plot(x_csch_neg, y_csch_neg, color='darkred', linewidth=2, linestyle='--',
         label='Asymptotic: $r = \mathrm{csch}(k\\theta)$, $\\theta < 0$')
plt.title("Poinsot Spirals Comparison")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# Mathematical verification: print key properties
print(f"Bounded spiral maximum radius: {np.max(r_sech):.6f}")
print(f"Bounded spiral minimum radius: {np.min(r_sech):.6f}")
print(f"Asymptotic spiral radius range (positive θ): [{np.nanmin(r_csch_pos):.6f}, {np.nanmax(r_csch_pos):.6f}]")
print(f"Number of valid points in asymptotic spiral: {np.sum(~np.isnan(r_csch))}")
