import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(-10, 10, 2000)

# Parameters
k = 1/3   # scaling factor
R = 1.0   # overall scale

# Bounded Poinsot spiral: r = R * sech(k*theta)
r_sech = R / np.cosh(k*theta)
x_sech = r_sech * np.cos(theta)
y_sech = r_sech * np.sin(theta)

# Asymptotic Poinsot spiral: r = R * csch(k*theta)
# Avoid division by zero near theta=0
r_csch = R / np.sinh(k*theta)
x_csch = r_csch * np.cos(theta)
y_csch = r_csch * np.sin(theta)

# Plot
plt.figure(figsize=(8,8))
plt.plot(x_sech, y_sech, label=r'$r=\mathrm{sech}(\theta/3)$', color='navy')
plt.plot(x_csch, y_csch, label=r'$r=\mathrm{csch}(\theta/3)$', color='crimson')
plt.title("Poinsot Spirals (bounded and asymptotic)")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()
