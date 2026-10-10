import numpy as np
import matplotlib.pyplot as plt

psi = np.linspace(-10, 10, 2000)

# Sech spiral (closed Hopf fibers, S^3)
r_sech = 1/np.cosh(0.4*psi)
x_sech = r_sech * np.cos(psi)
y_sech = r_sech * np.sin(psi)

# Csch spiral (open Hopf fibers, H^3)
r_csch = 1/np.sinh(0.4*psi)
x_csch = r_csch * np.cos(psi)
y_csch = r_csch * np.sin(psi)

plt.figure(figsize=(7,7))
plt.plot(x_sech, y_sech, label="sech spiral (closed fiber)", color="navy")
plt.plot(x_csch, y_csch, label="csch spiral (open fiber)", color="darkorange")
plt.legend()
plt.axis("equal")
plt.title("Hopf fiber projections: sech vs csch spirals")
plt.show()
