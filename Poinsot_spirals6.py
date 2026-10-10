import numpy as np
import matplotlib.pyplot as plt

psi = np.linspace(-10,10,2000)

# Closed Hopf fiber projection (sech spiral)
r_sech = 1/np.cosh(0.3*psi)
x_sech = r_sech*np.cos(psi)
y_sech = r_sech*np.sin(psi)

# Open Hopf fiber projection (csch spiral)
r_csch = 1/np.sinh(0.3*psi)
x_csch = r_csch*np.cos(psi)
y_csch = r_csch*np.sin(psi)

plt.figure(figsize=(6,6))
plt.plot(x_sech,y_sech,label="sech spiral (closed fiber)",color='navy')
plt.plot(x_csch,y_csch,label="csch spiral (open fiber)",color='darkorange')
plt.legend()
plt.axis("equal")
plt.title("Hopf fibers projected to plane: closed vs open")
plt.show()
