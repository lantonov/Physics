import numpy as np
import matplotlib.pyplot as plt

# Fiber parameter
psi = np.linspace(-10, 10, 2000)

# Parameters for Villarceau circles (Clifford torus slice)
theta_vals = [np.pi/6, np.pi/3, np.pi/2]  # different slices
colors = ['tab:blue','tab:green','tab:orange']

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14,6))

# 3D torus embedding (for intuition)
# Parametrize torus in R^3: (R + r cos v) cos u, (R + r cos v) sin u, r sin v
# Villarceau circles are oblique slices, here approximated by Hopf coords
for theta, col in zip(theta_vals, colors):
    z1 = np.cos(theta/2)*np.exp(1j*psi/2)
    z2 = np.sin(theta/2)*np.exp(-1j*psi/2)
    x = 2*np.real(z1*np.conj(z2))
    y = 2*np.imag(z1*np.conj(z2))
    z = np.abs(z1)**2 - np.abs(z2)**2
    ax1.plot(x,y,label=f"θ={theta:.2f}",color=col)

ax1.set_title("Hopf fibers (Villarceau circles) projected to (x,y)")
ax1.set_aspect('equal')
ax1.legend()

# Now overlay with Poinsot spirals
k=0.3
r_sech = 1/np.cosh(k*psi)
r_csch = 1/np.sinh(k*psi)
ax2.plot(r_sech*np.cos(psi), r_sech*np.sin(psi), 'navy', label="sech spiral")
ax2.plot(r_csch*np.cos(psi), r_csch*np.sin(psi), 'darkorange', label="csch spiral")
ax2.set_title("Poinsot spirals (sech vs csch)")
ax2.set_aspect('equal')
ax2.legend()

plt.show()
