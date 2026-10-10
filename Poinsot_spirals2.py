import numpy as np
import matplotlib.pyplot as plt

# --- Define the cube root function that always returns the real root ---
def real_cuberoot(x):
    return np.sign(x) * np.abs(x)**(1/3)

# --- Define D(t) from eq. 50h (with c=1, c1=1 for simplicity) ---
def D(t):
    return real_cuberoot(
        45*t**4 + 5.1962*np.sqrt(4*t**6 + 71*t**8 + 8*t**10) + 2*t**6
    )

# --- Define the three roots (coefficients from your Overleaf eq. 50h) ---
def r_real(t):
    return 0.26457*D(t) - 0.41997*(3*t**2 - t**4)/D(t) + 0.33333*t**2

def r_complex1(t):
    return (-(0.13228 - 0.22912j)*D(t)
            + (0.20999 + 0.36371j)*(3*t**2 - t**4)/D(t)
            + 0.33333*t**2)

def r_complex2(t):
    return (-(0.13228 + 0.22912j)*D(t)
            + (0.20999 - 0.36371j)*(3*t**2 - t**4)/D(t)
            + 0.33333*t**2)

# --- Sample domain ---
t = np.linspace(-5, 5, 2000)

# --- Plot 1: Real root in Cartesian (r vs t) ---
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(t, r_real(t), color='navy')
plt.title("Real root: r vs t (cardioid-like)")
plt.xlabel("t")
plt.ylabel("r")
plt.grid(True)

# --- Plot 2: Complex roots in the complex plane ---
plt.subplot(1,2,2)
plt.plot(np.real(r_complex1(t)), np.imag(r_complex1(t)), label="Root 1", color='crimson')
plt.plot(np.real(r_complex2(t)), np.imag(r_complex2(t)), label="Root 2", color='green')
plt.title("Complex roots trajectory in C-plane")
plt.xlabel("Re(r)")
plt.ylabel("Im(r)")
plt.legend()
plt.grid(True)
plt.axis("equal")

plt.tight_layout()
plt.show()

# --- Optional: Polar projection of the real root (Poinsot spiral) ---
theta = np.linspace(-15, 15, 4000)
rvals = r_real(theta)
x = rvals * np.cos(theta)
y = rvals * np.sin(theta)

plt.figure(figsize=(5,5))
plt.plot(x, y, color='darkorange')
plt.title("Polar projection of real root (Poinsot spiral)")
plt.axis("equal")
plt.grid(True)
plt.show()
