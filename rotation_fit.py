import numpy as np
import matplotlib.pyplot as plt

# Define the orbital velocity function
def orbital_velocity(r, omega):
    return np.sqrt((1.48*1e5*np.exp(-r/550)*omega**2 * r) / np.sqrt(1 + omega**2 * r**2))

def orbital_newton(r, omega):
    return np.sqrt((5e4*np.exp(-r/550) * r) /(1+r**3))

# Define parameters
omega = 0.2  # Angular velocity (example value)
radius_model = np.linspace(0.1, 20, 500)  # Radial distance from the center in kpc
velocity_model = orbital_velocity(radius_model, omega)
newton_model = orbital_newton(radius_model, omega)

# Experimental data for NGC 3198 rotation curve
radius_data = np.array([0.1, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0])  # in kpc
velocity_data = np.array([10, 50, 90, 120, 150, 155, 160, 165, 165])      # in km/s

# Plot both the model and the experimental data
plt.figure(figsize=(10, 6))
plt.plot(radius_model, velocity_model, label=f"Model: $\\omega$ = {omega}", color="blue")
plt.plot(radius_model, newton_model, label=f"Model: Newton", color="black", linestyle="--")
plt.scatter(radius_data, velocity_data, color="red", label="Experimental Data (NGC 3198)", zorder=5)
plt.xlabel("Radius (kpc)")
plt.ylabel("Orbital Velocity (km/s)")
plt.title("Comparison of Model and Experimental Rotation Curves")
plt.legend()
plt.grid()
plt.show()
