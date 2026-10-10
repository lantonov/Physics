import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# Define modified quadratic function in t
def f_t(t, beta):
    return (beta**2 - 1) * t**2 + 2 * beta * t + 1

# Set parameters
beta_values = np.linspace(0, 0.99, 50)  # Avoid exactly beta = 1 (singular case)
roots_t = []

# Compute roots for different beta values
for beta in beta_values:
    try:
        root_1 = (-beta + np.sqrt(1 - beta**2)) / (beta**2 - 1)
        root_2 = (-beta - np.sqrt(1 - beta**2)) / (beta**2 - 1)
        roots_t.append((root_1, root_2))
    except:
        # Handle cases where beta -> 1 causes singularity
        roots_t.append((None, None))

# Plot roots vs beta
beta_plot = [b for b in beta_values if roots_t[beta_values.tolist().index(b)][0] is not None]
t_plot = [roots_t[beta_values.tolist().index(b)] for b in beta_plot]

plt.plot(beta_plot, [t[0] for t in t_plot], label='t_+', color='blue')
plt.plot(beta_plot, [t[1] for t in t_plot], label='t_-', color='red', linestyle='dashed')
plt.xlabel("Boost Parameter β")
plt.ylabel("t Solutions")
plt.title("Modified Quadratic Solutions in t")
plt.legend()
plt.show()
