import matplotlib.pyplot as plt
import numpy as np

# Define the range for t
t = np.linspace(-10, 10, 400)

# Define the function r = t
r**2 = t**2

# Create the plot
plt.figure(figsize=(6, 6))
plt.plot(t, r, label="r = t", color="blue")
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
plt.title("Graph of r = t in 2D")
plt.xlabel("t")
plt.ylabel("r")
plt.grid(True)
plt.legend()
plt.show()
