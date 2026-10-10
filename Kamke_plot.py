import numpy as np
import matplotlib.pyplot as plt

# Define the function for x >= 1
def y(x):
    return (x**2 - 1)**(1/4)*(x + np.sqrt(x**2 - 1))

# Generate x-values from 1 to 5:
x_vals = np.linspace(1.001, 5, 400)  # start a little above 1 to avoid 0^(1/4)
y_vals = y(x_vals)

plt.figure(figsize=(8,5))
plt.plot(x_vals, y_vals, label=r"$y(x)=(x^2-1)^{1/4}(x+\sqrt{x^2-1})$")
plt.title("Graph of Solution y(x) for 4(x²-1)²y''+(x²+2)y=0\n(using C₁=1, C₂=0)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()
