# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 13:28:29 2023

@author: lanto
"""

# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import optimize as opt
from scipy.special import lambertw

# Define functions
def f(x):
    return -np.log(1-np.log(x))

def g(x):
    return x

# Create x and y ranges

x = np.linspace(0.0001, 3, 1000)
y1 = f(x)
y2 = g(x)

# Plot the functions
fig, ax = plt.subplots()
ax.plot(x, y1, label='LHS')
ax.plot(x, y2, label='RHS')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
plt.show()

# Find the solutions
solutions = opt.fsolve(lambda x: f(x) - g(x), [-1.5, 2, -1.2, 2.5])
print('The solutions are:', solutions)