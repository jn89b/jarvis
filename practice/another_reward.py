"""
Gaussian Function e−x2e−x2:
This function is widely used because it naturally peaks at x=0x=0 and symmetrically decays as ∣x∣∣x∣ increases. 
It’s a common choice when you want a “bump” or peak at the origin in one dimension.

Sigmoid (Logistic) Function 11+exp⁡(−ky)1+exp(−ky)1​:
The sigmoid function is famous for its smooth transition from 0 to 1. By using a steepness factor (here, k=10k=10), the function quickly rises for positive yy and stays near 0 for negative yy. 
This is why it’s often used in contexts like machine learning (e.g., in neural networks) to model threshold behavior.
"""


import numpy as np
import matplotlib.pyplot as plt


# Create a grid of x and y values
x = np.linspace(-3.14, 3.14, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)

# Define the function:
# alpha adjusts the steepness of the sigmoid function which is the
# representation of the y-axis, smaller alpha means a smoother curve
# and larger alpha means a steeper curve.
beta = -1
# beta is the same thing as alpha but for the x-axis for the gaussian
# function, smaller beta means a smoother curve and larger beta means
alpha = 0.5
# f(x, y) = exp(-x^2) * (1 / (1 + exp(-10*y)))
Z = np.exp(-alpha*X**2) * (1.0 / (1.0 + np.exp(beta * Y)))

# Set up the figure and a 3D subplot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_title('3D Function: $e^{-x^2} \\cdot \\frac{1}{1+e^{-10y}}$')
ax.set_xlabel('heading error')
ax.set_ylabel('distance')
ax.set_zlabel('f(x,y)')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
