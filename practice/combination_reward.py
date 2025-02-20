import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Define an offset value to lower the output for negative x and y.
offset = 0.2  # Adjust this value as needed.

# Create a grid of x and y values.
x = np.linspace(-1, 1, 200)   # x-axis for the dot product values.
y = np.linspace(-0.5, 0.5, 200)   # y-axis for the additional parameter.
X, Y = np.meshgrid(x, y)

alpha = 5
beta = 12.0
# Compute the hyperbolic tangent components for x and y.
# tanh_component_x = np.tanh(alpha*X)
# tanh_component_y = np.tanh(beta*Y)

sigmoid_x = sigmoid(alpha*X)
sigmoid_y = sigmoid(beta*Y)
# Compute the raw combined function as the product of the two tanh components.
Z_raw = sigmoid_x * sigmoid_y

# Modify Z so that if both x and y are negative, Z is forced negative and lowered by the offset.
# Z = np.where((X < 0) & (Y < 0), -0.5, Z_raw)
Z = Z_raw * 2

# Set up the figure and a 3D subplot.
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# Set labels and title.
ax.set_title(
    '3D Plot: Modified Function\n(For negative x and y, z is forced negative and lowered)')
ax.set_xlabel('Dot Product (x)/heading')
ax.set_ylabel('y Delta distance')
ax.set_zlabel('f(x,y)')

# Add a color bar.
fig.colorbar(surf, shrink=0.5, aspect=10)

plt.show()
