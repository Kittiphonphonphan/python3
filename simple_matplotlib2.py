import numpy as np
import matplotlib.pyplot as plt

# data
x = np.linspace(-10, 10, 400)
y = 1 / (1 + np.exp(-x))   # sigmoid function

# plot sigmoid
plt.plot(x, y, label="sigmoid")

# infinite lines
plt.axvline(0, linestyle="--", color="black")   # x = 0
plt.axhline(0.5, linestyle=":", color="gray")   # y = 0.5

# labels
plt.xlabel("x")
plt.ylabel("y")
plt.title("Sigmoid Function with Infinite Lines")
plt.legend()

plt.show()
