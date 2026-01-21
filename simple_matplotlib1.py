import matplotlib.pyplot as plt

fruits = ['apple', 'banana', 'orange', 'grape']
counts = [40, 25, 55, 30]
colors = ['red', 'yellow', 'orange', 'purple']

plt.figure()
plt.bar(fruits, counts, color=colors)

plt.xlabel("Fruit")
plt.ylabel("Quantity")
plt.title("Fruit Quantity with Different Colors")

plt.show()
