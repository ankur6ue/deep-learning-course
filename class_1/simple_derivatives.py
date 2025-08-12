import numpy as np # linear algebra
import matplotlib.pyplot as plt

# This program calculates the derivative of a function manually and compares it against the exact value
# for different values of \delta
x = 1
n = 2
# f(x) = x^n
# f'(x) = nx^(n-1)
df = n*(1 ** (n-1))
df_manual_list = []
for delta_x in np.arange(0.0001, 1, 0.05):
    df_manual = ((1+delta_x) ** n - 1 ** n)/delta_x
    df_manual_list.append(df_manual)
    print(f"delta_x: {delta_x: .2f}, derivative: {df_manual: .2f}")

# plot x^n for two different values of n on the same plot

# Generate x-values
# np.linspace creates an array of evenly spaced numbers over a specified interval.
x = np.linspace(-2, 2, 100) # 100 points between -5 and 5
# Calculate y-values based on x^n
n = 2
y = x**n
# Create a figure and a primary Axes object
fig, ax1 = plt.subplots()
ax1.plot(x, y, label=f'$x^{n}$', color='r') # Plot x vs y, add a label for the legend
# Add labels and title
ax1.set_xlabel('x')
ax1.set_ylabel(f'$x^{n}$')

ax2 = ax1.twinx()
n = 4
y = x**n
ax2.plot(x, y, label=f'$x^{n}$', color='b') # Plot x vs y, add a label for the legend
# Add labels and title
ax2.set_xlabel('x')
ax2.set_ylabel(f'$x^{n}$')
# Add a grid for better readability
plt.grid(True)
# Add a legend
plt.legend()
# Ensure tight layout
# fig.tight_layout()
# Display the plot
plt.show()
print('done')



