import operator
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sin, cos, simplify
from genetic_programming import gp_minimize

def protected_division(x, y):
    return x / y if y != 0 else 1


functions = {
    operator.add: 2,
    operator.sub: 2,
    operator.mul: 2,
    # protected_division: 2,
    np.sin: 1,
    np.cos: 1
}

# Define your terminals - these could be constants and variables like 'x'.
terminals = ['x_0', 10]

max_depth = None  # The maximum depth of the tree
n_pop = 500  # The number of individuals in the population
n_generations = 100  # The number of generations
crossover_probability = 0.9  # The probability of crossover
mutation_probability = 0.1  # The probability of mutation
depth_coefficient = 0.1  # The coefficient for the depth penalty
tournament_size = 1  # The tournament size for tournament selection

# Generate data
x_0 = np.linspace(-10, 10, 300).reshape(-1, 1)
y = x_0**2 + 10*np.cos(x_0*10) + 10

X = x_0
y = y.reshape(-1, 1)
plt.plot(X, y, label='True')


res = gp_minimize(X, y, functions, terminals, max_depth, population_size=n_pop, n_generations=n_generations, crossover_probability=crossover_probability,
                  mutation_probability=mutation_probability, depth_coefficient=depth_coefficient, tournament_size=tournament_size)

y_pred = [res.evaluate({f'x_{i}': x_i for i, x_i in enumerate(row)})
          for row in X]

plt.plot(X, y, label='True')
plt.plot(X, y_pred, label='Predicted')
plt.legend()
plt.show()

print(res.expression())
print(res.depth())

sympy_functions = {
    operator.add: lambda x, y: x + y,
    operator.sub: lambda x, y: x - y,
    operator.mul: lambda x, y: x * y,
    protected_division: lambda x, y: x / y,
    np.sin: lambda x: sin(x),
    np.cos: lambda x: cos(x)
}

x_0 = symbols('x')

x = res.sympy_expression({f'x_0': x_0}, sympy_functions)

x

simplify(x)
