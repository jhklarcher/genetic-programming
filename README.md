# Genetic Programming for Symbolic Regression

This project is an experimental implementation of a genetic programming algorithm for symbolic regression. It's designed for personal use and educational purposes. The algorithm evolves a population of program trees to model data, optimizing for both accuracy and simplicity. The idea is to be able to find possible equations that describe the data.

For a more robust, complete and performant implementation of symbolic regression, check out:
- [PySR](https://github.com/MilesCranmer/PySR)
- [SymbolicRegression.jl](https://github.com/MilesCranmer/SymbolicRegression.jl)

For a Scikit-learn compatible implementation of genetic programming for symbolic regression, check out:
- [gplearn](https://github.com/trevorstephens/gplearn)

## Features
- Implements genetic programming concepts for symbolic regression.
- Allows for customization of functions, terminals, and evolutionary parameters.
- Sympy integration for expression simplification and analysis.

## Installation
Clone the repository, install the required dependencies and install the project using pip:
```bash
git clone https://github.com/jhklarcher/genetic-programming.git
cd genetic-programming
pip install numpy matplotlib sympy
pip install .
```

## Usage
This tool is used to evolve program trees for symbolic regression. Define functions, terminals, and parameters, then run the algorithm on your dataset.

### Step-by-step example
Here's an example demonstrating how to use this tool to evolve a program tree for symbolic regression.

- Import required libraries.
    ```python
    import operator
    import numpy as np
    import matplotlib.pyplot as plt
    from sympy import symbols, sin, cos, simplify
    from genetic_programming import simbolic_regression
    ```
- Generate data.
    ```python
    x_0 = np.linspace(-10, 10, 300).reshape(-1, 1)
    y = x_0**2 + 10*np.cos(x_0*10) + 10
    y = y.reshape(-1, 1)
    ```
- Define functions and terminals.
    ```python
    functions = {
        operator.add: 2,
        operator.sub: 2,
        operator.mul: 2,
        np.sin: 1,
        np.cos: 1
    }
    terminals = ['x_0', 10]
    ```
- Set parameters.
    ```python
    max_depth = None
    n_pop = 500
    n_generations = 100
    crossover_probability = 0.9
    mutation_probability = 0.1
    depth_coefficient = 0.1
    tournament_size = 1
    ```
- Run symbolic regression.
    ```python
    res = simbolic_regression(
        x_0, y, functions, terminals, max_depth,
        population_size=n_pop,
        n_generations=n_generations,
        crossover_probability=crossover_probability,
        mutation_probability=mutation_probability,
        depth_coefficient=depth_coefficient,
        tournament_size=tournament_size
    )
    ```
- Plot and analyze results.
    ```python
    y_pred = [res.evaluate({f'x_{i}': x_i for i, x_i in enumerate(row)}) for row in x_0]

    plt.plot(x_0, y, label='True')
    plt.plot(x_0, y_pred, label='Predicted')
    plt.legend()
    plt.show()

    print(res.expression())
    print(res.depth())
    ```
    The output should look like this:
    ```
    add(add(mul(x_0, x_0), 10), mul(10, cos(mul(10, x_0))))
    4
    ```
    The expression is a program tree that can be evaluated to predict values. 4 is the depth of the tree.
- Analyze results with Sympy.
    ```python
    sympy_functions = {
        operator.add: lambda x, y: x + y,
        operator.sub: lambda x, y: x - y,
        operator.mul: lambda x, y: x * y,
        np.sin: lambda x: sin(x),
        np.cos: lambda x: cos(x)
    }

    x_0 = symbols('x')
    expression = res.sympy_expression({f'x_0': x_0}, sympy_functions)
    print(expression)
    simplified_expression = simplify(expression)
    print(simplified_expression)
    ```
    This should result in  the following output equation:
    $$x^{2} + 10 \cos{\left (10 x \right )} + 10$$
    The simplified expression is the same as the original expression, as the expression is already in its simplest form. This is in fact the original equation used to generate the data. This can vary depending on the data and the parameters used. Getting the same equation as the original equation is not guaranteed.

## License
This project is released under the [MIT License](https://opensource.org/licenses/MIT).

