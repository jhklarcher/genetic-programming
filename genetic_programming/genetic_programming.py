import copy
import random
import numpy as np

random.seed(42)


class Node:
    def __init__(self, value, children=[]):
        self.value = value
        self.children = children

    def evaluate(self, variables):
        if self.children == []:
            # If it's a terminal node, return its value.
            # If the terminal is a variable, return the variable's value.
            return variables.get(self.value, self.value)
        else:
            # If it's a function node, first evaluate its children.
            # Evaluate children nodes with the variables dictionary
            evaluated_children = [child.evaluate(
                variables) for child in self.children]

            # Apply the function to the evaluated children
            return self.value(*evaluated_children)

    def expression(self):
        if self.children == []:
            # If it's a terminal node, return its value.
            return str(self.value)
        else:
            # If it's a function node, first get the expression of its children.
            return f'{self.value.__name__}({", ".join([child.expression() for child in self.children])})'

    def depth(self):
        if self.children == []:
            return 0
        else:
            return 1 + max([child.depth() for child in self.children])

    def sympy_expression(self, variables, functions):
        # Returns the sympy expression of the tree
        if self.children == []:
            # If it's a terminal node, return its value.
            return variables.get(self.value, self.value)
        else:
            # If it's a function node, first get the expression of its children.
            return functions[self.value](*[child.sympy_expression(variables, functions) for child in self.children])


def generate_tree(functions, terminals, max_depth=None, depth=0):
    if (max_depth and depth == max_depth) or (depth > 0 and random.random() < 0.5) or (max_depth == 0):
        return Node(random.choice(terminals))
    else:
        func, arity = random.choice(list(functions.items()))
        children = [generate_tree(
            functions, terminals, max_depth, depth + 1) for _ in range(arity)]
        return Node(func, children)


def select_random_node(tree, max_depth=None):
    nodes = []
    stack = [tree]

    while stack:
        current_node = stack.pop()
        nodes.append(current_node)

        if current_node.children != []:
            # Reverse to maintain order
            stack.extend(reversed(current_node.children))

    return random.choice([node for node in nodes if (max_depth is None or node.depth() <= max_depth)])


def tournament_selection(population, fitnesses, tournament_size=5):
    # Select k individuals from the population at random
    tournament = random.choices(population, k=tournament_size)
    # Return the best individual from the tournament
    return sorted(tournament, key=lambda individual: fitnesses[population.index(individual)])[0]


def random_node_crossover(tree1, tree2):
    tree1 = copy.deepcopy(tree1)
    tree2 = copy.deepcopy(tree2)
    node1 = select_random_node(tree1)
    node2 = select_random_node(tree2)

    # Copy of node2
    node2 = copy.deepcopy(node2)

    # Swap the selected node from tree1 with the selected node from tree2
    node1.value, node2.value = node2.value, node1.value
    node1.children, node2.children = node2.children, node1.children

    return tree1


def max_depth_crossover(tree1, tree2):
    tree1 = copy.deepcopy(tree1)
    tree2 = copy.deepcopy(tree2)
    node1 = select_random_node(tree1)
    node2 = select_random_node(tree2, max_depth=node1.depth())

    # Copy of node2
    node2 = copy.deepcopy(node2)

    # Swap the selected node from tree1 with the selected node from tree2
    node1.value, node2.value = node2.value, node1.value
    node1.children, node2.children = node2.children, node1.children

    return tree1


def tree_mutation(tree, functions, terminals):
    node = select_random_node(tree)
    new_node = generate_tree(functions, terminals, max_depth=node.depth())
    node.value, node.children = new_node.value, new_node.children


def value_mutation(tree, functions, terminals):
    node = select_random_node(tree)
    if node.children == []:
        node.value = random.choice(terminals)
    else:
        # Find the arity of the function
        arity = functions[node.value]
        # Find the new function with the same arity
        new_func = random.choice(
            [func for func, func_arity in functions.items() if func_arity == arity])
        node.value = new_func


cache = {}
cache_size = 1000

def fitness(tree, X, y, depth_coefficient):
    # Check if the tree has already been evaluated
    if tree.expression() in cache:
        return cache[tree.expression()]

    # Evaluate the tree for each row in X
    y_pred = [tree.evaluate(
        {f'x_{i}': x_i for i, x_i in enumerate(row)}) for row in X]
    y_pred = np.array(y_pred).reshape(-1, 1)

    # Calculate the mean squared error
    mse = np.mean((y - y_pred)**2)

    res = mse + tree.depth() * depth_coefficient

    # Cache the result
    cache[tree.expression()] = res

    # If the cache is full, remove the oldest entry
    if len(cache) > cache_size:
        cache.pop(list(cache.keys())[0])

    return res


def gp_minimize(X, y, functions, terminals, max_depth=None, population_size=100, n_generations=100, crossover_probability=0.5, mutation_probability=0.1, depth_coefficient=0.1, tournament_size=5):
    # Generate the initial population
    population = [generate_tree(functions, terminals, max_depth)
                  for _ in range(population_size)]

    for _ in range(n_generations):

        # Create the next generation
        new_population = []

        while len(new_population) < population_size:
            # Select two random parents
            if tournament_size in [0, 1]:
                parent1, parent2 = random.choices(population, k=2)
            else:
                # Select two parents using tournament selection
                fitnesses = [fitness(tree, X, y, depth_coefficient)
                             for tree in population]
                indexes_1 = np.random.choice(
                    len(population), size=tournament_size, replace=False)
                winner_1 = np.argmin([fitnesses[i] for i in indexes_1])
                parent1 = population[indexes_1[winner_1]]
                indexes_2 = np.random.choice(
                    len(population), size=tournament_size, replace=False)
                winner_2 = np.argmin([fitnesses[i] for i in indexes_2])
                parent2 = population[indexes_2[winner_2]]

            # With crossover_probability cross them over to generate two children
            if random.random() < crossover_probability:
                if max_depth:
                    child1 = max_depth_crossover(parent1, parent2)
                else:
                    child1 = random_node_crossover(parent1, parent2)
            else:
                child1 = copy.deepcopy(parent1)

            # With mutation_probability mutate the children
            if random.random() < mutation_probability:
                tree_mutation(child1, functions, terminals)
                # value_mutation(child1, functions, terminals)

            new_population.extend([child1])

        population = population + new_population
        fitnesses = [fitness(tree, X, y, depth_coefficient)
                     for tree in population]
        orderd_indexes = np.argsort(fitnesses)
        population = [population[i] for i in orderd_indexes[:population_size]]
        print(f'Generation: {_}, Best fitness: {min(fitnesses)}')

    # Return the best individual
    return sorted(population, key=lambda tree: fitness(tree, X, y, depth_coefficient))[-1]

