import numpy as np
import pygad
from hyperparameters import LAMBDA
from functools import partial


# Define the objective function
def fit_func(lambda_, W, v, ga_instance, solution, solution_idx):
    x = np.array(solution)
    term1 = lambda_ * np.dot(x.T, np.dot(W, x))
    term2 = (1 - lambda_) * np.dot(v.T, x)
    loss = term1 + term2
    return loss


def on_generation(ga_instance):
    for sol_idx, solution in enumerate(ga_instance.population):
        # Ensure non-negativity
        solution = np.maximum(solution, 0)
        ga_instance.population[sol_idx] = solution
    print(
        f"Generation {ga_instance.generations_completed}: Best Fitness = {ga_instance.best_solution()[1]}"
    )


if __name__ == "__main__":

    W = np.load("W_case1.npy")
    v = np.load("v_case1.npy")
    fitness_func = lambda ga_instance, solution, solution_idx: fit_func(
        LAMBDA, W, v, ga_instance, solution, solution_idx
    )

    # GA parameters
    num_iterations = 60000  # Set the number of iterations
    num_generations = num_iterations  # Assuming one generation per iteration
    num_parents_mating = 10
    sol_per_pop = 20
    num_genes = len(W)  # Ensure num_genes matches the dimension of W and v
    mutation_rate = 1 / (num_genes + 1) + 1  # Set mutation rate
    crossover_rate = 0.5  # Set crossover rate

    # Creating an instance of the GA
    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        mutation_type="random",
        mutation_percent_genes=20,  # pygad uses percentage
        crossover_type="single_point",  # Experiment with different crossover methods
        crossover_probability=crossover_rate,  # Set crossover rate
        on_generation=on_generation,
        stop_criteria=[
            "saturate_100"
        ],  # Stop if no improvement for 100 generations
        keep_parents=1,
        initial_population=np.random.uniform(
            low=0.0, high=1.0, size=(sol_per_pop, num_genes)
        ),  # Ensure initial population in [0,1]
    )
    # Running the GA
    ga_instance.run()

    # Best solution
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Best Solution: ", solution)
    print("Best Solution Fitness: ", solution_fitness)
