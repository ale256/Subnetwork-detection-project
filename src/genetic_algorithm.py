import numpy as np
import sys

sys.path.append("src/")
import pygad
from calcu_lambda import compute_lambda
from functools import partial
from soft_thres import soft_thres_l1, soft_thres_l1_2d
import sys


# Define the objective function
def fit_func(W, v, lambda_, ga_instance, solution, solution_idx):

    x = np.array(solution)

    term1 = lambda_ * np.dot(x.T, np.dot(W, x))
    term2 = (1 - lambda_) * np.dot(v.T, x)
    loss = term1 + term2
    return loss


def on_generation(ga_instance):
    # for sol_idx, solution in enumerate(ga_instance.population):
    # solution = np.maximum(solution, 0)
    # solution = np.minimum(solution, 1)
    # solution = soft_thres_l1(solution)
    # ga_instance.population[sol_idx] = solution
    print(
        f"Generation {ga_instance.generations_completed}: Best Fitness = {ga_instance.best_solution()[1]}"
    )
    # Print the best solution after each generation
    best_solution, best_solution_fitness, best_solution_idx = (
        ga_instance.best_solution()
    )
    # print(f"Best Solution (index {best_solution_idx}): {best_solution.sum()}")

    evaluation(best_solution)
    # print(f"Best Solution (index {best_solution_idx}): {best_solution[:5]}")
    # print(f"Best Solution Fitness: {best_solution_fitness}")


def on_mutation(ga_instance, offspring):

    for sol_idx, solution in enumerate(offspring):
        solution = np.maximum(solution, 0)
        solution = np.minimum(solution, 1)
        solution = soft_thres_l1(solution)
        offspring[sol_idx] = solution
    return offspring


def create_normalized_population(sol_per_pop, num_genes):
    population = np.random.uniform(
        low=0.0, high=1.0, size=(sol_per_pop, num_genes)
    )
    for idx in range(sol_per_pop):
        population[idx] = soft_thres_l1(population[idx])
    return population


def scramble_mutation(offspring, ga_instance):
    mutation_rate = ga_instance.mutation_probability  # Get the mutation rate
    for i in range(len(offspring)):
        if np.random.rand() < mutation_rate:
            # Select a random subset to scramble
            start, end = sorted(
                np.random.choice(
                    range(ga_instance.num_genes), 2, replace=False
                )
            )
            if start != end:
                np.random.shuffle(offspring[i, start : end + 1])
        if np.random.rand() < mutation_rate:
            # gene_idx = np.random.randint(0, ga_instance.num_genes)

            indices = np.random.choice(
                range(ga_instance.num_genes), 20, replace=False
            )
            offspring[i, indices] = np.random.uniform(
                0.0, 1.0, size=len(indices)
            )
            offspring[i] = soft_thres_l1(offspring[i])
        pass

    return offspring


# def arithmetic_crossover(parents, offspring_size, ga_instance):
#     offspring = np.empty(offspring_size)
#     for k in range(offspring_size[0]):
#         parent1_idx = k % parents.shape[0]
#         parent2_idx = (k + 1) % parents.shape[0]
#         alpha = np.random.uniform(0, 1, size=offspring_size[1])
#         # alpha = np.random.uniform(0, 1)
#         offspring[k, :] = (
#             alpha * parents[parent1_idx, :]
#             + (1 - alpha) * parents[parent2_idx, :]
#         )
#         offspring[k, :] = soft_thres_l1(offspring[k, :])

#         # # Print the selected parents and the resulting offspring
#         # print(f"Parent 1 (index {parent1_idx}): {parents[parent1_idx, :5]}")
#         # print(f"Parent 2 (index {parent2_idx}): {parents[parent2_idx, :5]}")
#         # print(f"Offspring (index {k}): {offspring[k, :5]}")

#     return offspring


def arithmetic_crossover(parents, offspring_size, ga_instance):
    parent1_idx = np.arange(offspring_size[0]) % parents.shape[0]
    parent2_idx = (np.arange(offspring_size[0]) + 1) % parents.shape[0]

    alpha = np.random.uniform(
        0, 1, size=(offspring_size[0], offspring_size[1])
    )

    offspring = (
        alpha * parents[parent1_idx, :] + (1 - alpha) * parents[parent2_idx, :]
    )

    # Apply soft_thres_l1 row-wise
    offspring = soft_thres_l1_2d(offspring)

    return offspring


def evaluation(solution):
    label = np.load("simulated_data/significant_genes.npy")
    label.sort()
    # print(label)

    solution_soft = soft_thres_l1(solution, 1)
    # print("Soft-thresholded Solution: ", solution_soft.sum())
    # print non-zero elements as list
    # print("Non-zero elements: ", np.nonzero(solution_soft)[0])
    indices = np.nonzero(solution_soft)[0]

    # calculate precision and recal and f1 of indices and label
    TP = len(set(indices) & set(label))
    FP = len(set(indices) - set(label))
    FN = len(set(label) - set(indices))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")


if __name__ == "__main__":
    np.random.seed(42)
    for idx in range(1, 2):
        W = np.load(f"output/sparse_v_W/W_case{idx}.npy")
        v = np.load(f"output/sparse_v_W/v_case{idx}.npy")
        lambda_ = compute_lambda(W, v, num_samples=1000)
        # print(f"Lambda: {lambda_}")
        fitness_func = lambda ga_instance, solution, solution_idx: fit_func(
            W, v, lambda_, ga_instance, solution, solution_idx
        )

        # GA parameters
        num_iterations = 10000  # Set the number of iterations # 60000
        num_generations = (
            num_iterations  # Assuming one generation per iteration
        )
        parents_mating_rate = 0.5
        keep_parents_rate = 0.2
        sol_per_pop = 1000
        num_parents_mating = int(sol_per_pop * parents_mating_rate)
        keep_parents = int(sol_per_pop * keep_parents_rate)
        num_genes = len(W)  # Ensure num_genes matches the dimension of W and v
        # mutation_rate = 100 / (num_genes + 1)  # Set mutation rate
        mutation_rate = 0.1
        crossover_rate = 0.8  # Set crossover rate
        initial_population = create_normalized_population(
            sol_per_pop, num_genes
        )

        # Creating an instance of the GA
        ga_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            fitness_func=fitness_func,
            sol_per_pop=sol_per_pop,
            num_genes=num_genes,
            mutation_type=scramble_mutation,
            # mutation_type="random",
            mutation_probability=mutation_rate,  # pygad uses percentage
            crossover_type=arithmetic_crossover,  # Experiment with different crossover methods
            crossover_probability=crossover_rate,  # Set crossover rate
            on_generation=on_generation,
            stop_criteria=[
                "saturate_100"
            ],  # Stop if no improvement for 100 generations
            keep_parents=keep_parents,
            initial_population=initial_population,
        )
        # Running the GA
        ga_instance.run()

        # Best solution
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print("Best Solution: ", solution)
        print("Best Solution Fitness: ", solution_fitness)

        # Save the best solution
        # np.save(f"output/best_solution_case{idx}.npy", solution)

        evaluation(solution)
        print("Non-zero elements: ", np.nonzero(solution)[0])
