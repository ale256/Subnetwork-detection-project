import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd
import os


def generate_data(
    k,
    r,
    significant_genes,
    save_name,
    num_genes=1000,
    num_samples=50,
    data_range=(-0.5, 0.5),
    seed=0,
):
    np.random.seed(seed)
    # Generate means from a uniform distribution
    means = np.random.uniform(
        low=data_range[0], high=data_range[1], size=num_genes
    )
    # Generate the standard deviation matrix
    std_dev = np.zeros((num_genes, num_genes))
    np.fill_diagonal(
        std_dev,
        np.random.uniform(low=0.01, high=data_range[1], size=num_genes),
    )

    # Initialize the correlation matrix
    corr_matrix = np.eye(num_genes)  # with zeros

    # Define the significant genes
    l = significant_genes

    # Generate the modified means and correlations for significant genes
    for i in l:
        means[i] += k
        for j in l:
            if i != j:
                corr_matrix[i, j] += r

    cov_matrix = std_dev.dot(corr_matrix).dot(std_dev)
    # Generate the simulated expression data
    simulated_data = multivariate_normal.rvs(
        mean=means, cov=cov_matrix, size=num_samples
    )
    clipped_data = np.clip(simulated_data, data_range[0], data_range[1])

    # Save as csv
    df = pd.DataFrame(clipped_data)
    os.makedirs("simulated_data", exist_ok=True)
    df.to_csv(f"simulated_data/simulated_data_{save_name}.csv", index=False)
    print(f"Data saved to simulated_data/simulated_data_{save_name}.csv")


if __name__ == "__main__":
    case_parameters = {
        "case0": (0, 0),  # (k, r)
        "case1": (0.6, 0.65),
        "case2": (0.3, 0.65),
        "case3": (0.6, 0.35),
        "case4": (0.3, 0.35),
    }
    seed = 0
    num_genes = 1000
    np.random.seed(seed)
    significant_genes = np.random.choice(
        range(num_genes), size=int(0.2 * num_genes), replace=False
    )

    for case, (k, r) in case_parameters.items():
        generate_data(
            k, r, significant_genes, case, num_genes=num_genes, seed=seed
        )
