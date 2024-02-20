import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd

# Set the parameters
num_genes = 1000  # Number of variables (genes)
num_samples = 50  # Number of samples
mean_range = (-0.5, 0.5)

# Define the standard deviation vector
std_dev = np.ones(num_genes)

k = 0.6
r = 0.65

# Generate means from a uniform distribution
means = np.random.uniform(
    low=mean_range[0], high=mean_range[1], size=num_genes
)

# Initialize the correlation matrix
corr_matrix = np.zeros((num_genes, num_genes))  # with zeros

# Define the significant genes
l = np.random.choice(
    range(num_genes), size=int(0.2 * num_genes), replace=False
)

# Generate the modified means and correlations for significant genes
for i in l:
    means[i] += k

    for j in range(num_genes):
        if j in l:
            corr_matrix[i, j] += r

# For the non-significant genes, fill the diagonal with variances

# cov matrix is corr matrix_ih * std_i * std_h
cov_matrix = corr_matrix * np.outer(std_dev, std_dev)


# Generate the simulated expression data
simulated_data = multivariate_normal.rvs(
    mean=means, cov=cov_matrix, size=num_samples
)

# Convert the numpy array to a pandas DataFrame
df = pd.DataFrame(simulated_data)

# Save the DataFrame to a CSV file
df.to_csv("simulated_data_test.csv", index=False)

print("Data saved to simulated_data.csv")

print(simulated_data.shape)
