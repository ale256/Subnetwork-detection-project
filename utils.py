import numpy as np
from scipy.stats import norm


def calculate_modified_t_stat(
    mean_normal, mean_cancer, std_dev_normal, std_dev_cancer, n, m, sigma_y
):
    """
    Calculate the modified T-statistic for differential gene expression.

    Parameters:
    mean_normal (float): The mean expression of the gene in the normal condition.
    mean_cancer (float): The mean expression of the gene in the cancer condition.
    std_dev_normal (float): The standard deviation of the gene expression in the normal condition.
    std_dev_cancer (float): The standard deviation of the gene expression in the cancer condition.
    n (int): The number of samples in the normal condition.
    m (int): The number of samples in the cancer condition.
    sigma_y (float): A constant chosen to minimize the coefficient of variation of T-statistic.

    Returns:
    float: The modified T-statistic for the gene.
    """

    numerator = mean_normal - mean_cancer
    denominator = (
        np.sqrt(
            ((std_dev_normal**2) + (std_dev_cancer**2))
            * (1 / n + 1 / m)
            / (n + m - 2)
        )
        + sigma_y
    )
    t_statistic = numerator / denominator

    return t_statistic


def calculate_M(
    mean_Y_normal,
    mean_Y_cancer,
    std_Y_normal,
    std_Y_cancer,
    mean_Z_normal,
    mean_Z_cancer,
    std_Z_normal,
    std_Z_cancer,
    corr_N,
    corr_C,
    sigma_y,
    n,
    m,
):
    """
    Calculate the expectation M(T_YZ|Z=z) which is the connectivity relationship between gene Y and gene Z.

    Parameters:
    mean_Y_normal, mean_Y_cancer: means of gene Y in normal and cancer conditions
    std_Y_normal, std_Y_cancer: standard deviations of gene Y in normal and cancer conditions
    mean_Z_normal, mean_Z_cancer: means of gene Z in normal and cancer conditions
    std_Z_normal, std_Z_cancer: standard deviations of gene Z in normal and cancer conditions
    pN, pC: probabilities that a sample is selected from the normal or cancer conditions
    sigma_y: a constant chosen to minimize the coefficient of variation of T-statistic
    n, m: number of samples in normal and cancer conditions

    Returns:
    float: The expectation M(T_YZ|Z=z) value.
    """

    # Define the integrand for the expectation calculation
    def integrand(z):
        T_YZ_z = (mean_Y_normal + corr_N * (z - mean_Z_normal)) * (
            std_Y_normal / std_Z_normal
        ) - (mean_Y_cancer - corr_C * (z - mean_Z_cancer)) * (
            std_Y_cancer / std_Z_cancer
        )
        T_YZ_z /= (
            np.sqrt(
                (
                    std_Y_normal**2 * (1 - corr_N**2)
                    + std_Y_cancer**2 * (1 - corr_C**2)
                )
                * (1 / n + 1 / m)
                / (n + m - 2)
            )
            + sigma_y
        )

        # Probability density function values for Z in normal and cancer conditions
        fzN = norm.pdf(z, mean_Z_normal, std_Z_normal)
        fzC = norm.pdf(z, mean_Z_cancer, std_Z_cancer)

        # Combined PDF for Z under both conditions
        pN = n / (n + m)
        pC = m / (n + m)
        fz = pN * fzN + pC * fzC

        return T_YZ_z * fz

    # Calculate the expectation by numerical integration
    z_values = np.linspace(
        min(
            mean_Z_normal - 3 * std_Z_normal, mean_Z_cancer - 3 * std_Z_cancer
        ),
        max(
            mean_Z_normal + 3 * std_Z_normal, mean_Z_cancer + 3 * std_Z_cancer
        ),
        1000,
    )
    M_value = np.trapz(integrand(z_values), z_values)

    return M_value


def optimization_objective(x, W, v, lambda_value):
    """
    Compute the value of the optimization objective function.

    Parameters:
    x (numpy.ndarray): The vector of variables in the optimization problem.
    W (numpy.ndarray): The symmetric matrix representing edge scores between genes.
    v (numpy.ndarray): The vector representing node scores for differential expression of genes.
    lambda_value (float): The lambda parameter to balance the two terms in the objective function.

    Returns:
    float: The value of the objective function for given x.
    """

    # Compute the quadratic term
    quadratic_term = x.T @ W @ x

    # Compute the linear term
    linear_term = v.T @ x

    # Combine the terms with the lambda parameter
    objective_value = (
        lambda_value * quadratic_term + (1 - lambda_value) * linear_term
    )

    return objective_value
