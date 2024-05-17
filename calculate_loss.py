import numpy as np
from scipy.integrate import quad
from functools import partial
from get_stats import get_stats


def Ti(mu_Y_N, mu_Y_C, sigma_Y_N, sigma_Y_C, n, m, sigma_v):
    numerator = mu_Y_N - mu_Y_C
    denominator = (
        np.sqrt((sigma_Y_N**2 + sigma_Y_C**2) * (1 / n + 1 / m) / (n + m - 2))
        + sigma_v
    )
    return numerator / denominator


def T_Y_given_Z(
    mu_Y_N,
    mu_Y_C,
    sigma_Y_N,
    sigma_Y_C,
    rho_N,
    rho_C,
    mu_Z_N,
    mu_Z_C,
    sigma_Z_N,
    sigma_Z_C,
    n,
    m,
    sigma_v,
    z,
):
    # Numerator
    term1 = mu_Y_N + rho_N * ((z - mu_Z_N) / sigma_Z_N) * sigma_Y_N
    term2 = mu_Y_C + rho_C * ((z - mu_Z_C) / sigma_Z_C) * sigma_Y_C
    numerator = term1 - term2

    # Denominator
    denominator_inner = (
        sigma_Y_N**2 * (1 - rho_N**2) + sigma_Y_C**2 * (1 - rho_C**2)
    ) * (1 / n + 1 / m)
    denominator = np.sqrt(denominator_inner / (n + m - 2)) + sigma_v

    return numerator / denominator


def f_Z_N(mu_Z_N, sigma_Z_N, z):
    return (1 / np.sqrt(2 * np.pi * sigma_Z_N**2)) * np.exp(
        -((z - mu_Z_N) ** 2) / (2 * sigma_Z_N**2)
    )


def f_Z_C(mu_Z_C, sigma_Z_C, z):
    return (1 / np.sqrt(2 * np.pi * sigma_Z_C**2)) * np.exp(
        -((z - mu_Z_C) ** 2) / (2 * sigma_Z_C**2)
    )


def Mij(
    T_Y_given_Z, P_N, P_C, f_Z_N, f_Z_C, Z_limits
):  # T_Y_given_Z, f_Z_N, f_Z_C need to be functions

    integrand = lambda z: T_Y_given_Z(z) * (P_N * f_Z_N(z) + P_C * f_Z_C(z))
    integral_result, _ = quad(integrand, Z_limits[0], Z_limits[1])
    return integral_result


def get_params(stats):
    num_genes = stats["num_genes"]
    mu_N = stats["mu_N"]
    mu_C = stats["mu_C"]
    sigma_N = stats["sigma_N"]
    sigma_C = stats["sigma_C"]
    rho_N = stats["rho_N"]
    rho_C = stats["rho_C"]
    p_N = stats["p_N"]
    p_C = stats["p_C"]
    sigma_v = stats["sigma_v"]
    n = stats["n"]
    m = stats["m"]

    v = np.zeros(num_genes)
    W = np.zeros((num_genes, num_genes))

    # [ ] may want to optimize these for-loops in the future to speed up code
    for i in range(num_genes):
        v[i] = Ti(mu_N[i], mu_C[i], sigma_N[i], sigma_C[i], n, m, sigma_v)

    for j in range(num_genes):
        for k in range(num_genes):
            T_Y_given_Z_as_z = partial(
                T_Y_given_Z,
                mu_N[j],
                mu_C[j],
                sigma_N[j],
                sigma_C[j],
                rho_N[j, k],
                rho_C[j, k],
                mu_N[k],
                mu_C[k],
                sigma_N[k],
                sigma_C[k],
                n,
                m,
                sigma_v,
            )
            f_Z_N_as_z = partial(f_Z_N, mu_N[k], sigma_N[k])
            f_Z_C_as_z = partial(f_Z_C, mu_C[k], sigma_C[k])

            Z_upper = max(mu_N[k] + 3 * sigma_N[k], mu_C[k] + 3 * sigma_C[k])
            Z_lower = min(mu_N[k] - 3 * sigma_N[k], mu_C[k] - 3 * sigma_C[k])
            Z_limits = (Z_lower, Z_upper)

            W[j, k] = Mij(
                T_Y_given_Z_as_z, p_N, p_C, f_Z_N_as_z, f_Z_C_as_z, Z_limits
            )

    return v, W


if __name__ == "__main__":

    base_case = "simulated_data/simulated_data_case0.csv"
    comparing_case = "simulated_data/simulated_data_case1.csv"

    stats = get_stats(base_case, comparing_case)
    v, W = get_params(stats)
    print(v.shape, W.shape)
