import numpy as np
from scipy.integrate import quad
from functools import partial


def Ti(mu_Y_N, mu_Y_C, sigma_Y_N, sigma_Y_C, n, m, sigma_v):
    numerator = mu_Y_N - mu_Y_C
    denominator = (
        np.sqrt((sigma_Y_N**2 + sigma_Y_C**2) * (1 / n + 1 / m) / (n + m - 2))
        + sigma_v
    )
    return numerator / denominator


def T_Y_given_Z(
    z,
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


def f_Z_N(z, mu_Z_N, sigma_Z_N):
    return (1 / np.sqrt(2 * np.pi * sigma_Z_N**2)) * np.exp(
        -((z - mu_Z_N) ** 2) / (2 * sigma_Z_N**2)
    )


def f_Z_C(z, mu_Z_C, sigma_Z_C):
    return (1 / np.sqrt(2 * np.pi * sigma_Z_C**2)) * np.exp(
        -((z - mu_Z_C) ** 2) / (2 * sigma_Z_C**2)
    )


def Mij(
    T_Y_given_Z, P_N, P_C, f_Z_N, f_Z_C, Z_limits
):  # T_Y_given_Z, f_Z_N, f_Z_C need to be functions

    integrand = lambda z: T_Y_given_Z(z) * (P_N * f_Z_N(z) + P_C * f_Z_C(z))
    integral_result, _ = quad(integrand, Z_limits[0], Z_limits[1])
    return integral_result


if __name__ == "__main__":

    # Constants
    P_N = 0.5
    P_C = 0.5
    Z_limits = (-np.inf, np.inf)

    # Calculate M_ij
    M_ij = calculate_Mij(T_Y_given_Z, P_N, P_C, f_Z_N, f_Z_C, Z_limits)
    print(f"M_ij: {M_ij}")

    # Values for T_i calculation
    mu_Y_N = 1.0
    mu_Y_C = 0.5
    sigma_Y_N = 1.0
    sigma_Y_C = 1.0
    n = 30
    m = 30
    sigma_v = 0.1

    # Calculate T_i
    T_i = calculate_Ti(mu_Y_N, mu_Y_C, sigma_Y_N, sigma_Y_C, n, m, sigma_v)
    print(f"T_i: {T_i}")
