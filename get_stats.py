import pandas as pd


def get_stats(base_case, comparing_case):
    # Load the data from the CSV files
    base_data = pd.read_csv(base_case)
    comparing_data = pd.read_csv(comparing_case)

    num_genes = len(base_data.columns)
    assert len(base_data.columns) == len(
        comparing_data.columns
    ), "They don't have the same number of genes!"

    # get the length of the data
    n = len(base_data)
    m = len(comparing_data)
    assert n == 50, "The number of samples should be 50"
    assert m == 50, "The number of samples should be 50"

    # Calculate the sample means for each row
    mu_N = base_data.mean(axis=0).to_numpy()
    mu_C = comparing_data.mean(axis=0).to_numpy()
    assert len(mu_N) == 1000
    assert len(mu_C) == 1000

    sigma_N = base_data.std(axis=0).to_numpy()
    sigma_C = comparing_data.std(axis=0).to_numpy()

    # calculate sample correlations
    rho_N = base_data.corr().to_numpy()
    rho_C = comparing_data.corr().to_numpy()

    p_N = n / (n + m)
    p_C = m / (n + m)

    sigma_v = 0.084

    stats = {
        "num_genes": num_genes,
        "n": n,
        "m": m,
        "mu_N": mu_N,
        "mu_C": mu_C,
        "sigma_N": sigma_N,
        "sigma_C": sigma_C,
        "rho_N": rho_N,
        "rho_C": rho_C,
        "p_N": p_N,
        "p_C": p_C,
        "sigma_v": sigma_v,
    }

    return stats


if __name__ == "__main__":
    # Load the data and calculate the sample means
    file1 = "simulated_data/simulated_data_case0.csv"
    file2 = "simulated_data/simulated_data_case1.csv"
    stats = get_stats(file1, file2)
