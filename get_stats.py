import pandas as pd


def load_data_and_calculate_means(base_case, comparing_case):
    # Load the data from the CSV files
    base_data = pd.read_csv(base_case)
    comparing_data = pd.read_csv(comparing_case)

    # get the length of the data
    n = len(base_data)
    m = len(comparing_data)
    assert n == 50, "The number of samples should be 50"
    assert m == 50, "The number of samples should be 50"

    # Calculate the sample means for each row
    mu_N = base_data.mean(axis=0)
    mu_C = comparing_data.mean(axis=0)
    assert len(mu_N) == 1000
    assert len(mu_C) == 1000

    sigma_N = base_data.std(axis=0)
    sigma_C = comparing_data.std(axis=0)

    # calculate sample correlations
    rho_N = base_data.corr()
    rho_C = comparing_data.corr()
    print(rho_N.shape)

    return


if __name__ == "__main__":
    # Load the data and calculate the sample means
    file1 = "simulated_data/simulated_data_case0.csv"
    file2 = "simulated_data/simulated_data_case1.csv"
    means1, means2 = load_data_and_calculate_means(file1, file2)

    print(f"Means for case0: {means1.values}")
    print(f"Means for case1: {means2.values}")
