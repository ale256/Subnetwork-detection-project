import numpy as np


def compute_lambda_star(a, array_sum=1):
    # Step 1: Calculate the l1 norm of a
    norm_a = np.sum(np.abs(a))

    # Step 2: Check if norm_a <= 1
    if norm_a <= array_sum:
        return 0.0

    # Define a0 = 0 and sort |a_k| in ascending order
    a_sorted = np.sort(np.abs(a))
    a_sorted = np.insert(a_sorted, 0, 0)  # Insert a0 = 0 at the beginning

    # Step 3: Compute g'(λ)/2 at points λ = |a_0|, |a_1|, ..., |a_n|
    n = len(a)
    g_prime_by_2 = np.zeros(n + 1)
    for k in range(n + 1):
        if k == 0:
            g_prime_by_2[k] = norm_a - array_sum
        elif k == n:
            g_prime_by_2[k] = -array_sum
        else:
            g_prime_by_2[k] = (
                (k - n) * a_sorted[k] + np.sum(a_sorted[k + 1 :]) - array_sum
            )

    # Step 4: Locate the interval where g'(λ)/2 changes its sign
    k = 0
    while k < n and g_prime_by_2[k] >= 0:
        k += 1

    # Step 5: Compute λ*
    if k == 0:
        lambda_star = (np.sum(a_sorted[0:]) - array_sum) / n
    else:
        lambda_star = (np.sum(a_sorted[k:]) - array_sum) / (n - k + 1)

    return lambda_star


def soft_thres_l1(y, array_sum=1):
    lambda_val = compute_lambda_star(y, array_sum)
    return np.sign(y) * np.maximum(np.abs(y) - lambda_val, 0)


def soft_thres_l1_2d(arr, array_sum=1):
    return np.apply_along_axis(soft_thres_l1, 1, arr, array_sum)


if __name__ == "__main__":
    a = np.array([0.5, 0.2, -0.3, 0.4, -0.1])  # Example input vector
    lambda_star = compute_lambda_star(a)
    print(f"λ* = {lambda_star}")

    y = np.array(
        [0.5, 0.2, -0.3, 0.4, -0.1, 0.8, -0.9]
    )  # Example input vector
    y_soft = soft_thres_l1(y, 2)
    print(f"Soft-thresholded vector: {y_soft}")
    print(f"λ* = {compute_lambda_star(y_soft)}")
    print(f"Sum of the soft-thresholded vector: {np.sum(np.abs(y_soft))}")

    arr = np.array([[3, -1, 2], [-2, 5, -3], [0.5, 0.2, -0.8]])

    # Apply soft_thres_l1 to each row
    result = soft_thres_l1_2d(arr, array_sum=1)
    print(result)
