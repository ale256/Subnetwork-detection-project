import numpy as np


# Function to generate a random vector x of length 1000 with entries summing to 1
def generate_random_vector(length=1000):
    x = np.random.rand(length)
    x /= x.sum()
    return x


# Function to perform the task
def compute_lambda(W, v, num_samples=1000, seed=42):
    np.random.seed(seed)
    edge_scores = []
    node_scores = []

    # Generate a large number of random vectors and calculate scores
    for _ in range(num_samples):
        x = generate_random_vector()
        edge_score = np.dot(x.T, np.dot(W, x))  # x^T W x
        node_score = np.dot(v, x)  # v^T x
        edge_scores.append(edge_score)
        node_scores.append(node_score)

    # Convert to numpy arrays for mean and std calculations
    edge_scores = np.array(edge_scores)
    node_scores = np.array(node_scores)

    # Compute means and standard deviations
    mu_W = np.mean(edge_scores)
    mu_v = np.mean(node_scores)
    s_W = np.std(edge_scores)
    s_v = np.std(node_scores)

    # Compute magnitudes
    M_W = mu_W / s_W
    M_v = mu_v / s_v

    # Set lambda
    lambda_value = M_W / (M_W + M_v)

    return lambda_value


if __name__ == "__main__":
    # Example usage

    for idx in range(1, 5):
        W = np.load(f"output/sparse_v_W/W_case{idx}.npy")
        v = np.load(f"output/sparse_v_W/v_case{idx}.npy")

        num_samples = 1000
        lambda_value = compute_lambda(W, v, num_samples=num_samples)

        print("Lambda:", lambda_value)
