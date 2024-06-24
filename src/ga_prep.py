import numpy as np
import os


def sparse_v_W(v, W):
    mean_v = np.mean(v)
    v[v < mean_v] = 0

    # for W, calculate row mean
    row_mean_W = np.mean(W, axis=1)
    mean_W = np.max(row_mean_W)
    W[W < mean_W] = 0

    return v, W


if __name__ == "__main__":
    for idx in range(1, 5):
        W = np.load(f"output/v_W/W_case{idx}.npy")
        v = np.load(f"output/v_W/v_case{idx}.npy")
        v, W = sparse_v_W(v, W)

        os.makedirs("output/sparse_v_W", exist_ok=True)
        np.save(f"output/sparse_v_W/v_case{idx}.npy", v)
        np.save(f"output/sparse_v_W/W_case{idx}.npy", W)
