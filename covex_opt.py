import cvxpy as cp


def optimization_objective(W, v, lambda_value):
    """
    Sets up and solves the optimization problem.

    Parameters:
    W (numpy.ndarray): The symmetric matrix of edge scores.
    v (numpy.ndarray): The vector of node scores.
    lambda_value (float): The balance parameter.

    Returns:
    numpy.ndarray: The optimal x vector that maximizes the objective.
    """

    # Number of genes (nodes)
    k = len(v)

    # Ensure W is symmetric
    W = (W + W.T) / 2

    # Define the optimization variable
    x = cp.Variable(k, nonneg=True)

    # Define the objective function
    objective = cp.Maximize(
        lambda_value * cp.quad_form(x, W)
        + (1 - lambda_value) * cp.sum(cp.multiply(v, x))
    )

    # Define the constraints
    constraints = [cp.sum(x) == 1]

    # Formulate the problem
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    problem.solve()

    # Return the solution
    return x.value
