"""Munkres Algorithm implementation (Hungarian Algorithm)"""

from typing import Sequence, Callable, Any

# NOTE If using numpy this should not be needed
from math import isnan

__EPS = 1e-9  # Floating point tolerance


def make_cost_matrix(
    workers: Sequence,
    jobs: Sequence,
    cost_function: Callable[[Any, Any, int, int], float],
) -> list[list[float]]:
    """
    Utility function to create a cost matrix by calculating the cost of pairing every
    element from the workers sequence with every element from the jobs sequence.

    The resulting matrix M will have shape (len(workers), len(jobs)), where
    M[i][j] = cost_function(workers[i], jobs[j], i, j).

    #### Parameters
    - `workers` (***Sequence[Any]***) Sequence of workers (defines the matrix rows).
    - `jobs` (***Sequence[Any]***) Sequence of jobs (defines the matrix columns).
    - `cost_function` (***Callable[[Any, Any, int, int], float]***) The function
      used to calculate the cost for a specific worker-job pairing.

    ##### Cost function signature
    `cost_function` will accept 4 positional arguments in this order:

    - `value_a`: The specific value from the `workers` Sequence (workers[i]).
    - `value_b`: The specific value from the `jobs` Sequence (jobs[j]).
    - `index_i`: The index `i` of the task value in the `workers` Sequence (row index).
    - `index_j`: The index `j` of the job value in the `jobs` Sequence (column index).

    And it will return a floating point value (or an integer) that will represent the cost of assigning worker a to job b.
    """

    cost_matrix = []
    for i, a in enumerate(workers):
        row = []
        for j, b in enumerate(jobs):
            row.append(cost_function(a, b, i, j))
        cost_matrix.append(row)
    return cost_matrix


def munkres(
    cost_matrix: list[list[float]], pad_cost: float = 0
) -> tuple[list[int], list[int], bool]:
    """
    Computes the minimum cost bipartite matching on a rectangular (N x M) cost matrix.

    #### Parameters

    ##### Cost matrix

    - The cost matrix is a 2-D list (N x M) where cost[i][j] is the numeric cost of assigning row i to column j.
    - Rows typically represent "agents" (e.g., workers), columns represent "tasks" (e.g., jobs); the algorithm finds assignments that minimize the total cost.
    - Entries can be integers or floats. When there are fewer jobs than workers, or vice versa, the resulting rectangular cost matrix is ​​filled with predefined costs, which in most cases can be 0, but can also be a specific value that best fits the problem.

    ##### Padding cost

    `pad_cost` is an optional floating point value (defaults to 0) that will be used as the default cost for padded assignments.
    You can usually leave this to default for most of the problems, but you can also choose a specific value that best fits your problem.

    #### Return values

    - `assignments` (***list[int]***): `assignments[i] = j` if the worker at row `i` is assigned to the job at column `j`, or `-1` if **unassigned** or assigned to a nonexistent job/column.
    - `inversions` (***list[int]***): `inversions[j] = i` if the job at column `j` is assigned to the worker at row `i`, or `-1` if **free** or assigned to a nonexistent worker/row.
    - `is_optimal` (***bool***): Indicates whether the algorithm's potentials certify optimality.
    """

    # Get the dimensions of the cost matrix
    N = len(cost_matrix)

    assert N > 0, "Empty cost matrix"

    M = len(cost_matrix[0])

    assert M > 0, "Cost matrix has no columns (jobs)"

    # Base 1->1 case
    if N == 1 and M == 1:
        return [0], [0], True

    # Get padding dimensions
    PAD_N = max(N, M)
    PAD_M = PAD_N

    # Calculate potentials U (minimum for each row)
    u = []
    for i in range(PAD_N):
        if i >= N:
            u.append(pad_cost)
            continue

        row_min = min(cost_matrix[i][j] if j < M else pad_cost for j in range(PAD_M))
        if isnan(row_min):
            row_min = 0
        u.append(row_min)

    # Calculate potentials V
    # (minimum for each column - u[i])
    v = []
    for j in range(PAD_M):
        col_min = float("inf")
        for i in range(PAD_N):
            if i < N and j < M:
                cost = cost_matrix[i][j] - u[i]
            else:
                # If either the row or column index are in the padding zone, use padding cost
                cost = pad_cost - (u[i] if i < N else pad_cost)
            if isnan(cost):
                cost = 0
            col_min = min(col_min, cost)

        v.append(col_min)

    # Initialize (padded) assignments (Z[i] -> j)
    Z = [-1] * PAD_N

    # Initialize (padded) inversion vector (inversions[j] -> i)
    inversions = [-1] * PAD_M

    # Iterate over unassigned (padded) rows
    for i in range(PAD_N):

        # Run augmented path search for this row
        path_found = False
        while not path_found:

            # Initialize alternated path
            S = set((i,))
            T = set()

            # Walk through the alternated path to find an augmented path
            path, path_found = __search_augmented_path(
                i,
                cost_matrix,
                N,
                M,
                PAD_M,
                pad_cost,
                inversions,
                u,
                v,
                S,
                T,
            )
            if not path_found:
                # Calculate delta
                # (minimum reduced cost considering all visited rows,
                #  and all unvisited columns in the alternated path)
                delta = min(
                    [
                        __reduced_cost(cost_matrix, u, v, row, col, N, M, pad_cost)
                        for row in S  # All visited rows
                        for col in range(PAD_M)
                        if col not in T  # All unvisited columns
                    ]
                    or [0]
                )

                if abs(delta) < __EPS:
                    # In theory this should not happen, floating-point rounding or pathological matrices could trigger this.
                    # if it happens return assigments with path_found = False
                    break

                # Update potentials
                for row_i, u_i in enumerate(u):
                    new_u = u_i + delta if row_i in S else u_i
                    if isnan(new_u):
                        new_u = 0
                    u[row_i] = new_u

                for col_j, v_j in enumerate(v):
                    new_v = v_j - delta if col_j in T else v_j
                    if isnan(new_v):
                        new_v = 0
                    v[col_j] = new_v
                continue

            # Walk through augmented path and invert each edge to assign this row
            for edge in path:
                row_i, col_j = edge

                # Check if this edge represents an assignment
                if Z[row_i] == col_j:
                    # Invert the assignment
                    Z[row_i] = -1
                    inversions[col_j] = -1
                else:
                    # Perform assignment
                    Z[row_i] = col_j
                    inversions[col_j] = row_i

    # Returns assignments, inversions, and a flag indicating whether the solution is optimal.
    # Padded assignments and inversions will be reduced to the actual size of the input.
    # Optimality checking considers padded assignments.
    return (
        [assignment if assignment < M else -1 for assignment in Z[:N]],
        [inversion if inversion < N else -1 for inversion in inversions[:M]],
        __optimality_check(cost_matrix, Z, u, v, N, M, pad_cost),
    )


def __optimality_check(
    cost_matrix: list[list[float]],
    assignments: list[int],
    u_potentials: list[float],
    v_potentials: list[float],
    N: int,
    M: int,
    pad_cost: float,
) -> bool:
    # For the solution to be optimal:
    # The sum of potentials must be equal the sum of the total cost of assignments
    u_sum = 0
    v_sum = 0
    cost_sum = 0
    for i, j in enumerate(assignments):
        if j == -1:
            continue
        u_sum += u_potentials[i]
        v_sum += v_potentials[j]

        # Use padding cost if this cell is a padded cell
        cost_sum += cost_matrix[i][j] if i < N and j < M else pad_cost

        # Also each reduced cost generated by the assignment must be 0
        if (
            abs(
                __reduced_cost(
                    cost_matrix, u_potentials, v_potentials, i, j, N, M, pad_cost
                )
            )
            >= __EPS
        ):
            return False
    optimal = u_sum + v_sum == cost_sum
    if not optimal:
        # Try to apply floating tolerance
        optimal = abs(cost_sum - u_sum - v_sum) < __EPS
    return optimal


def __reduced_cost(
    cost_matrix: list[list[float]],
    u_potentials: list[float],
    v_potentials: list[float],
    i: int,
    j: int,
    N: int,
    M: int,
    pad_cost: float,
) -> float:
    if i < N and j < M:
        # Reduced cost
        rc = cost_matrix[i][j] - u_potentials[i] - v_potentials[j]
    else:
        # If in padding dimension, use padding cost
        rc = pad_cost - u_potentials[i] - v_potentials[j]

    if isnan(rc):
        rc = 0

    # The reduced cost must always be >= 0
    return rc


def __search_augmented_path(
    row_i: int,
    cost_matrix: list[list[float]],
    N: int,
    M: int,
    PAD_M: int,
    pad_cost: float,
    inversionVector: list[int],
    u_potential: list[float],
    v_potential: list[float],
    S: set[int],
    T: set[int],
    path_found: bool = False,
) -> tuple[list[tuple[int]], bool]:

    # TODO Replace recursion with iterative approach as it may hit recursion limit

    # Initialize (sub)path
    path = []
    for j in range(PAD_M):
        # Find zeroed reduced cost in this row
        rc = __reduced_cost(
            cost_matrix, u_potential, v_potential, row_i, j, N, M, pad_cost
        )
        if abs(rc) > __EPS or j in T:  # Also skip visited columns
            continue

        # Initialize subpath
        path = [(row_i, j)]

        # Check if this column is free
        if inversionVector[j] == -1:
            # Return the augmented path
            return path, True

        # Check if the subpath brings to a cycle
        if inversionVector[j] in S:
            return [], False

        # Update alternated path
        T.add(j)  # Add j to the visited columns
        S.add(inversionVector[j])  # Visit the row that occupies this column

        # Extend alternated path until we find a free column or a wall
        new_path, path_found = __search_augmented_path(
            inversionVector[j],
            cost_matrix,
            N,
            M,
            PAD_M,
            pad_cost,
            inversionVector,
            u_potential,
            v_potential,
            S,
            T,
            path_found=path_found,
        )

        # Build the alternated path
        path.extend(new_path)
        if path_found:
            # An augmented path was found, stop the search
            break

    # Return the alternated path
    # (if path_found = True it's an augmented path)
    return path, path_found
