"""Munkres Algorithm implementation (Hungarian Algorithm)"""

from typing import Callable


def munkres(
    a: list[int | float],
    b: list[int | float],
    cost_function: Callable[[int | float, int | float], int | float],
) -> tuple[list[int | float], list[int | float], bool]:

    # Get the input sizes
    N = len(a)
    M = len(b)

    # Check if any input is empty
    assert N > 0 and M > 0

    # Base 1->1 case
    if N == M == 1:
        return [0]

    # Use padded input sizes
    PAD_N = N
    PAD_M = M

    # Padding check
    if N > M:
        PAD_M = N
    elif M > N:
        PAD_N = M

    # Potentials U and V
    u = [float("inf") for _ in range(N)]
    v = [float("inf") for _ in range(M)]

    # Cost matrix
    C = []
    for i in range(PAD_N):
        row = []
        for j in range(PAD_M):
            if i >= N or j >= M:
                # Initialize padding cells to one
                row.append(0)
                continue
            # Calculate the cost for this assignment
            cost = cost_function(a[i], b[j])
            row.append(cost)
        C.append(row)

    # Calculate potentials U (minimum for each row)
    u = [min(C[i]) for i in range(N)]

    # Calculate potentials V (minimum for each column - u[i])
    v = []
    for j in range(M):
        v.append(float("inf"))
        for i in range(N):
            v[j] = min(v[j], C[i][j] - u[i])

    Z = [-1 for _ in range(PAD_N)]  # Assignments
    inversions = [-1 for _ in range(PAD_M)]  # Inversion vector for these assignments

    # Iterate over non padded rows
    for i in range(N):

        # Run augmented path search for this row
        path_found = False
        while not path_found:
            path, S, T, path_found = __search_augmented_path(i, C, inversions, u, v)
            if not path_found:
                # Update potentials
                delta = min(
                    __reduced_cost(C, u, v, row, col)
                    for row in S  # All visited rows
                    for col in range(PAD_M)
                    if col not in T  # All unvisited columns
                )

                u = [u_i + delta if i in S else u_i for i, u_i in enumerate(u)]
                v = [v_i + delta if i in T else v_i for i, v_i in enumerate(v)]
                continue

            # Invert path to assign this row
            for node in path:
                i, j = node

                # Check if this node represent an assignment
                if Z[i] == j:
                    # Invert the assignment
                    Z[i] = -1
                    inversions[j] = -1
                else:
                    # Perform assignment
                    Z[i] = j
                    inversions[j] = i

    # Return assignments, inversions, and a flag indicating if the solution is optimal
    return Z[:N], inversions[:M], __optimal_check(C, Z, u, v)


def __optimal_check(cost_matrix, assignments, u_potentials, v_potentials) -> bool:
    # For the solution to be optimal:
    # The sum of potentials must be equal the sum of the total cost of assignments
    return sum(u_potentials) + sum(v_potentials) == sum(
        [cost_matrix[i][j] for i, j in enumerate(assignments)]
    )


def __reduced_cost(
    cost_matrix: list[list[int | float]],
    potentials_u: list[int | float],
    potentials_v: list[int | float],
    i: int,
    j: int,
) -> int | float:
    return cost_matrix[i][j] - potentials_u[i] - potentials_v[j]


def __search_augmented_path(
    row_i: int,
    cost_matrix: int,
    inversionVector: list[int | float],
    u_potential: list[int | float],
    v_potential: list[int | float],
    S: set[int | float] = None,
    T: set[int | float] = None,
    path_found: bool = False,
) -> tuple[list[tuple[int]], set, set, bool]:

    if S is None:
        # Initialize visited rows
        S = set((row_i,))

    if T is None:
        # Initialize visited columns
        T = set()

    path = []
    for j in range(len(cost_matrix[row_i])):
        # Find zeroed reduced cost in this row
        rc = __reduced_cost(cost_matrix, u_potential, v_potential, row_i, j)
        if rc != 0 or j in T:
            continue

        # Save edge
        path = [(row_i, j)]

        # Check if this column is free
        if inversionVector[j] == -1:
            return path, S, T, True

        # Check if the path search is stuck
        if inversionVector[j] in S:
            return [], S, T, False

        T.add(j)  # Column visited
        S.add(inversionVector[j])  # Visit the row that occupies this column

        # Walk through augmented path to find a free column
        new_path, S, T, path_found = __search_augmented_path(
            inversionVector[j],
            cost_matrix,
            inversionVector,
            u_potential,
            v_potential,
            S=S,
            T=T,
            path_found=path_found,
        )
        path.extend(new_path)
        if path_found:
            # Break search and return result
            break

    # Return path and visited rows, columns
    return path, S, T, path_found

