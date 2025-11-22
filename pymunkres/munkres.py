"""Munkres Algorithm implementation (Hungarian Algorithm)"""

__EPS = 1e-9  # Floating point tolerance


def munkres(
    cost_matrix: list[list[int | float]], N: int, M: int
) -> tuple[list[int | float], list[int | float], bool]:
    """
    #### Cost matrix

    The `cost_matrix` is a 2-D list (`N` x `M`) where cost_matrix[i][j] is the numeric cost of assigning row i to column j.
    Rows typically represent "agents" (e.g., workers), columns represent "tasks"; the algorithm finds assignments that minimize the total cost.
    Entries can be integers or floats. The matrix must be non-empty and square (equal number of rows and columns); rectangular and irregular matrices must be filled with zero costs or a default cost appropriate to the problem.

    #### Return values

    - `assignments` (***list[int]***): `assignments[i] = j` if row `i` is assigned to column `j`, or `-1` if **unassigned**.
    - `inversions` (***list[int]***): `inversions[j] = i` if column `j` is assigned to row `i`, or `-1` if **free**.
    - `is_optimal` (***bool***): indicates whether the algorithm's potentials certify optimality.
    """

    # Check if number of rows matches N
    assert len(cost_matrix) == N

    # Check if any input is empty
    assert N > 0 and M > 0

    # Base 1->1 case
    if N == M == 1:
        return [0]

    # Calculate potentials U (minimum for each row)
    u = []
    for i in range(N):
        # Check if number of rows matches M
        assert len(cost_matrix[i]) == M
        u.append(min(cost_matrix[i]))

    # Calculate potentials V (minimum for each column - u[i])
    v = []
    for j in range(M):
        v.append(float("inf"))
        for i in range(N):
            v[j] = min(v[j], cost_matrix[i][j] - u[i])

    Z = [-1 for _ in range(N)]  # Assignments
    inversions = [-1 for _ in range(M)]  # Inversion vector for these assignments

    # Iterate over unassigned rows
    for i in range(N):

        # Run augmented path search for this row
        path_found = False
        while not path_found:
            S = set((i,))
            T = set()
            path, path_found = __search_augmented_path(
                i, cost_matrix, M, inversions, u, v, S, T
            )
            if not path_found:
                # Update potentials
                delta = min(
                    [
                        __reduced_cost(cost_matrix, u, v, row, col)
                        for row in S  # All visited rows
                        for col in range(M)
                        if col not in T  # All unvisited columns
                    ]
                    or (0,)
                )

                if abs(delta) < __EPS:
                    # Loop check
                    break

                u = [u_i + delta if i in S else u_i for i, u_i in enumerate(u)]
                v = [v_i - delta if i in T else v_i for i, v_i in enumerate(v)]
                continue

            # Invert path to assign this row
            for edge in path:
                i, j = edge

                # Check if this edge represent an assignment
                if Z[i] == j:
                    # Invert the assignment
                    Z[i] = -1
                    inversions[j] = -1
                else:
                    # Perform assignment
                    Z[i] = j
                    inversions[j] = i

    # Return assignments, inversions, and a flag indicating if the solution is optimal
    return Z, inversions, __optimal_check(cost_matrix, Z, u, v)


def __optimal_check(cost_matrix, assignments, u_potentials, v_potentials) -> bool:
    # For the solution to be optimal:
    # The sum of potentials must be equal the sum of the total cost of assignments
    u_sum = 0
    v_sum = 0
    cost_sum = 0
    for i, j in enumerate(assignments):
        if j == -1:
            continue
        u_sum += u_potentials[i]
        v_sum += v_potentials[i]
        cost_sum += cost_matrix[i][j]
    return u_sum + v_sum == cost_sum


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
    M: int,
    inversionVector: list[int | float],
    u_potential: list[int | float],
    v_potential: list[int | float],
    S: set[int | float],
    T: set[int | float],
    path_found: bool = False,
) -> tuple[list[tuple[int]], bool]:

    path = []
    for j in range(len(cost_matrix[row_i])):
        # Find zeroed reduced cost in this row
        rc = __reduced_cost(cost_matrix, u_potential, v_potential, row_i, j)
        if abs(rc) > __EPS or j in T:
            continue

        # Save edge
        path = [(row_i, j)]

        # Check if this column is free
        if inversionVector[j] == -1:
            return path, True

        # Check if the path search is stuck
        if inversionVector[j] in S:
            return [], False

        T.add(j)  # Column visited
        S.add(inversionVector[j])  # Visit the row that occupies this column

        # Walk through augmented path to find a free column
        new_path, path_found = __search_augmented_path(
            inversionVector[j],
            cost_matrix,
            M,
            inversionVector,
            u_potential,
            v_potential,
            S,
            T,
            path_found=path_found,
        )
        path.extend(new_path)
        if path_found:
            # Break search and return result
            break

    # Return path and visited rows, columns
    return path, path_found
