"""Munkres Algorithm implementation (Hungarian Algorithm)"""

from typing import Callable


def munkres(
    a: list[int | float],
    b: list[int | float],
    cost_function: Callable[[int | float, int | float], int | float],
) -> tuple[list[int | float], bool]:

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

    # Negative slack used to pad negative costs
    negative_slack = 0

    # Potentials U and V
    u = []
    v = []

    # Cost matrix
    C = []
    for i in range(PAD_N):

        # Initialize potentials
        v.append(0)
        u.append(0)

        row = []
        for j in range(PAD_M):
            if i >= N or j >= M:
                # Initialize padding cells to one
                row.append(0)
                continue

            # Calculate the cost for this assignment
            cost = cost_function(a[i], b[j])

            # Update minimum row cost (potential U)
            u[i] = min(u[i] or cost, cost)

            # Update negative slack
            negative_slack = min(negative_slack, cost)

            row.append(cost)
        C.append(row)

    # negative_slack >= 0
    negative_slack = abs(negative_slack)

    Z = [-1 for _ in range(PAD_N)]  # Assignments
    inversions = [-1 for _ in range(PAD_M)]  # Inversion vector for these assignments

    # Iterate over non padded rows
    for i in Z[:N]:

        # Run augmented path search for this row
        path_found = False
        while not path_found:
            path, S, T, path_found = __search_augmented_path(i, C, inversions, u, v)
            if not path_found:
                # Update potentials
                delta = min(
                    __reduced_cost(C, u, v, row, col, negative_slack)
                    for row in range(S)  # All visited rows
                    for col in range(PAD_M)
                    if col not in T  # All unvisited columns
                )

                u = [u_i + delta for i, u_i in enumerate(u) if i in S]
                v = [v_i + delta for i, v_i in enumerate(v) if i in T]
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

    # Return assignments and a flag indicating if the solution is optimal
    return Z, __optimal_check(C, Z, u, v)


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
    negative_slack: int | float,
) -> int | float:
    return cost_matrix[i][j] - potentials_u[i] - potentials_v[j] + (negative_slack * 3)


def __search_augmented_path(
    row: int,
    cost_matrix: int,
    inversionVector: list[int | float],
    u_potential: int | float,
    v_potential: int | float,
    negative_slack: int | float,
    _visited_rows: set[int | float] = None,
    _visited_columns: set[int | float] = None,
    path_found: bool = False,
) -> tuple[list[tuple[int]], set, set, bool]:

    if _visited_rows is None:
        _visited_rows = set((row,))

    if _visited_columns is None:
        _visited_columns = set()

    path = []
    for j in range(len(cost_matrix[row])):
        # Find zeroed reduced cost in this row
        rc = __reduced_cost(
            cost_matrix, u_potential, v_potential, row, j, negative_slack
        )
        if rc != 0 or j in _visited_columns:
            continue

        # Initialize a new path
        path = [(row, j)]

        # Check if this column is free
        if inversionVector[j] == -1:
            return path, _visited_rows, _visited_columns, True

        # Check if the path search is stuck
        if inversionVector[j] in _visited_rows:
            return path, _visited_rows, _visited_columns, False

        _visited_columns.add(j)  # Column visited
        _visited_rows.add(inversionVector[j])  # Visit the row that occupies this column

        # Walk through augmented path to find a free column
        new_path, _visited_rows, _visited_columns, path_found = __search_augmented_path(
            inversionVector[j],
            cost_matrix,
            inversionVector,
            u_potential,
            v_potential,
            negative_slack,
            _visited_rows=_visited_rows,
            _visited_columns=_visited_columns,
            path_found=path_found,
        )
        path.extend(new_path)
        if path_found:
            # Break search and return result
            break

    # Return path and visited rows, columns
    return path, _visited_rows, _visited_columns, path_found


if __name__ == "__main__":

    def difference(a: int | float, b: int | float) -> int | float:
        return a - b

    def __test_main():
        A = [6, 7, 5]
        B = [8, 1, 4]
        print(munkres(A, B, difference))

    __test_main()
