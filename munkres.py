"""Munkres Algorithm implementation (Hungarian Algorithm)"""

from typing import Callable, List


def __reduced_cost(
    cost_matrix: List[List[int | float]],
    potentials_u: List[int | float],
    potentials_v: List[int | float],
    i: int,
    j: int,
    negative_slack: int | float
) -> int | float:
    """Returns the reduced cost given the potentials"""
    return cost_matrix[i][j] - potentials_u[i] - potentials_v[j] + (negative_slack * 3)


def __optimal_check(cost_matrix, assignments, u_potentials, v_potentials) -> bool:
    """Performs optimality check given the cost matrix, assignments, and potentials"""
    return sum(u_potentials) + sum(v_potentials) == sum(
        [cost_matrix[i][j] for i, j in enumerate(assignments)]
    )


def munkres(
    a: List[int | float],
    b: List[int | float],
    cost_function: Callable[[int | float, int | float], int | float],
) -> List[int | float]:
    """."""

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
                # This cell involves a padded cell
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

    # First assignment by calculating reduced cost matrix
    Z = []
    inversions = [-1 for _ in range(PAD_M)]
    first_unassigned_pos = -1
    for i in range(PAD_N):

        # Avoid creating extra assignments
        if i < N:
            Z.append(-1)  # Unassigned by default

        for j in range(PAD_M):
            rc = __reduced_cost(C, u, v, i, j, negative_slack)
            if (
                i < N  # Don't assign padding units
                and Z[i] == -1  # Only assign one time per row
                and rc == 0  # Must be a zero for assignment
                and inversions[j] != -1  # Column must be free
            ):
                inversions[j] = i  # Occupy column
                Z[i] = j if j < M else -1  # Prevent assignment to padding units
                break
        
        if Z[i] == -1 and first_unassigned_pos == -1:
            first_unassigned_pos = i

    # Check if solution is already optimal
    if __optimal_check(C, Z, u, v):
        return Z

    S = set((first_unassigned_pos,))
    T = set()



    return None


A = [6, 7, 5]
B = [8, 1, 4]

print(munkres(A, B, lambda a, b: a - b))
