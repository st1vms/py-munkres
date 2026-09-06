"""Munkres Algorithm implementation (Hungarian Algorithm)"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Sequence,
    Callable,
    Any,
    Tuple,
    List,
    Dict,
    Set,
    Union,
    Optional,
    overload,
    Literal,
)
from math import isnan

if TYPE_CHECKING:
    import numpy as np
    from numpy import ndarray
else:
    try:
        import numpy as np
        from numpy import ndarray
        HAS_NUMPY = True
    except ImportError:
        HAS_NUMPY = False
        np = None
        ndarray = Any

__EPS = 1e-9  # Floating point tolerance


@overload
def make_cost_matrix(
    workers: Union[Sequence[Any], ndarray],
    jobs: Union[Sequence[Any], ndarray],
    cost_function: Callable[[Any, Any, int, int], Any],
    as_numpy: Literal[False] = False,
) -> List[List[float]]: ...


@overload
def make_cost_matrix(
    workers: Union[Sequence[Any], ndarray],
    jobs: Union[Sequence[Any], ndarray],
    cost_function: Callable[[Any, Any, int, int], Any],
    as_numpy: Literal[True],
) -> ndarray: ...


@overload
def make_cost_matrix(
    workers: Union[Sequence[Any], ndarray],
    jobs: Union[Sequence[Any], ndarray],
    cost_function: Callable[[Any, Any, int, int], Any],
    as_numpy: bool = False,
) -> Union[List[List[float]], ndarray]: ...


def make_cost_matrix(
    workers: Union[Sequence[Any], ndarray],
    jobs: Union[Sequence[Any], ndarray],
    cost_function: Callable[[Any, Any, int, int], Any],
    as_numpy: bool = False,
) -> Union[List[List[float]], Any]:
    """
    Utility function to create a cost matrix by calculating the cost of pairing every
    element from the workers sequence with every element from the jobs sequence.

    The resulting matrix M will have shape (len(workers), len(jobs)), where
    M[i][j] = cost_function(workers[i], jobs[j], i, j).

    #### Parameters
    - `workers` (***Sequence[Any] | np.ndarray***) Sequence of workers (defines the matrix rows).
    - `jobs` (***Sequence[Any] | np.ndarray***) Sequence of jobs (defines the matrix columns).
    - `cost_function` (***Callable[[Any, Any, int, int], float]***) The function
      used to calculate the cost for a specific worker-job pairing.
    - `as_numpy` (***bool***) If True and NumPy is available, return an ndarray.

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

    if as_numpy and np is not None:
        return np.array(cost_matrix, dtype=float)

    return cost_matrix


def munkres(
    cost_matrix: Union[List[List[float]], ndarray, Sequence[Any]],
    maximization: bool = False,
    pad_cost: float = 0,
    disallowment_map: Optional[Dict[int, Set[int]]] = None,
) -> Tuple[List[int], List[int], bool]:
    """
    Computes the minimum cost bipartite matching on a rectangular (N x M) cost matrix.

    #### Parameters

    ##### Cost matrix

    - The cost matrix is a 2-D list or NumPy ndarray (N x M) where cost[i][j] is the numeric cost of assigning row i to column j.
    - Rows typically represent "agents" (e.g., workers), columns represent "tasks" (e.g., jobs);
    the algorithm computes the assignment that minimizes (or maximizes) the total cost or profit.
    - Entries can be integers or floats. When there are fewer jobs than workers, or vice versa,
    the resulting rectangular cost matrix is filled with predefined costs, which in most cases can be 0, but can also be a specific value that best fits the problem.

    ##### Maximization flag

    Set the problem as a minimization or maximization problem (defaults to `False`, minimization problem)

    ##### Padding cost

    `pad_cost` is an optional floating point value (defaults to 0) that will be used as the default cost for padded assignments.
    You can usually leave this to default for most of the problems, but you can also choose a specific value that best fits your problem.

    ##### Disallowment Map

    To disallow specific assignments, you can set `disallowment_map`, a dictionary that maps worker row indices to sets of non-assignable task column indices.

    For example, `disallowment_map = {0: {1,2,3}, 3: {1}}` would prevent worker 0 from being assigned to jobs 1, 2, 3, and worker 3 from being assigned to job 1.

    Each of these assignments will always have an infinite cost, and if the optimal solution still includes any of these assignments, they will be set to unassigned in post-processing.

    #### Return values

    - `assignments` (***List[int]***): `assignments[i] = j` if the worker at row `i` is assigned to the job at column `j`, or `-1` if **unassigned** or assigned to a nonexistent job/column.
    - `inversions` (***List[int]***): `inversions[j] = i` if the job at column `j` is assigned to the worker at row `i`, or `-1` if **free** or assigned to a nonexistent worker/row.
    - `is_optimal` (***bool***): Indicates whether the algorithm's potentials certify optimality.
    """
    if disallowment_map is None:
        disallowment_map = {}

    # In order to solve the maximization problem by solving the minimization problem, all costs get negated
    SIGN = -1 if maximization else 1

    # Get the dimensions of the cost matrix
    if np is not None and isinstance(cost_matrix, ndarray):
        assert cost_matrix.ndim == 2, "Cost matrix must be 2-dimensional"
        N, M = cost_matrix.shape
    else:
        N = len(cost_matrix)
        assert N > 0, "Empty cost matrix"
        M = len(cost_matrix[0])

    assert N > 0, "Empty cost matrix"
    assert M > 0, "Cost matrix has no columns (jobs)"

    # Base 1->1 case
    if N == 1 and M == 1:
        if disallowment_map and 0 in disallowment_map and 0 in disallowment_map[0]:
            return [-1], [-1], False
        return [0], [0], True

    # Get padding dimensions
    PAD_N = max(N, M)
    PAD_M = PAD_N

    # Precompute row disallowment sets for fast lookups
    disallow_sets: Dict[int, Set[int]] = disallowment_map if disallowment_map else {}

    # Calculate potentials U (minimum for each row)
    u = []
    for i in range(PAD_N):
        if i >= N:
            # Padding zone
            u.append(pad_cost)
            continue

        disallowed_i = disallow_sets.get(i)
        row_cost = cost_matrix[i]
        row_min = min(
            (
                # Disallowment check
                float("inf")
                if disallowed_i and j in disallowed_i
                # Padding check
                else (SIGN * row_cost[j] if j < M else pad_cost)
            )
            for j in range(PAD_M)
        )
        if isnan(row_min):
            # NaN values are converted to 0
            row_min = 0
        u.append(row_min)

    # Calculate potentials V
    # (minimum for each column - u[i])
    v = []
    for j in range(PAD_M):
        col_min = float("inf")
        for i in range(PAD_N):
            disallowed_i = disallow_sets.get(i)
            cost = (
                # Disallowment check
                float("inf")
                if disallowed_i and j in disallowed_i
                # Padding check
                else (SIGN * cost_matrix[i][j] if i < N and j < M else pad_cost)
            ) - u[i]
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
                SIGN,
                disallow_sets,
            )
            if not path_found:
                # Calculate delta
                # (minimum reduced cost considering all visited rows,
                #  and all unvisited columns in the alternated path)
                unvisited_cols = [col for col in range(PAD_M) if col not in T]
                delta = min(
                    [
                        __reduced_cost(
                            cost_matrix,
                            u,
                            v,
                            row,
                            col,
                            N,
                            M,
                            pad_cost,
                            SIGN,
                            disallow_sets,
                        )
                        for row in S  # All visited rows
                        for col in unvisited_cols  # All unvisited columns
                    ]
                    or [0]
                )

                if abs(delta) < __EPS:
                    # In theory this should not happen, floating-point rounding or pathological matrices could trigger this.
                    # if it happens return assigments with path_found = False
                    break

                # Update potentials
                for row_i in S:
                    new_u = u[row_i] + delta
                    if isnan(new_u):
                        new_u = 0
                    u[row_i] = new_u

                for col_j in T:
                    new_v = v[col_j] - delta
                    if isnan(new_v):
                        new_v = 0
                    v[col_j] = new_v
                continue

            # Walk through augmented path and invert each arc to assign this row
            for arc in path:
                row_i, col_j = arc

                # Check if this arc represents an assignment
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
        [
            # Filter any disallowed assignment from the optimal solution
            -1 if j >= M or (disallow_sets and i in disallow_sets and j in disallow_sets[i]) else int(j)
            for i, j in enumerate(Z[:N])
        ],
        [
            # Filter any disallowed assignment from the optimal solution
            -1 if i >= N or (disallow_sets and i in disallow_sets and j in disallow_sets[i]) else int(i)
            for j, i in enumerate(inversions[:M])
        ],
        bool(__optimality_check(
            cost_matrix, Z, u, v, N, M, pad_cost, SIGN, disallow_sets
        )),
    )


def __optimality_check(
    cost_matrix: Any,
    assignments: List[int],
    u_potentials: List[float],
    v_potentials: List[float],
    N: int,
    M: int,
    pad_cost: float,
    sign: int,
    disallowment_map: Dict[int, Set[int]],
) -> bool:
    # For the solution to be optimal:
    # The sum of potentials must be equal the sum of the total cost of assignments
    u_sum = 0
    v_sum = 0
    cost_sum = 0
    for i, j in enumerate(assignments):
        if j == -1 or (disallowment_map and i in disallowment_map and j in disallowment_map[i]):
            continue

        u_sum += u_potentials[i]
        v_sum += v_potentials[j]

        # Use padding cost if this cell is a padded cell
        cost_sum += sign * cost_matrix[i][j] if i < N and j < M else pad_cost

        # Also each reduced cost generated by the assignment must be 0
        if (
            abs(
                __reduced_cost(
                    cost_matrix,
                    u_potentials,
                    v_potentials,
                    i,
                    j,
                    N,
                    M,
                    pad_cost,
                    sign,
                    disallowment_map,
                )
            )
            >= __EPS
        ):
            return False

    optimal = u_sum + v_sum == cost_sum
    # Try to apply floating tolerance
    return bool(optimal if optimal else abs(cost_sum - u_sum - v_sum) < __EPS)


def __reduced_cost(
    cost_matrix: Any,
    u_potentials: List[float],
    v_potentials: List[float],
    i: int,
    j: int,
    N: int,
    M: int,
    pad_cost: float,
    sign: int,
    disallowment_map: Dict[int, Set[int]],
) -> float:
    # Reduced cost
    rc = (
        (
            # Disallowment check
            float("inf")
            if disallowment_map and i in disallowment_map and j in disallowment_map[i]
            # Padding check
            else (sign * cost_matrix[i][j] if i < N and j < M else pad_cost)
        )
        - u_potentials[i]
        - v_potentials[j]
    )

    return 0.0 if isnan(rc) else rc


def __search_augmented_path(
    row_i: int,
    cost_matrix: Any,
    N: int,
    M: int,
    PAD_M: int,
    pad_cost: float,
    inversionVector: List[int],
    u_potential: List[float],
    v_potential: List[float],
    S: Set[int],
    T: Set[int],
    sign: int,
    disallowment_map: Dict[int, Set[int]],
    path_found: bool = False,
) -> Tuple[List[Tuple[int, int]], bool]:
    """
    Iteratively search for an augmented path using a stack
    """
    # stack element: [curr_row, next_j_to_check, chosen_col]
    stack = [[row_i, 0, None]]

    while stack:
        curr = stack[-1]
        r = curr[0]
        start_j = curr[1]

        found_next = False
        for j in range(start_j, PAD_M):
            if j in T:
                continue

            rc = __reduced_cost(
                cost_matrix,
                u_potential,
                v_potential,
                r,
                j,
                N,
                M,
                pad_cost,
                sign,
                disallowment_map,
            )

            if abs(rc) > __EPS:
                continue

            # Found valid edge (r, j) with zero reduced cost
            if inversionVector[j] == -1:
                # Augmenting path found!
                path = [(frame[0], frame[2]) for frame in stack[:-1]] + [(r, j)]
                return path, True

            if inversionVector[j] in S:
                return [], False

            T.add(j)
            next_row = inversionVector[j]
            S.add(next_row)

            curr[1] = j + 1
            curr[2] = j

            stack.append([next_row, 0, None])
            found_next = True
            break

        if not found_next:
            stack.pop()

    return [], False
