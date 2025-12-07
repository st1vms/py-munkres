# py-munkres

Python implementation of the Munkres (Hungarian) algorithm for solving assignment problems (minimum-cost bipartite matching), typical complexity is O(n^3).

## How to use

The `munkres` function accepts a cost matrix:

- The cost matrix is a 2-D list (N x M) where cost[i][j] is the numeric cost of assigning row i to column j.
- Rows typically represent "agents" (e.g., workers), columns represent "tasks" (e.g., jobs); the algorithm computes the assignment that minimizes (or maximizes) the total cost or profit.
- Entries can be integers or floats. When there are fewer jobs than workers, or vice versa, the resulting rectangular cost matrix is ​​filled with predefined costs, which in most cases can be 0, but can also be a specific value that best fits the problem.


The returned values are:

- `assignments` (***list[int]***): `assignments[i] = j` if the worker at row `i` is assigned to the job at column `j`, or `-1` if **unassigned** or assigned to a nonexistent job/column.
- `inversions` (***list[int]***): `inversions[j] = i` if the job at column `j` is assigned to the worker at row `i`, or `-1` if **free** or assigned to a nonexistent worker/row.
- `is_optimal` (***bool***): Indicates whether the algorithm's potentials certify optimality.

### Example usage starting from inputs

```py
from pymunkres import make_cost_matrix, munkres

workers = [1.0, 4.0, 10.0]
tasks = [2.0, 3.5]

cost_function = lambda a,b,i,j: abs(a-b)

cost_matrix = make_cost_matrix(workers, tasks, cost_function)

assignments, inversions, is_optimal = munkres(cost_matrix)
print("row i -> col j:", assignments) # [0, 1, -1]
print("col j -> row i:", inversions) # [0, 1]
print("optimal:", is_optimal) # True
```

### Example usage starting from cost matrix

```py
from pymunkres import munkres

cost_matrix = [
    [10, 5, 13, 15, 16],
    [3, 9, 18, 13, 6],
    [10, 7, 2, 2, 2],
    [7, 11, 9, 7, 12],
    [7, 9, 10, 4, 12],
]

assignments, inversions, is_optimal = munkres(cost_matrix)
total_cost = sum(
    cost_matrix[i][j] if j != -1 else 0 for i, j in enumerate(assignments)
)
print("row i -> col j:", assignments) # [1, 0, 4, 2, 3]
print("col j -> row i:", inversions) # [1, 0, 3, 4, 2]
print("Total cost:", total_cost) # 23
print("optimal:", is_optimal) # True
```

### Solving maximization problems

```py
from pymunkres import munkres

profit_matrix = [
    [10, 5, 13, 15, 16],
    [3, 9, 18, 13, 6],
    [10, 7, 2, 2, 2],
    [7, 11, 9, 7, 12],
    [7, 9, 10, 4, 12],
]

# Set maximization = True to set maximization problem
assignments, inversions, is_optimal = munkres(profit_matrix, maximization=True)
total_profit = sum(
    profit_matrix[i][j] if j != -1 else 0 for i, j in enumerate(assignments)
)
print("row i -> col j:", assignments)  # [3, 2, 0, 1, 4]
print("col j -> row i:", inversions)  # [2, 3, 1, 0, 4]
print("Total profit:", total_profit) # 66
print("optimal:", is_optimal) # True
```

### Disallowing specific assignments

To disallow specific assignments, you can set `disallowment_map`, a dictionary that maps worker row indices to sets of non-assignable task column indices.

For example, `disallowment_map = {0: {1,2,3}, 3: {1}}` would prevent worker 0 from being assigned to jobs 1, 2, 3, and worker 3 from being assigned to job 1.

Each of these assignments will always have an infinite cost, and if the optimal solution still includes any of these assignments, they will be set to unassigned in post-processing.

```py
from pymunkres import munkres

profit_matrix = [
    [5, 9, 0],
    [10, 0, 2],
    [8, 0, 4],
]

disallowment_map = {
    0: {2},  # Disallow Worker 0 -> Job 2
    1: {1},  # Disallow Worker 1 -> Job 1
    2: {1},  # Disallow Worker 2 -> Job 1
}

assignments, inversions, is_optimal = munkres(
    profit_matrix, disallowment_map=disallowment_map
)
total_profit = sum(
    profit_matrix[i][j] if j != -1 else 0 for i, j in enumerate(assignments)
)
print("row i -> col j:", assignments)  # [1, 2, 0]
print("col j -> row i:", inversions)  # [2, 0, 1]
print("Total profit:", total_profit)  # 19
print("optimal:", is_optimal)  # True
```

## Tests
Basic unittests are included in `tests/` folder.

To run the tests:
- From project root:
    - `python -m unittest discover`
- Or run single test:
    - `python -m unittest tests.test_optimal_min`

## Contributing

This repository is in the early stages of development and requires thorough testing, particularly for edge cases. Contributions are highly welcome and will help us move toward a stable release.


## 🪜 Roadmap

* [x] Out of the box support for rectangular matrices
* [x] Support for both minimization and maximization problems
* [x] Logic for disallowing specific assignments
* [ ] Iterative approach for `__search_augmented_path`, in order to avoid recursion depth limit with large cost matrices
* [ ] Numpy
* [ ] Pre-release revision and test freezing


## References

This project aims to provide a modern alternative to the now-abandoned [bmc/munkres](https://github.com/bmc/munkres) implementation. However, it may not yet cover all edge cases. For reference or production use, you may still need to consult the original `bmc/munkres` repository.