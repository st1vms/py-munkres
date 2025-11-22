# py-munkres

Python implementation of the Munkres (Hungarian) algorithm for solving assignment problems (minimum-cost bipartite matching), typical complexity is O(n^3).

## Example usage

```py
from pymunkres import munkres

cost = [
    [10, 5, 13, 15, 16],
    [3, 9, 18, 13, 6],
    [10, 7, 2, 2, 2],
    [7, 11, 9, 7, 12],
    [7, 9, 10, 4, 12],
]

assignments, inversions, is_optimal = munkres(cost)
print("row -> col:", assignments)
print("col -> row:", inversions)
print("optimal:", is_optimal)
```

Cost matrix
- The cost matrix is a 2-D list (N x M) where cost[i][j] is the numeric cost of assigning row i to column j.
- Rows typically represent "agents" (e.g., workers), columns represent "tasks"; the algorithm finds assignments that minimize the total cost.
- Entries can be integers or floats. The matrix must be non-empty and square (equal number of rows and columns); rectangular and irregular matrices must be filled with zero costs or a default cost appropriate to the problem.

Return values:
- `assignments` (***list[int]***): `assignments[i] = j` if row `i` is assigned to column `j`, or `-1` if **unassigned**.
- `inversions` (***list[int]***): `inversions[j] = i` if column `j` is assigned to row `i`, or `-1` if **free**.
- `is_optimal` (***bool***): indicates whether the algorithm's potentials certify optimality.

## Tests
Basic unittests are included in `tests/` folder.

To run the tests:
- From project root:
    - `python -m unittest discover`
- Or run the single test:
    - `python -m unittest tests.test_optimal`