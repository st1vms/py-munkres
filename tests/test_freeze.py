import unittest
import numpy as np
from pymunkres import munkres, make_cost_matrix

class TestPreReleaseFreeze(unittest.TestCase):
    """
    Test freezing suite for pre-release revision.
    Validates edge cases, input validation, and guarantees deterministic outputs.
    """

    def test_empty_matrix_validation(self):
        with self.assertRaises(AssertionError):
            munkres([])
        with self.assertRaises(AssertionError):
            munkres([[]])
        with self.assertRaises(AssertionError):
            munkres(np.empty((0, 0)))

    def test_dimension_validation(self):
        with self.assertRaises(AssertionError):
            munkres(np.array([1, 2, 3]))  # 1D array

    def test_1x1_matrices(self):
        # Basic 1x1
        assignments, inversions, is_optimal = munkres([[42]])
        self.assertEqual(assignments, [0])
        self.assertEqual(inversions, [0])
        self.assertTrue(is_optimal)

        # 1x1 disallowed
        assignments, inversions, is_optimal = munkres([[42]], disallowment_map={0: {0}})
        self.assertEqual(assignments, [-1])
        self.assertEqual(inversions, [-1])
        self.assertFalse(is_optimal)

    def test_single_row_rectangular(self):
        cost_matrix = [[10, 5, 20]]
        assignments, inversions, is_optimal = munkres(cost_matrix)
        self.assertEqual(assignments, [1])
        self.assertEqual(inversions[1], 0)
        self.assertTrue(is_optimal)

    def test_single_column_rectangular(self):
        cost_matrix = [[10], [5], [20]]
        assignments, inversions, is_optimal = munkres(cost_matrix)
        self.assertEqual(assignments[1], 0)
        self.assertEqual(inversions, [1])
        self.assertTrue(is_optimal)

    def test_all_zeros_matrix(self):
        cost_matrix = [[0, 0], [0, 0]]
        assignments, inversions, is_optimal = munkres(cost_matrix)
        self.assertTrue(is_optimal)
        self.assertEqual(len(assignments), 2)
        self.assertEqual(set(assignments), {0, 1})

    def test_frozen_benchmark_determinism(self):
        # Frozen regression case to guarantee deterministic assignments across releases
        matrix = [
            [12, 9, 27, 10, 23],
            [7, 13, 13, 30, 19],
            [25, 18, 26, 11, 26],
            [9, 28, 26, 23, 13],
            [16, 16, 24, 6, 9],
        ]
        # Run multiple times to verify deterministic behavior
        for _ in range(5):
            assignments, inversions, is_optimal = munkres(matrix)
            self.assertEqual(assignments, [1, 2, 3, 0, 4])
            self.assertEqual(inversions, [3, 0, 1, 2, 4])
            self.assertTrue(is_optimal)

if __name__ == "__main__":
    unittest.main()
