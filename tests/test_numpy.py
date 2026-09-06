import unittest
import numpy as np
from pymunkres import munkres, make_cost_matrix

class TestNumpySupport(unittest.TestCase):
    def test_square_numpy_array(self):
        cost_matrix = np.array([
            [10, 5, 13, 15, 16],
            [3, 9, 18, 13, 6],
            [10, 7, 2, 2, 2],
            [7, 11, 9, 7, 12],
            [7, 9, 10, 4, 12],
        ])
        assignments, inversions, is_optimal = munkres(cost_matrix)
        self.assertIsInstance(assignments, list)
        self.assertIsInstance(inversions, list)
        self.assertIsInstance(is_optimal, bool)
        self.assertEqual(assignments, [1, 0, 4, 2, 3])
        self.assertEqual(inversions, [1, 0, 3, 4, 2])
        self.assertTrue(is_optimal)

    def test_rectangular_numpy_arrays(self):
        # More rows than columns (5x4)
        cost_matrix_5x4 = np.array([
            [34.01, 26.02, 17.03, 12.04],
            [43.05, 43.06, 36.07, 10.08],
            [97.09, 47.1, 66.11, 34.12],
            [52.13, 42.14, 19.15, 36.16],
            [15.17, 93.18, 55.19, 80.2],
        ])
        assignments, inversions, is_optimal = munkres(cost_matrix_5x4)
        self.assertEqual(assignments, [1, 3, -1, 2, 0])
        self.assertTrue(is_optimal)

        # More columns than rows (3x5)
        cost_matrix_3x5 = np.array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
        ])
        assignments, inversions, is_optimal = munkres(cost_matrix_3x5)
        self.assertEqual(len(assignments), 3)
        self.assertEqual(len(inversions), 5)
        self.assertTrue(is_optimal)

    def test_numpy_dtypes(self):
        for dtype in [np.int32, np.int64, np.float32, np.float64]:
            cost = np.array([[4, 2], [1, 3]], dtype=dtype)
            assignments, inversions, is_optimal = munkres(cost)
            self.assertEqual(assignments, [1, 0])
            self.assertTrue(is_optimal)

    def test_numpy_maximization(self):
        profit_matrix = np.array([
            [10, 5, 13, 15, 16],
            [3, 9, 18, 13, 6],
            [10, 7, 2, 2, 2],
            [7, 11, 9, 7, 12],
            [7, 9, 10, 4, 12],
        ])
        assignments, inversions, is_optimal = munkres(profit_matrix, maximization=True)
        self.assertEqual(assignments, [3, 2, 0, 1, 4])
        self.assertTrue(is_optimal)

    def test_numpy_disallowment(self):
        profit_matrix = np.array([
            [5, 9, 0],
            [10, 0, 2],
            [8, 0, 4],
        ])
        disallowment_map = {
            0: {2},
            1: {1},
            2: {1},
        }
        assignments, inversions, is_optimal = munkres(
            profit_matrix, disallowment_map=disallowment_map
        )
        self.assertEqual(assignments, [1, 2, 0])
        self.assertTrue(is_optimal)

    def test_numpy_nan_and_inf(self):
        matrix = np.array([
            [10.0, np.nan, 5.0],
            [np.inf, 2.0, 8.0],
            [4.0, 6.0, 1.0],
        ])
        assignments, inversions, is_optimal = munkres(matrix)
        self.assertEqual(len(assignments), 3)

    def test_make_cost_matrix_numpy(self):
        workers = np.array([1.0, 4.0, 10.0])
        tasks = np.array([2.0, 3.5])
        cost_func = lambda a, b, i, j: float(abs(a - b))

        mat_list = make_cost_matrix(workers, tasks, cost_func, as_numpy=False)
        self.assertIsInstance(mat_list, list)

        mat_np = make_cost_matrix(workers, tasks, cost_func, as_numpy=True)
        self.assertIsInstance(mat_np, np.ndarray)
        assert isinstance(mat_np, np.ndarray)
        self.assertEqual(mat_np.shape, (3, 2))

if __name__ == "__main__":
    unittest.main()
