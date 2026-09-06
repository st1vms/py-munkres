import unittest
import sys
from pymunkres import munkres

class TestIterativeAugmentedPath(unittest.TestCase):
    def test_large_matrix_exceeds_recursion_limit(self):
        """
        Verify that iterative __search_augmented_path does not hit
        RecursionError on large matrices where path depth exceeds sys.getrecursionlimit().
        """
        # Create a 1050 x 1050 matrix designed to create a long alternating path.
        # Python's default recursion limit is 1000.
        size = 1050
        self.assertGreater(size, sys.getrecursionlimit())

        # An identity cost structure with slight perturbation:
        # Off-diagonal = 10, diagonal = 1.
        # When solving, this will create matches along 1050 items.
        cost_matrix = [[1.0 if i == j else 10.0 for j in range(size)] for i in range(size)]

        assignments, inversions, is_optimal = munkres(cost_matrix)

        self.assertEqual(len(assignments), size)
        self.assertEqual(len(inversions), size)
        self.assertTrue(is_optimal)
        self.assertEqual(assignments, list(range(size)))
        self.assertEqual(inversions, list(range(size)))

if __name__ == "__main__":
    unittest.main()
