from pymunkres import munkres
import unittest

class Test_Optimal_MXSquare(unittest.TestCase):

    test_cost_mx = [
        [10, 5, 13, 15, 16],
        [3, 9, 18, 13, 6],
        [10, 7, 2, 2, 2],
        [7, 11, 9, 7, 12],
        [7, 9, 10, 4, 12],
    ]

    def test_optimal(self):
        res = munkres(self.test_cost_mx)
        print(f"M={res[0]}, I={res[1]}")
        self.assertTrue(res[-1])


if __name__ == "__main__":
    unittest.main()
