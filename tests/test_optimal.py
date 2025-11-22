from pymunkres import munkres
import unittest


def _difference(x: int | float, y: int | float) -> int | float:
    return abs(x - y)


def _linear_func_one(x: int | float, y: int | float) -> int | float:
    return abs(x - y) + (x + y) % 3


class Test_Optimal_MXSquare(unittest.TestCase):

    test_data = [([1, 3, 6, 8], [2, 4, 5, 7])]

    def test_optimal(self):
        for A, B in self.test_data:
            res = munkres(A, B, _difference)
            print(f"A={A}, B={B} -> M={res[0]}, I={res[1]}")
            self.assertTrue(res[-1])

            res = munkres(A, B, _linear_func_one)
            print(f"A={A}, B={B} -> M={res[0]}, I={res[1]}")
            self.assertTrue(res[-1])


class Test_Optimal_MXRectangle(unittest.TestCase):

    less_jobs_test_data = [
        ([1, 3, 6, 8], [2, 4]),
    ]

    less_workers_test_data = [
        ([1, 3], [2, 4, 5, 7]),
    ]

    def test_optimal_less_jobs(self):
        for A, B in self.less_jobs_test_data:
            res = munkres(A, B, _difference)
            print(f"A={A}, B={B} -> M={res[0]}, I={res[1]}")
            self.assertTrue(res[-1])

            res = munkres(A, B, _linear_func_one)
            print(f"A={A}, B={B} -> M={res[0]}, I={res[1]}")
            self.assertTrue(res[-1])

    def test_optimal_less_workers(self):

        for A, B in self.less_workers_test_data:
            res = munkres(A, B, _difference)
            print(f"A={A}, B={B} -> M={res[0]}, I={res[1]}")
            self.assertTrue(res[-1])

            res = munkres(A, B, _linear_func_one)
            print(f"A={A}, B={B} -> M={res[0]}, I={res[1]}")
            self.assertTrue(res[-1])


if __name__ == "__main__":
    unittest.main()
