from pymunkres import munkres
import unittest


def _difference(x: int | float, y: int | float) -> int | float:
    return abs(x - y)


def _linear_func_one(x: int | float, y: int | float) -> int | float:
    return abs(x - y) + (x + y) % 3


class Test_Optimal_MXSquare(unittest.TestCase):

    A = [1, 3, 6, 8]
    B = [2, 4, 5, 7]

    def test_optimal(self):

        res = munkres(self.A, self.B, _difference)
        print(f"A={self.A}, B={self.B} -> M={res[0]}, I={res[1]}")
        self.assertTrue(res[-1])

        res = munkres(self.A, self.B, _linear_func_one)
        print(f"A={self.A}, B={self.B} -> M={res[0]}, I={res[1]}")
        self.assertTrue(res[-1])


class Test_Optimal_MXRectangle(unittest.TestCase):

    A1 = [1, 3, 6, 8]
    B1 = [2, 4]

    A2 = [1, 3]
    B2 = [2, 4, 5, 7]

    def test_optimal_less_jobs(self):

        res = munkres(self.A1, self.B1, _difference)
        print(f"A={self.A1}, B={self.B1} -> M={res[0]}, I={res[1]}")
        self.assertTrue(res[-1])

        res = munkres(self.A1, self.B1, _linear_func_one)
        print(f"A={self.A1}, B={self.B1} -> M={res[0]}, I={res[1]}")
        self.assertTrue(res[-1])

    def test_optimal_less_workers(self):

        res = munkres(self.A2, self.B2, _difference)
        print(f"A={self.A2}, B={self.B2} -> M={res[0]}, I={res[1]}")
        self.assertTrue(res[-1])

        res = munkres(self.A2, self.B2, _linear_func_one)
        print(f"A={self.A2}, B={self.B2} -> M={res[0]}, I={res[1]}")
        self.assertTrue(res[-1])


if __name__ == "__main__":
    unittest.main()
