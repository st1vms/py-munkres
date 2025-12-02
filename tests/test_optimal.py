from pymunkres import munkres
import unittest


class Test_Optimal_MXSquare(unittest.TestCase):

    test_cases = [
        # Square 1
        ([[400, 150, 400], [400, 450, 600], [300, 225, 300]], [1, 0, 2]),
        # Square 2
        ([[10, 10, 8], [9, 8, 1], [9, 7, 4]], [0, 2, 1]),
        # Square 3
        (
            [
                [10, 5, 13, 15, 16],
                [3, 9, 18, 13, 6],
                [10, 7, 2, 2, 2],
                [7, 11, 9, 7, 12],
                [7, 9, 10, 4, 12],
            ],
            [1, 0, 4, 2, 3],
        ),
        # Rectangular 1
        (
            [[400, 150, 400, 1], [400, 450, 600, 2], [300, 225, 300, 3]],
            [1, 3, 2],
        ),
        # Rectangular 2
        ([[10, 10, 8, 11], [9, 8, 1, 1], [9, 7, 4, 10]], [1, 3, 2]),
        # Rectangular 3
        (
            [
                [10, float("inf"), float("inf")],
                [float("inf"), float("inf"), 1],
                [float("inf"), 7, float("inf")],
            ],
            [0, 2, 1],
        ),
        # Negative Square 1
        (
            [[-400, -150, -400], [-400, -450, -600], [-300, -225, -300]],
            [0, 2, 1],
        ),
        # Negative Square 2
        ([[-10, -10, -8], [-9, -8, -1], [-9, -7, -4]], [2, 1, 0]),
        # Negative Square 3
        (
            [
                [-10, -5, -13, -15, -16],
                [-3, -9, -18, -13, -6],
                [-10, -7, -2, -2, -2],
                [-7, -11, -9, -7, -12],
                [-7, -9, -10, -4, -12],
            ],
            [3, 2, 0, 1, 4],
        ),
        # Negative Rectangular 1
        (
            [[-400, -150, -400, -1], [-400, -450, -600, -2], [-300, -225, -300, -3]],
            [0, 2, 1],
        ),
        # Negative Square 2
        ([[-10, -10, -8, -11], [-9, -8, -1, -1], [-9, -7, -4, -10]], [1, 0, 3]),
        # Negative Rectangular 3
        (
            [
                [-10, float("-inf"), float("-inf")],
                [float("-inf"), float("-inf"), -1],
                [float("-inf"), -7, float("-inf")],
            ],
            [2, 1, 0],
        ),
    ]

    def _padding(self, cost_matrix):

        N = len(cost_matrix)
        M = 0

        for i in range(N):
            M = max(M, len(cost_matrix[i]))

        if M > N:
            cost_matrix += [[0] * M] * (M - N)
            return M - N
        elif M < N:
            cost_matrix = [
                (
                    cost_matrix[i]
                    if len(cost_matrix[i]) == N
                    else cost_matrix[i] + [0] * (N - M)
                )
                for i in range(N)
            ]
            return N - M
        return 0

    def test_optimal(self):
        for i, test_case in enumerate(self.test_cases):
            cost_matrix, exp_results = test_case

            # NOTE Remove this in case padding logic gets merged with munkres
            pad = self._padding(cost_matrix)

            res = munkres(cost_matrix)
            print(f"test_optimal#{i} M={res[0]}, I={res[1]}\n")

            self.assertSequenceEqual(res[0][:-pad] if pad > 0 else res[0], exp_results)


if __name__ == "__main__":
    unittest.main()
