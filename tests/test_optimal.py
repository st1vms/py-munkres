from pymunkres.pymunkres import munkres

A = [1, 3, 6, 8]
B = [2, 4, 5, 7]


def _difference(x: int | float, y: int | float) -> int | float:
    return abs(x - y)


def _linear_func_one(x: int | float, y: int | float) -> int | float:
    return abs(x - y) + (x + y) % 3


def test_optimal():

    res = munkres(A, B, _difference)
    assert res[-1]

    print(f"{A=}, {B}, {res=}")

    res = munkres(A, B, _linear_func_one)
    assert res[-1]

    print(f"{A=}, {B}, {res=}")


if __name__ == "__main__":
    test_optimal()
