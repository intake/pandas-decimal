import pytest
import pandas as pd
import pandas_decimal


@pytest.mark.parametrize(
    "func", [str, repr]
)
@pytest.mark.parametrize(
    "data", [
        [[1, 0], "1\n"],
        [[1, 1], "1.0\n"],
        [[[1, 1], 1], "1.0\n"],
        [[[1, 1], 3], "1.000\n"],
        [[[0.0001], 3], "0.000\n"],
        [[[0.1, 1], 3], "0.100\n"]
    ]
)
def test_series_repr(data, func):
    (values, places), expected = data
    s = pd.Series(values, dtype=pandas_decimal.DecimaldDtype(places))
    out = func(s)
    assert expected in out
