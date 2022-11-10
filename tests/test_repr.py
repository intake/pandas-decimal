import pytest
import pandas as pd
import pandas_decimal


@pytest.mark.parametrize(
    "data", [
        [[1, 0], "1\n"]
    ]
)
def test_series_repr(data):
    (values, places), expected = data
    s = pd.Series(values, dtype=pandas_decimal.DecimaldDtype(places))
    out = repr(s)
    assert expected in out
