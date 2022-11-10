import pytest

import numpy as np
import pandas as pd

@pytest.mark.parametrize("degree", [0,1,2,3])
def test_adding_of_same_degree_works(degree):
    dtype_str = f'decimal[{degree}]'
    original = [0,1,2,3]
    expected = [x+y for x,y in zip(original, original)]
    x = pd.Series(original, dtype=dtype_str)
    y = pd.Series(original, dtype=dtype_str)
    z = x + y
    # Test degree didn't change
    assert str(x.dtype) == dtype_str
    # Test values are correct
    assert all([np.isclose(x, y) for x, y in zip(expected, z)])

@pytest.mark.parametrize("degree", [0,1,2,3])
def test_adding_of_same_degree_works2(degree):
    dtype_str = f'decimal[{degree}]'
    original = [0,1,2,3]
    expected = [x+y for x,y in zip(original, original)]
    x = pd.Series(original, dtype=dtype_str)
    y = pd.Series(original, dtype=dtype_str)
    z = x + y
    # Test degree didn't change
    assert str(x.dtype) == dtype_str
    # Test values are correct
    assert np.all(expected == z)