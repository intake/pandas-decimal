import pytest

import pandas as pd
import pandas_decimal

@pytest.mark.parametrize("test_input,expected", [('', '[0]'), ('[0]', '[0]'), ('[1]', '[1]'),('[3]', '[3]')])
def test_instantiation_works_looking_at_type(test_input, expected):
    x = pd.Series([1,2,3], dtype=f'decimal{test_input}')
    assert str(x.dtype) == f'decimal{expected}'

def test_bad_instantiation_detects():
    with pytest.raises(ValueError):
        x = pd.Series([1,2,3], dtype=f'decimal[-1]')

def test_type_of_individual_elements():
    x = pd.Series([1,2,3], dtype=f'decimal[0]')
    assert all([isinstance(t, int) for t in x])
