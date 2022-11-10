import pytest

import numpy as np
import pandas as pd

@pytest.mark.parametrize("test_input,expected", [('', '[0]'), ('[0]', '[0]'), ('[1]', '[1]'),('[3]', '[3]')])
def test_instantiation_works_looking_at_type(test_input, expected):
    x = pd.Series([1000,2000,3000], dtype=f'decimal{test_input}')
    assert str(x.dtype) == f'decimal{expected}'


def test_bad_instantiation_detects():
    with pytest.raises(TypeError):
        x = pd.Series([1, 2, 3], dtype=f'decimal[-1]')


def test_type_of_individual_elements():
    x = pd.Series([1,2,3], dtype=f'decimal[0]')
    assert all([t.dtype.kind == "f" for t in x])  # these are np scalars

@pytest.mark.parametrize("test_input", ['[0]', '[1]', '[3]'])
def test_values_are_correct(test_input):
    original = [1,2,3]
    series = pd.Series(original, dtype=f'decimal{test_input}')
    assert all(np.isclose(x, y) for x, y in zip(original, series))
