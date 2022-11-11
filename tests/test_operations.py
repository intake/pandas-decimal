import numpy as np
import pandas as pd
import pytest

import pandas_decimal


### Vector and Vector on Ints ####
@pytest.fixture
def sample_int_list():
    return np.arange(5).tolist()


@pytest.fixture
def make_sample_decimal_series(sample_int_list):
    def _make_sample_decimal_vector(degree, ):
        return pd.Series(sample_int_list, dtype=f"decimal[{degree}]")

    return _make_sample_decimal_vector


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_adding_of_same_degree_works_by_item(degree, sample_int_list, make_sample_decimal_series):
    expected = [x + y for x, y in zip(sample_int_list, sample_int_list)]
    x = make_sample_decimal_series(degree)
    y = make_sample_decimal_series(degree)
    z = x + y
    # Test degree didn't change
    assert str(z.dtype) == f"decimal[{degree}]"
    # Test values are correct
    assert all([i == j for i, j in zip(expected, z)])


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_adding_of_same_degree_works_by_vector(degree, sample_int_list, make_sample_decimal_series):
    expected = [x + y for x, y in zip(sample_int_list, sample_int_list)]
    x = make_sample_decimal_series(degree)
    y = make_sample_decimal_series(degree)
    z = x + y
    # Test degree didn't change
    assert str(z.dtype) == f"decimal[{degree}]"
    # Test values are correct
    assert np.all(expected == z)


@pytest.mark.parametrize("delta_degree", [-1, 1, 2, 3])
def test_adding_of_different_degree_works_by_item(delta_degree, sample_int_list, make_sample_decimal_series):
    degree = 1
    other_degree = degree + delta_degree
    expected = [x + y for x, y in zip(sample_int_list, sample_int_list)]
    x = make_sample_decimal_series(degree)
    y = make_sample_decimal_series(other_degree)
    z = x + y
    # Test degree didn't change
    assert str(z.dtype) == f"decimal[{max(degree, other_degree)}]"
    # Test values are correct
    assert all([i == j for i, j in zip(expected, z)])


@pytest.mark.parametrize("delta_degree", [-1, 1, 2, 3])
def test_adding_of_different_degree_works_by_vector(delta_degree, sample_int_list, make_sample_decimal_series):
    degree = 1
    other_degree = degree + delta_degree
    expected = [x + y for x, y in zip(sample_int_list, sample_int_list)]
    x = make_sample_decimal_series(degree)
    y = make_sample_decimal_series(other_degree)
    z = x + y
    # Test degree didn't change
    assert str(z.dtype) == f"decimal[{max(degree, other_degree)}]"
    # Test values are correct
    assert np.all(expected == z)


def test_neg():
    s = pd.Series([0.1, 2, 3], dtype="decimal[1]")
    assert (-s).tolist() == [-_ for _ in s.tolist()]


def test_pow():
    s = pd.Series([1, 2, 3], dtype="decimal[1]")
    s2 = s**2
    assert (s2 == s * s).all()
    assert s2.dtype == "decimal[2]"
