import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize(
    "test_input,expected", [("", "[0]"), ("[0]", "[0]"), ("[1]", "[1]"), ("[3]", "[3]")]
)
def test_instantiation_works_looking_at_type(test_input, expected):
    x = pd.Series([1000, 2000, 3000], dtype=f"decimal{test_input}")
    assert str(x.dtype) == f"decimal{expected}"


def test_bad_instantiation_detects():
    with pytest.raises(TypeError):
        pd.Series([1, 2, 3], dtype="decimal[-1]")


def test_type_of_individual_elements():
    x = pd.Series([1, 2, 3], dtype="decimal[0]")
    assert all([t.dtype.kind == "f" for t in x])  # these are np scalars


@pytest.mark.parametrize("test_degree", [0, 1, 3])
def test_instantiation_int_values_are_correct_and_not_morphed(test_degree):
    original = [1, 2, 3]
    series = pd.Series(original, dtype=f"decimal[{test_degree}]")
    assert all(np.isclose(x, y) for x, y in zip(original, series))


@pytest.mark.parametrize("test_degree", [0, 1, 2, 3])
def test_instantiation_int_values_are_correct_and_internally_correct(test_degree):
    original = [1, 2, 3]
    factor = int(10**test_degree)
    series = pd.Series(original, dtype=f"decimal[{test_degree}]")
    assert all(x * factor == y for x, y in zip(original, series.values._data))

@pytest.mark.parametrize("test_degree", [1, 2, 3])
def test_instantiation_float_values_are_correct_and_not_morphed(test_degree):
    original = [.1, .2, .3]
    series = pd.Series(original, dtype=f"decimal[{test_degree}]")
    assert all(np.isclose(x, y) for x, y in zip(original, series))


@pytest.mark.parametrize("test_degree", [1, 2, 3])
def test_instantiation_float_values_are_correct_and_internally_correct(test_degree):
    original = [.1, .2, .3]
    factor = int(10**test_degree)
    series = pd.Series(original, dtype=f"decimal[{test_degree}]")
    assert all(x * factor == y for x, y in zip(original, series.values._data))

# TODO: Make tests that test what to do when decimals are truncated and their correct behavior