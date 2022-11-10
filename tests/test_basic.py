import pytest

def test_if_basic_import_works():
    # This slightly tests for syntax errors as well
    import pandas_decimal

@pytest.mark.parametrize("test_input,expected", [('', '[0]'), ('[0]', '[0]'), ('[1]', '[1]'),('[3]', '[3]')])
def test_instantiation_works_looking_at_type(test_input, expected):
    import pandas as pd
    import pandas_decimal
    x = pd.Series([1,2,3], dtype=f'decimal{test_input}')
    assert str(x.dtype) == f'decimal{expected}'

def test_bad_instantiation_detects():
    with pytest.raises(ValueError):
        import pandas as pd
        import pandas_decimal
        x = pd.Series([1,2,3], dtype=f'decimal[-1]')
