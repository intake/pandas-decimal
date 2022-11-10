from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pandas.core.arrays.base import (
    ExtensionArray,
    ExtensionScalarOpsMixin,
    set_function_name,
)
from pandas.core.dtypes.base import (
    register_extension_dtype,
)
from pandas.core.dtypes.generic import ABCDataFrame, ABCIndex, ABCSeries

from pandas_decimal.dtype import DecimaldDtype


class DecimalExtensionArray(ExtensionArray, ExtensionScalarOpsMixin):
    _dtype: DecimaldDtype  # .decimal_places gives precision as exponent of 10
    _data: np.Array  # int64

    def __init__(self, data: Any, decimal_places=0, dtype=None) -> None:
        """

        Parameters
        ----------
        data: array-like
            Starting values
        decimal_places: int
            Number of decimal places of accuracy.  Ignored if ``dtype`` given
        dtype: DecimaldDtype instance
            If given, use this exact dtype.
        """
        if dtype:
            if dtype.kind != ".":
                raise ValueError("dtype of DecimalExtensionArray must be "
                                 "a DecimalExtensionArray")
            decimal_places = dtype.decimal_places
            self._dtype = dtype
        else:
            self._dtype = DecimaldDtype(decimal_places)
        if not hasattr(data, "dtype"):
            data = np.array(data)
        data = np.atleast_1d(data)
        self._data = np.round(data * 10**decimal_places).astype("int64")

    @classmethod
    def from_internal(cls, data, dtype):
        """Construct extension array where data is already in correct int form"""
        arr = cls([], dtype=dtype)
        arr._data = data
        return arr

    @classmethod
    def _from_sequence(cls, scalars, *, decimal_places=0, dtype=None, copy=False):
        return cls(np.array(scalars), decimal_places=decimal_places, dtype=dtype)

    @classmethod
    def _empty(cls, shape, dtype):
        if isinstance(shape, tuple) and len(shape) != 1:
            raise ValueError
        return cls(np.empty(shape), dtype=dtype)

    @classmethod
    def _from_factorized(cls, values, original):
        return cls.from_internal(values, original.dtype)

    def __getitem__(self, item):
        return self._data[item] / 10 ** self._dtype.decimal_places

    def __setitem__(self, key, value):
        self._data[key] = np.round(value * 10**self.dtype.decimal_places).astype("int64")

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        for i in range(len(self)):
            yield self._data[i] / 10 ** self._dtype.decimal_places

    def _formatter(self, boxed: bool = False) -> Callable[[Any], str | None]:
        st = "{" + f":.{self._dtype.decimal_places}f" + "}"
        return lambda x: st.format(x)

    @classmethod
    def _create_comparison_method(cls, op):
        def _binop(self, other):
            if not hasattr(other, "dtype"):
                other = np.asanyarray(other)
            if other.dtype.kind in ["i", "f"]:
                other = cls(other * 10 ** self._dtype.decimal_places, dtype=self.dtype)
            elif other.dtype.kind == ".":
                other = other
            else:
                raise ValueError

            diff = self._dtype.decimal_places - other._dtype.decimal_places
            if diff >= 0:
                other = np.round(other._data * 10 ** diff).astype("int64")
                return op(self._data, other)
            else:
                these = np.round(self._data * 10 ** -diff).astype("int64")
                return op(these, other._data)
        return _binop

    @classmethod
    def _create_method(cls, op, coerce_to_dtype=True, result_dtype=None):
        def _binop(self, other):
            if isinstance(other, (ABCSeries, ABCIndex, ABCDataFrame)):
                # rely on pandas to unbox and dispatch to us
                return NotImplemented

            if not hasattr(other, "dtype"):
                other = np.asanyarray(other)
            if "add" in str(op) or "sub" in str(op):
                if other.dtype.kind in ["i", "f"]:
                    other = cls(other * 10**self._dtype.decimal_places, dtype=self.dtype)
                elif other.dtype.kind == ".":
                    other = other
                else:
                    raise ValueError

                diff = self._dtype.decimal_places - other._dtype.decimal_places
                if diff >= 0:
                    other = np.round(other._data * 10**diff).astype("int64")
                    return cls.from_internal(op(self._data, other), dtype=self._dtype)
                else:
                    these = np.round(self._data / 10**diff).astype("int64")
                    return cls.from_internal(op(these, other._data), dtype=other._dtype)
            elif "mul" in str(op) or "div" in str(op):
                if other.dtype.kind == ".":
                    other = other._data / 10**other._dtype.decimal_places
                elif other.dtype.kind not in ["i", "f"]:
                    raise ValueError
                return cls.from_internal(op(self._data, other), dtype=self._dtype)

        op_name = f"__{op.__name__}__"
        return set_function_name(_binop, op_name, cls)

    #def _reduce(self, name: str, *, skipna: bool = True, axis=None, **kwargs):
    #    return getattr(ak, name)(self._data, **kwargs)

    @property
    def dtype(self) -> DecimaldDtype:
        return self._dtype

    @property
    def nbytes(self) -> int:
        return self._data.nbytes

    def isna(self):
        return np.full(self._data.shape, False, dtype="bool")

    def take(self, indices, *, allow_fill=False, fill_value=None):
        return self[indices]

    def copy(self):
        return type(self)(np.copy(self._data), dtype=self._dtype)

    @classmethod
    def _concat_same_type(cls, to_concat):
        max_decimals = max(_._dtype.decimal_places for _ in to_concat)
        return cls(
            np.concatenate([
                _._data * 10**(_._dtype.decimal_places - max_decimals) for _ in to_concat
                ]),
            dtype=DecimaldDtype(max_decimals)
        )

    @property
    def ndim(self) -> Literal[1]:
        return 1

    @property
    def shape(self) -> tuple[int]:
        return self._data.shape

    def __array__(self, dtype=None) -> np.NDArray:
        return np.asarray(self._data / 10**self.dtype.decimal_places, dtype=dtype)

    def __arrow_array__(self):
        raise NotImplementedError

    def tolist(self) -> list:
        return (self._data / 10**self.dtype.decimal_places).tolist()

    def __array_ufunc__(self, *inputs, **kwargs):
        return type(self)(self._data.__array_ufunc__(*inputs, **kwargs), dtype=self._dtype)

    def max(self, **kwargs):
        kwargs.pop("min_count")
        kwargs.pop("skipna")
        return self._data.max(**kwargs) / 10**self.dtype.decimal_places

    def min(self, **kwargs):
        kwargs.pop("min_count")
        kwargs.pop("skipna")
        return self._data.min(**kwargs) / 10**self.dtype.decimal_places

    def mean(self, **kwargs):
        kwargs.pop("min_count")
        kwargs.pop("skipna")
        return self._data.mean(**kwargs) / 10**self.dtype.decimal_places

    def std(self, **kwargs):
        kwargs.pop("min_count")
        kwargs.pop("skipna")
        return self._data.std(**kwargs) / 10**self.dtype.decimal_places

    def sum(self, **kwargs):
        kwargs.pop("min_count")
        kwargs.pop("skipna")
        return self._data.sum(**kwargs) / 10**self.dtype.decimal_places


DecimalExtensionArray._add_arithmetic_ops()
DecimalExtensionArray._add_comparison_ops()
