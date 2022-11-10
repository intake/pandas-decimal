from __future__ import annotations
import decimal

import re

from typing import TYPE_CHECKING, Any

import numpy as np
from pandas.core.dtypes.base import ExtensionDtype, register_extension_dtype


@register_extension_dtype
class DecimaldDtype(ExtensionDtype):
    name = "decimal"
    _match = re.compile(r"decimal\[(\d+)\]")

    type: decimal.Decimal
    base = np.dtype("O")
    _metadata = ("freq",)

    def __new__(cls, decimal_places=0):
        instance = object.__new__(DecimaldDtype)
        instance.decimal_places = decimal_places
        return instance

    def __reduce__(self):
        return type(self), (self.decimal_places,)

    def __eq__(self, other):
        return other.kind == "." and other.decimal_places == self.decimal_places

    @property
    def name(self) -> str:
        return f"decimal[{self.decimal_places}]"

    @property
    def type(self) -> type[decimal.Decimal]:
        return decimal.Decimal

    @property
    def kind(self) -> str:
        return "."

    @property
    def na_value(self) -> object:
        return np.nan

    @property
    def _is_numeric(self) -> bool:
        return True

    @property
    def _is_boolean(self) -> bool:
        return True

    @classmethod
    def construct_from_string(cls, string: str) -> DecimaldDtype:
        """Construct an instance from a string.

        Parameters
        ----------
        string : str

        Returns
        -------
        DecimalDtype instance
        """

        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )

        if string == "decimal":
            return cls()
        out = re.match(cls._match, string)
        if not out:
            raise TypeError(f"Could not construct decimal dtype from {string}")
        return cls(int(out.groups()[0]))

    @classmethod
    def construct_array_type(cls) -> type:  # type: ignore[valid-type]
        from pandas_decimal.array import DecimalExtensionArray

        return DecimalExtensionArray

    def __from_arrow__(self, data: Any):
        from pandas_decimal.array import DecimalExtensionArray
        raise NotImplementedError
        return DecimalExtensionArray

    def __repr__(self) -> str:
        return f"<Decimal[{self.decimal_places}]>"
