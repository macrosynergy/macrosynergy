"""
Common types and functions used across the modules of the `macrosynergy.visuals` subpackage.
"""
import inspect
import warnings
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    SupportsFloat,
    SupportsInt,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
)

import numpy as np
import pandas as pd


class NumericType(type):
    """
    MetaClass to support type checks across `int`, `float`, `np.int64`, `np.float64`,
    `SupportsInt`, and `SupportsFloat`.
    """

    # A tuple of the desired types
    _numeric_types = (int, float, np.int64, np.float64, SupportsInt, SupportsFloat)

    def __instancecheck__(cls, instance):
        return isinstance(instance, cls._numeric_types)


class Numeric(metaclass=NumericType):
    """
    Custom class definition for a numeric type that supports type checks across `int`,
    `float`, `np.int64`, `np.float64`, `SupportsInt`, and `SupportsFloat`.
    """

    # Alternatively, use `numbers.Number` directly
    pass


class NoneType(type):
    """
    MetaClass to support type checks for `None`.
    """

    def __instancecheck__(cls, instance):
        return instance is None

