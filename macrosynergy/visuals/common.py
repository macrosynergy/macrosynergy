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


def is_matching_subscripted_type(value: Any, type_hint: Type[Any]) -> bool:
    """
    Implementation of `insinstance()` for type-hints imported from the `typing` module,
    and for subscripted types (e.g. `List[int]`, `Tuple[str, int]`, etc.).
    Parameters
    :param <Any> value: The value to check.
    :param <Type[Any]> type_hint: The type hint to check against.
    :return <bool>: True if the value is of the type hint, False otherwise.
    """
    origin = get_origin(type_hint)
    args = get_args(type_hint)

    # handling lists
    if origin in [list, List]:
        if not isinstance(value, list):
            return False
        return all(isinstance(item, args[0]) for item in value)

    # tuples
    if origin in [tuple, Tuple]:
        if not isinstance(value, tuple) or len(value) != len(args):
            return False
        # don't switch order of get_origin and is_matching_subscripted_type, is short-circuiting
        return all(
            [
                (get_origin(expected) and is_matching_subscripted_type(item, expected))
                or isinstance(item, expected)
                for item, expected in zip(value, args)
            ]
        )

    # dicts
    if origin in [dict, Dict]:
        if not isinstance(value, dict):
            return False
        key_type, value_type = args
        return all(
            [
                (get_origin(key_type) and is_matching_subscripted_type(k, key_type))
                or isinstance(k, key_type)
                or (isinstance(k, key_type) and isinstance(v, value_type))
                for k, v in value.items()
            ]
        )

    # unions and optionals
    if origin is Union:
        for possible_type in args:
            if get_origin(possible_type):  # is subscripted
                if is_matching_subscripted_type(value, possible_type):
                    return True
            elif isinstance(value, possible_type):
                return True
        return False

    return False


def _get_expected(arg_type_hint: Type[Any]) -> List[str]:
    """
    Based on the type hint, return a list of strings that represent
    the type hint - including any nested type hints.
    Parameters
    :param <Type[Any]> arg_type_hint: The type hint to get the expected types for.
    :return <List[str]>: A list of strings that represent the type hint.
    """
    origin = get_origin(arg_type_hint)
    args = get_args(arg_type_hint)

    # handling lists
    if origin in [list, List]:
        return [f"List[{_get_expected(args[0])[0]}]"]

    # tuples
    if origin in [tuple, Tuple]:
        return [f"Tuple[{', '.join(_get_expected(arg) for arg in args)}]"]

    # dicts
    if origin in [dict, Dict]:
        return [f"Dict[{', '.join(_get_expected(arg) for arg in args)}]"]

    # unions and optionals
    if origin in [Union, Optional]:
        # get a flat list of all the expected types
        expected_types: List[str] = []
        for possible_type in args:
            if get_origin(possible_type):
                expected_types.extend(_get_expected(possible_type))
            else:
                expected_types.append(str(possible_type))
        return expected_types

    return [str(arg_type_hint)]


def argvalidation(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator for validating the arguments and return value of a function.
    Parameters
    :param <Callable[..., Any]> func: The function to validate.
    :return <Callable[..., Any]>: The decorated function.
    """

    def format_expected_type(expected_types: List[Any]) -> str:
        # format the expected types to read nicely, and to remove 'typing.' from the string
        if isinstance(expected_types, tuple):
            expected_types = list(expected_types)
        for i, et in enumerate(expected_types):
            if str(et).startswith("typing."):
                expected_types[i] = str(et).replace("typing.", "")
            if et is NoneType:
                expected_types[i] = "None"

        if len(expected_types) == 1:
            return f"`{expected_types[0]}`"
        elif len(expected_types) == 2:
            return f"`{expected_types[0]}` or `{expected_types[1]}`"
        else:
            return f"{', '.join([f'`{t}`' for t in expected_types[:-1]])}, or `{expected_types[-1]}`"

    @wraps(func)
    def validation_wrapper(*args: Any, **kwargs: Any) -> Any:
        func_sig: inspect.Signature = inspect.signature(func)
        func_params: Dict[str, inspect.Parameter] = func_sig.parameters
        func_annotations: Dict[str, Any] = func_sig.return_annotation
        func_args: Dict[str, Any] = inspect.getcallargs(func, *args, **kwargs)

        # validate the arguments
        for arg_name, arg_value in func_args.items():
            if arg_name in func_params:
                arg_type: Type[Any] = func_params[arg_name].annotation
                if arg_type is not inspect._empty:
                    origin = get_origin(arg_type)
                    if origin:  # Handling subscripted types
                        # replace 'float' with 'typng.Union[float, int]' to make life easier
                        if not is_matching_subscripted_type(arg_value, arg_type):
                            exp_types: str = format_expected_type(get_args(arg_type))
                            raise TypeError(
                                f"Argument `{arg_name}` must be of type {exp_types}, "
                                f"not `{type(arg_value).__name__}` (with value `{arg_value}`)."
                            )
                    else:  # For simple, non-generic types
                        if not isinstance(arg_value, arg_type):
                            raise TypeError(
                                f"Argument `{arg_name}` must be of type `{arg_type}`, "
                                f"not `{type(arg_value).__name__}` (with value `{arg_value}`)."
                            )

        # validate the return value
        return_value: Any = func(*args, **kwargs)
        if func_annotations is not inspect._empty:
            origin = get_origin(func_annotations)
            if (
                origin
                and (not is_matching_subscripted_type(return_value, func_annotations))
            ) or (not origin and not isinstance(return_value, func_annotations)):
                exp_types: str = format_expected_type(get_args(func_annotations))
                raise warnings.warn(
                    f"Return value of `{func.__name__}` is not of type "
                    f"`{func_annotations}`, but of type `{type(return_value)}`."
                )

        return return_value

    return validation_wrapper


def argcopy(func: Callable) -> Callable:
    """
    Decorator for applying a "pass-by-value" method to the arguments of a function.
    Parameters
    :param <Callable> func: The function to copy arguments for.
    :return <Callable>: The decorated function.
    """

    @wraps(func)
    def copy_wrapper(*args, **kwargs):
        copy_types = (
            list,
            dict,
            pd.DataFrame,
            np.ndarray,
            pd.Series,
            pd.Index,
            pd.MultiIndex,
            set,
        )
        new_args: List[Tuple[Any, ...]] = []
        for arg in args:
            if isinstance(arg, copy_types) or issubclass(type(arg), copy_types):
                new_args.append(arg.copy())
            else:
                new_args.append(arg)
        new_kwargs: Dict[str, Any] = {}
        for key, value in kwargs.items():
            if isinstance(value, copy_types) or issubclass(type(value), copy_types):
                new_kwargs[key] = value.copy()
            else:
                new_kwargs[key] = value

        return func(*new_args, **new_kwargs)

    return copy_wrapper
