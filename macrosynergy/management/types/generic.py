from ..decorators import argvalidation


class NoneTypeMeta(type):
    """
    MetaClass to support type checks for `None`.
    """

    def __instancecheck__(cls, instance):
        return instance is None or isinstance(instance, type(None))


class NoneType(metaclass=NoneTypeMeta):
    """
    Custom class definition for a NoneType that supports type checks for `None`.
    """

    pass


class SubscriptableMeta(type):
    """
    Convenience metaclass to allow subscripting of methods on a class.
    """

    def __getitem__(cls, item):
        if hasattr(cls, item) and callable(getattr(cls, item)):
            return getattr(cls, item)
        else:
            raise KeyError(f"{item} is not a valid method name")


class ArgValidationMeta(type):
    def __new__(cls, name, bases, dct: dict):
        for key, value in dct.items():
            if callable(value):
                dct[key] = argvalidation(value)
        return super().__new__(cls, name, bases, dct)
