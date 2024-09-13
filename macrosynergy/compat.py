from . import PYTHON_3_8_OR_LATER

if PYTHON_3_8_OR_LATER:
    RESAMPLE_NUMERIC_ONLY = {"numeric_only": True}
    from sklearn.base import OneToOneFeatureMixin
else:
    RESAMPLE_NUMERIC_ONLY = {}
    from sklearn.base import _OneToOneFeatureMixin as OneToOneFeatureMixin
